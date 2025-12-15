#pragma once

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "acl/acl.h"
#include "experiment/runtime/runtime/rt.h"
#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"  // for checkAclErrors

namespace triton_jit {

struct NpuKernelMetadata {
    unsigned int shared;
    std::string mix_mode;
};

struct NpuBackend {
    using StreamType = aclrtStream;
    using ContextType = aclrtContext;
    using KernelHandle = void*;

    // NPU does not use warp concept, but we need a non-zero value for block size calculation
    static constexpr unsigned int WARP_SIZE = 1;

    struct ModuleData {
        void* bin_handle;
        void* fn_handle;
        NpuKernelMetadata metadata;
    };

    static inline std::unordered_map<std::string, ModuleData> module_cache_;
    static inline std::mutex cache_mutex_;
    // Static storage for function stubs
    static inline std::unordered_map<std::string, size_t> registered_names_;
    static inline std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs_;

    static void launch_kernel(
        aclrtStream stream,
        void* kernel,
        unsigned grid_x, unsigned grid_y, unsigned grid_z,
        unsigned block_x, unsigned block_y, unsigned block_z,
        void** args,
        unsigned int shared_memory = 0
    ) {
        // Calculate block count
        uint32_t blockNum = grid_x * grid_y * grid_z;

        // Get system control address
        rtError_t ret;
        void* ffts_addr = nullptr;
        uint32_t ffts_len;
        ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
        if (ret != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtGetC2cCtrlAddr failed: {}",
                                                static_cast<int>(ret)));
        }

        // Build kernel arguments structure for ADD operation
        // Layout: [system args] + [a, b, out, n] + [grid dims]
        // Note: tile_size is constexpr, not passed at runtime
        struct __attribute__((packed)) KernelArgs {
            void* ffts_addr __attribute__((aligned(8)));
            void* syncBlockLock __attribute__((aligned(8)));
            void* workspace_addr __attribute__((aligned(8)));
            void* arg0 __attribute__((aligned(8)));  // a (input tensor)
            void* arg1 __attribute__((aligned(8)));  // b (input tensor)
            void* arg2 __attribute__((aligned(8)));  // out (output tensor)
            int64_t arg3 __attribute__((aligned(8)));  // n (number of elements)
            int32_t gridX __attribute__((aligned(4)));
            int32_t gridY __attribute__((aligned(4)));
            int32_t gridZ __attribute__((aligned(4)));
        };

        KernelArgs kernel_args;
        kernel_args.ffts_addr = ffts_addr;
        kernel_args.syncBlockLock = nullptr;
        kernel_args.workspace_addr = nullptr;

        if (args != nullptr) {
            // Extract arguments from void** array
            // Add kernel signature: (a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE:constexpr)
            kernel_args.arg0 = (args[0] != nullptr) ? *reinterpret_cast<void**>(args[0]) : nullptr;
            kernel_args.arg1 = (args[1] != nullptr) ? *reinterpret_cast<void**>(args[1]) : nullptr;
            kernel_args.arg2 = (args[2] != nullptr) ? *reinterpret_cast<void**>(args[2]) : nullptr;
            kernel_args.arg3 = (args[3] != nullptr) ? *reinterpret_cast<int64_t*>(args[3]) : 1024;
        } else {
            LOG(WARNING) << "launch_kernel: args is nullptr!";
            kernel_args.arg0 = nullptr;
            kernel_args.arg1 = nullptr;
            kernel_args.arg2 = nullptr;
            kernel_args.arg3 = 1024;
        }

        kernel_args.gridX = static_cast<int32_t>(grid_x);
        kernel_args.gridY = static_cast<int32_t>(grid_y);
        kernel_args.gridZ = static_cast<int32_t>(grid_z);

        // Launch kernel
        rtError_t rt_err = rtKernelLaunch(kernel,
                                          blockNum,
                                          static_cast<void*>(&kernel_args),
                                          sizeof(kernel_args),
                                          nullptr,
                                          stream);

        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtKernelLaunch failed: {}",
                                                static_cast<int>(rt_err)));
        }
    }

    static void ensure_context() {
        aclrtContext ctx;
        aclError ret = aclrtGetCurrentContext(&ctx);

        if (ret != ACL_ERROR_NONE || ctx == nullptr) {
            LOG(WARNING) << "No ACL context found. Creating default context.";
            int deviceId = 0;
            aclError err = aclrtSetDevice(deviceId);
            if (err != ACL_ERROR_NONE) {
                throw std::runtime_error(fmt::format("aclrtSetDevice failed: {}",
                                                    static_cast<int>(err)));
            }
            err = aclrtCreateContext(&ctx, deviceId);
            if (err != ACL_ERROR_NONE) {
                throw std::runtime_error(fmt::format("aclrtCreateContext failed: {}",
                                                    static_cast<int>(err)));
            }
            err = aclrtSetCurrentContext(ctx);
            if (err != ACL_ERROR_NONE) {
                throw std::runtime_error(fmt::format("aclrtSetCurrentContext failed: {}",
                                                    static_cast<int>(err)));
            }
        }
    }

    static int get_device_index() {
        int device_id = -1;
        aclError err = aclrtGetDevice(&device_id);

        if (err != ACL_ERROR_NONE) {
            throw std::runtime_error(fmt::format("Failed to get NPU device: {}",
                                                static_cast<int>(err)));
        }

        return device_id;
    }

    static void* load_kernel(
        const std::string& dir,
        const std::string& kernel_name
    ) {
        std::string key = fmt::format("{}::{}", dir, kernel_name);

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check cache first
        auto it = module_cache_.find(key);
        if (it != module_cache_.end()) {
            return it->second.fn_handle;
        }

        // Load metadata
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);

        NpuKernelMetadata metadata;
        metadata.shared = 0;
        metadata.mix_mode = "mix";

        if (f.is_open()) {
            nlohmann::json meta_data = nlohmann::json::parse(f);
            metadata.shared = meta_data.contains("shared") ? meta_data["shared"].get<int>() : 0;
            metadata.mix_mode = meta_data.contains("mix_mode") ?
                               meta_data["mix_mode"].get<std::string>() : "mix";
        }

        LOG(INFO) << fmt::format(
            "Loading NPU kernel {} with mix_mode={}, shared={}",
            kernel_name, metadata.mix_mode, metadata.shared);

        // Find kernel binary file (try .npubin, .o, .ttadapter, .bin)
        std::string rt_bin_path = fmt::format("{}/{}.npubin", dir, kernel_name);
        std::ifstream bin_file(rt_bin_path, std::ios::binary | std::ios::ate);

        if (!bin_file.good()) {
            std::vector<std::string> fallback_exts = {".o", ".ttadapter", ".bin"};
            bool file_found = false;

            for (const auto& ext : fallback_exts) {
                rt_bin_path = fmt::format("{}/{}{}", dir, kernel_name, ext);
                bin_file.open(rt_bin_path, std::ios::binary | std::ios::ate);
                if (bin_file.good()) {
                    file_found = true;
                    break;
                }
                bin_file.close();
                bin_file.clear();
            }

            if (!file_found) {
                throw std::runtime_error(fmt::format("Kernel binary not found: {}/{}",
                                                    dir, kernel_name));
            }
        }

        // Read binary file
        std::streamsize size = bin_file.tellg();
        if (size <= 0) {
            throw std::runtime_error(fmt::format("Invalid binary size: {}", rt_bin_path));
        }

        bin_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));

        if (!bin_file.read(buffer.data(), size)) {
            throw std::runtime_error(fmt::format("Failed to read binary: {}", rt_bin_path));
        }
        bin_file.close();

        LOG(INFO) << fmt::format("Loading NPU binary from {}, size={}", rt_bin_path, size);

        // Get current device ID
        int device_id = -1;
        aclError err = aclrtGetDevice(&device_id);
        if (err != ACL_SUCCESS) {
            device_id = 0;  // fallback
        }

        // Set device
        rtError_t rt_err = rtSetDevice(device_id);
        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtSetDevice failed for device {}, error: {}",
                                                device_id, static_cast<int>(rt_err)));
        }

        // Register binary with RT API
        rtDevBinary_t binary;
        binary.data = buffer.data();
        binary.length = static_cast<uint32_t>(size);

        // Set magic value based on mix_mode
        binary.magic = (metadata.mix_mode == "aiv") ? RT_DEV_BINARY_MAGIC_ELF_AIVEC : RT_DEV_BINARY_MAGIC_ELF;
        binary.version = 0;

        void* rt_bin_handle = nullptr;
        rt_err = rtDevBinaryRegister(&binary, &rt_bin_handle);
        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtDevBinaryRegister failed: {}",
                                                static_cast<int>(rt_err)));
        }

        // Create function stub with unique name
        std::string stubName = kernel_name;
        stubName += "_" + std::to_string(registered_names_[kernel_name]);
        registered_names_[kernel_name]++;

        auto registered = func_stubs_.emplace(stubName, std::make_unique<size_t>(0));
        void* func_stub_handle = registered.first->second.get();

        // Register function
        rt_err = rtFunctionRegister(rt_bin_handle,
                                   func_stub_handle,
                                   stubName.c_str(),
                                   (void*)kernel_name.c_str(),
                                   0);
        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtFunctionRegister failed: {}",
                                                static_cast<int>(rt_err)));
        }

        // Cache the module
        module_cache_[key] = ModuleData{rt_bin_handle, func_stub_handle, metadata};

        return func_stub_handle;
    }

    static unsigned int get_shared_memory(
        const std::string& dir,
        const std::string& kernel_name
    ) {
        std::string key = fmt::format("{}::{}", dir, kernel_name);
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = module_cache_.find(key);
        if (it != module_cache_.end()) {
            return it->second.metadata.shared;
        }

        // If not in cache, load metadata
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);
        if (!f.is_open()) {
            return 0;
        }

        nlohmann::json meta_data = nlohmann::json::parse(f);
        return meta_data.contains("shared") ? meta_data["shared"].get<unsigned int>() : 0;
    }
};

static_assert(BackendPolicy<NpuBackend>, "NpuBackend must satisfy BackendPolicy concept");

} // namespace triton_jit
