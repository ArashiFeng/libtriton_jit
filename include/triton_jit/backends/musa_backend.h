#pragma once

#include <musa.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit {


struct MusaKernelMetadata {
    unsigned int shared;
    // MUSA may have its own architecture identifier, but we'll keep this for compatibility
    unsigned int arch;
};


struct MusaBackend {
    using StreamType = MUstream;
    using ContextType = MUcontext;
    using KernelHandle = MUfunction;

    // MUSA warp size is 64 threads, as specified by the user.
    static constexpr unsigned int WARP_SIZE = 64;

    struct ModuleData {
        MUmodule module;
        MusaKernelMetadata metadata;
    };

    static inline std::unordered_map<std::string, ModuleData> module_cache_;
    static inline std::mutex cache_mutex_;

    static void launch_kernel(
        MUstream stream,
        MUfunction kernel,
        unsigned grid_x, unsigned grid_y, unsigned grid_z,
        unsigned block_x, unsigned block_y, unsigned block_z,
        void** args,
        unsigned int shared_memory = 0
    ) {
        LOG(INFO) << "muLaunchKernel";

        MUresult result = muLaunchKernel(
            kernel,
            grid_x, grid_y, grid_z,        // Grid dimensions
            block_x, block_y, block_z,     // Block dimensions
            shared_memory,                 // Shared memory
            stream,                        // Stream
            args,                          // Arguments
            nullptr                        // Extra
        );

        if (result != MUSA_SUCCESS) {
            const char* error_string;
            muGetErrorString(result, &error_string);
            throw std::runtime_error(
                fmt::format("MUSA kernel launch failed: {}", error_string)
            );
        }
    }

    static void ensure_context() {
        // When using PyTorch with a MUSA backend, the context is typically already initialized.
        MUcontext ctx;
        MUresult result = muCtxGetCurrent(&ctx);

        if (result != MUSA_SUCCESS || ctx == nullptr) {
            LOG(WARNING) << "No MUSA context found. Creating default context.";
            MUdevice device;
            checkMusaErrors(muDeviceGet(&device, 0));
            checkMusaErrors(muCtxCreate(&ctx, 0, device));
        }
    }

    static int get_device_index() {
        MUdevice device;
        MUresult result = muCtxGetDevice(&device);

        if (result != MUSA_SUCCESS) {
            const char* error_string;
            muGetErrorString(result, &error_string);
            throw std::runtime_error(
                fmt::format("Failed to get MUSA device: {}", error_string)
            );
        }
        return static_cast<int>(device);
    }

    static MUfunction load_kernel(
        const std::string& dir,
        const std::string& kernel_name
    ) {
        std::string key = fmt::format("{}::{}", dir, kernel_name);
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = module_cache_.find(key);
        if (it != module_cache_.end()) {
            MUfunction kernel;
            checkMusaErrors(muModuleGetFunction(
                &kernel, it->second.module, kernel_name.c_str()));
            return kernel;
        }

        // Load metadata from .json file
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);
        if (!f.is_open()) {
            throw std::runtime_error(
                fmt::format("Failed to open metadata file: {}", metadata_path));
        }

        nlohmann::json meta_data = nlohmann::json::parse(f);
        MusaKernelMetadata metadata;
        metadata.shared = meta_data["shared"];
        
        LOG(INFO) << fmt::format(
            "Loading MUSA kernel {} with shared_mem={}",
            kernel_name, metadata.shared);

        // Try to load pre-compiled binaries in priority order: .o (ELF), .so, .llir
        MUmodule module = nullptr;
        std::string obj_path = fmt::format("{}/{}.o", dir, kernel_name);
        std::string so_path = fmt::format("{}/{}.so", dir, kernel_name);
        std::string llir_path = fmt::format("{}/{}.llir", dir, kernel_name);

        if (std::filesystem::exists(obj_path)) {
            LOG(INFO) << fmt::format("Loading MUSA object file from {} using muModuleLoadData", obj_path);

            // Read .o file as binary data
            std::ifstream obj_file(obj_path, std::ios::binary);
            if (!obj_file.is_open()) {
                throw std::runtime_error(
                    fmt::format("Failed to open object file: {}", obj_path));
            }

            std::vector<char> obj_data((std::istreambuf_iterator<char>(obj_file)),
                                        std::istreambuf_iterator<char>());

            LOG(INFO) << fmt::format("Loaded {} bytes from {}", obj_data.size(), obj_path);

            // Use muModuleLoadData to load the ELF object directly
            checkMusaErrors(muModuleLoadData(&module, obj_data.data()));

        } else if (std::filesystem::exists(so_path)) {
            LOG(INFO) << fmt::format("Loading MUSA shared library from {}", so_path);
            checkMusaErrors(muModuleLoad(&module, so_path.c_str()));

        } else if (std::filesystem::exists(llir_path)) {
            LOG(INFO) << fmt::format("Loading MUSA LLIR from {} (runtime JIT)", llir_path);

            // Read LLIR file
            std::ifstream llir_file(llir_path, std::ios::binary);
            if (!llir_file.is_open()) {
                throw std::runtime_error(
                    fmt::format("Failed to open LLIR file: {}", llir_path));
            }

            std::string llir_code((std::istreambuf_iterator<char>(llir_file)),
                                   std::istreambuf_iterator<char>());

            // Use muModuleLoadData for runtime JIT compilation
            checkMusaErrors(muModuleLoadData(&module, llir_code.c_str()));

        } else {
            throw std::runtime_error(
                fmt::format("No binary (.o, .so) or LLIR found for kernel {} in {}",
                            kernel_name, dir));
        }

        // Get function handle
        MUfunction kernel;
        checkMusaErrors(muModuleGetFunction(&kernel, module, kernel_name.c_str()));

        // Cache the loaded module and metadata
        module_cache_[key] = ModuleData{module, metadata};

        return kernel;
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

        // If not in cache, load from metadata file directly
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);
        if (!f.is_open()) {
            LOG(WARNING) << "Could not open metadata file to get shared memory: " << metadata_path;
            return 0;
        }
        nlohmann::json meta_data = nlohmann::json::parse(f);
        return meta_data["shared"];
    }
};

static_assert(BackendPolicy<MusaBackend>, "MusaBackend must satisfy BackendPolicy concept");

} // namespace triton_jit
