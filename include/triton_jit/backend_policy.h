#pragma once

#include <concepts>
#include <string>
#include <type_traits>

namespace triton_jit {

template<typename T>
concept BackendPolicy = requires {
    typename T::StreamType;
    typename T::ContextType;
    typename T::KernelHandle;

} && requires(
    typename T::StreamType stream,
    typename T::KernelHandle kernel,
    unsigned grid_x, unsigned grid_y, unsigned grid_z,
    unsigned block_x, unsigned block_y, unsigned block_z,
    void** args
) {

    { T::launch_kernel(stream, kernel,
                       grid_x, grid_y, grid_z,
                       block_x, block_y, block_z,
                       args) } -> std::same_as<void>;

    { T::ensure_context() } -> std::same_as<void>;

    { T::get_device_index() } -> std::same_as<int>;

} && requires(const std::string& dir, const std::string& name) {
    
    { T::load_kernel(dir, name) } -> std::same_as<typename T::KernelHandle>;

    { T::get_shared_memory(dir, name) } -> std::same_as<unsigned int>;
};

/**
 * @brief Helper concept: Check if a type has StreamType
 */
template<typename T>
concept HasStreamType = requires {
    typename T::StreamType;
};

/**
 * @brief Helper concept: Check if a type has ContextType
 */
template<typename T>
concept HasContextType = requires {
    typename T::ContextType;
};

/**
 * @brief Helper concept: Check if a type has KernelHandle
 */
template<typename T>
concept HasKernelHandle = requires {
    typename T::KernelHandle;
};

} // namespace triton_jit
