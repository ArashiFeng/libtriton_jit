# ==============================================================================
# Matmul Example - Multi-backend Support
# ==============================================================================

# ------------------------------------------------------------------------------
# Backend-specific Target Configuration
# ------------------------------------------------------------------------------
function(configure_backend_target TARGET_NAME)
    if(BACKEND STREQUAL "NPU")
        # ------------------------- NPU Backend --------------------------------
        target_link_ascend_libraries(${TARGET_NAME})

        if(TORCH_NPU_INCLUDE_PATHS)
            target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
        endif()

        if(TORCH_NPU_LIB)
            target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_NPU_LIB})
        endif()

    elseif(BACKEND STREQUAL "MUSA")
        # ------------------------- MUSA Backend -------------------------------
        target_link_musa_libraries(${TARGET_NAME})
        # MUSA backend uses pybind11 for torch_musa import
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)

    else()
        # ------------------------- CUDA / IX Backend --------------------------
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cuda_driver)

    endif()
endfunction()

# ------------------------------------------------------------------------------
# Copy Python Sources to Build Directory
# ------------------------------------------------------------------------------
add_custom_target(copy_triton_matmul_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/mm.py
            ${CMAKE_CURRENT_BINARY_DIR}/mm.py
    DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/mm.py
)

# ------------------------------------------------------------------------------
# mm_op Library
# ------------------------------------------------------------------------------
add_library(mm_example_op SHARED mm_op.cpp)

target_include_directories(mm_example_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(mm_example_op
    PUBLIC  Torch::Torch
    PRIVATE TritonJIT::triton_jit
)

configure_backend_target(mm_example_op)
add_dependencies(mm_example_op copy_triton_matmul_src)

# ------------------------------------------------------------------------------
# test_mm Executable
# ------------------------------------------------------------------------------
add_executable(test_mm_example test_mm.cpp)

target_include_directories(test_mm_example
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_mm_example
    PRIVATE mm_example_op
    PRIVATE TritonJIT::triton_jit
    PRIVATE Torch::Torch
)

configure_backend_target(test_mm_example)

# Force link ittnotify to resolve PyTorch's dependency
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_mm_example
        PRIVATE "-Wl,--no-as-needed"
        PRIVATE ${ITTNOTIFY_LIBRARY}
        PRIVATE "-Wl,--as-needed"
    )
endif()
add_dependencies(test_mm_example copy_triton_matmul_src)
