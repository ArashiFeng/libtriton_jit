# ==============================================================================
# GEMM Examples
# ==============================================================================

# ------------------------------------------------------------------------------
# Helper function: Configure NPU-related settings for a target
# ------------------------------------------------------------------------------
function(configure_npu_target TARGET_NAME)
    target_link_ascend_libraries(${TARGET_NAME})
    target_compile_definitions(${TARGET_NAME} PRIVATE WITH_NPU)
    
    if(TORCH_NPU_INCLUDE_PATHS)
        target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
    endif()
    
    if(TORCH_NPU_BASE_PATH)
        target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_BASE_PATH})
    endif()
    
    if(TORCH_NPU_LIB)
        target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_NPU_LIB})
    endif()
    
    if(GFORTRAN_LIB)
        target_link_libraries(${TARGET_NAME} PRIVATE ${GFORTRAN_LIB})
    endif()
endfunction()

# ------------------------------------------------------------------------------
# Copy Python sources to build directory
# ------------------------------------------------------------------------------
add_custom_target(copy_triton_gemm_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/gemm.py
            ${CMAKE_CURRENT_BINARY_DIR}/gemm.py
    DEPENDS 
            ${CMAKE_CURRENT_SOURCE_DIR}/gemm.py
)

# ------------------------------------------------------------------------------
# gemm_op library
# ------------------------------------------------------------------------------
add_library(gemm_op SHARED gemm_op.cpp)

target_include_directories(gemm_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gemm_op
    PUBLIC  Torch::Torch
    PRIVATE TritonJIT::triton_jit
)

configure_npu_target(gemm_op)
add_dependencies(gemm_op copy_triton_gemm_src)

# ------------------------------------------------------------------------------
# test_gemm executable
# ------------------------------------------------------------------------------
add_executable(test_gemm test_gemm.cpp)

target_include_directories(test_gemm
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_gemm
    PRIVATE gemm_op
    PRIVATE Torch::Torch
    PRIVATE TritonJIT::triton_jit
)

configure_npu_target(test_gemm)
add_dependencies(test_gemm copy_triton_gemm_src)

