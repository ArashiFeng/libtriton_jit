# ==============================================================================
# experiments/npu_min_vector - Test NPU minimum vector length
# ==============================================================================

# Copy the add kernel Python file
add_custom_target(copy_experiment_add_python
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PROJECT_SOURCE_DIR}/operators/pointwise/add/add.py
            ${CMAKE_CURRENT_BINARY_DIR}/add.py
    DEPENDS ${PROJECT_SOURCE_DIR}/operators/pointwise/add/add.py
)

# Test executable - link to existing add_op from operators
add_executable(test_min_vector test_min_vector.cpp)

target_include_directories(test_min_vector
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/operators/pointwise/add
)

target_link_libraries(test_min_vector
    PRIVATE add_op  # Reuse add_op from operators/pointwise/add
    PRIVATE TritonJIT::triton_jit
    PRIVATE Torch::Torch
)

# NPU-specific configuration using helper function
if(BACKEND STREQUAL "NPU")
    target_link_ascend_libraries(test_min_vector)
endif()

# Force link ittnotify if available
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_min_vector
        PRIVATE "-Wl,--no-as-needed"
        PRIVATE ${ITTNOTIFY_LIBRARY}
        PRIVATE "-Wl,--as-needed"
    )
endif()

add_dependencies(test_min_vector copy_experiment_add_python)
