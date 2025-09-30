
add_custom_target(
    copy_triton_pointwise_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_BINARY_DIR}/add.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/add_triton_cpp_rt.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add_cpp_rt_vs_python_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/add_cpp_rt_vs_python_rt.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
)

add_library(add_op SHARED add_op.cpp)

if (LIBTRITON_JIT_BACKEND STREQUAL "mthreads")
  set(BACKEND_PYTHON_INCLUDES "/usr/include/python3.10;/usr/local/lib/python3.10/dist-packages") 
  set(BACKEND_PYTHON_LIBS "/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so")
endif()

target_include_directories(add_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${BACKEND_PYTHON_INCLUDES}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(add_op
    PUBLIC Torch::Torch
    PRIVATE TritonJIT::triton_jit ${BACKEND_PYTHON_LIBS}
)
add_dependencies(add_op copy_triton_pointwise_src)

# add_executable(test_add test_add.cpp)
# target_link_libraries(test_add
#     PRIVATE add_op Torch::Torch)
# add_dependencies(test_add copy_triton_pointwise_src)
