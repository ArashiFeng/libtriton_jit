
# copy python code to the binary dir to resolve path issues
# this is just for the purpose of this example
# if we are buiding a library that use TritonJITFunction, make sure
# that it finds the python code, whether the library is in the build tree or
# the install tree
add_custom_target(
    copy_triton_reduce_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_compile.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum_compile.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_triton_cpp_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum_triton_cpp_rt.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_cpp_rt_vs_python_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum_cpp_rt_vs_python_rt.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sum.py
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_compile.py
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_triton_cpp_rt.py
)

add_library(sum_op SHARED sum_op.cpp)

if (LIBTRITON_JIT_BACKEND STREQUAL "mthreads")
  set(BACKEND_PYTHON_INCLUDES "/usr/include/python3.10;/usr/local/lib/python3.10/dist-packages") 
  set(BACKEND_PYTHON_LIBS "/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so")
endif()

target_include_directories(sum_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${BACKEND_PYTHON_INCLUDES}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(sum_op
    PRIVATE TritonJIT::triton_jit
    PUBLIC Torch::Torch ${BACKEND_PYTHON_LIBS}
)
add_dependencies(sum_op copy_triton_reduce_src)

# add_executable(test_sum test_sum.cpp)
# target_link_libraries(test_sum
#     PRIVATE sum_op Torch::Torch
# )
# add_dependencies(test_sum copy_triton_reduce_src)
