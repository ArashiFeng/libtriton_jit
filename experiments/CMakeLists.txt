# ==============================================================================
# experiments - Experimental tests and benchmarks
# ==============================================================================

# Only build experiments for NPU backend
if(BACKEND STREQUAL "NPU")
    add_subdirectory(npu_min_vector)
endif()
