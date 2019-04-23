
add_custom_target(tests)
add_custom_target(flann_gtests)
add_custom_target(test)
add_custom_target(flann_gtest)

add_dependencies(tests flann_gtests)
add_dependencies(test flann_gtest)
add_dependencies(test tests)

set(EXECUTABLE_OUTPUT_PATH ${TEST_OUTPUT_PATH})

if (PYTHON_EXECUTABLE)
    flann_download_test_data(sift10K.h5 0964a910946d2dd5fe28337507a8abc3)
    flann_download_test_data(sift10K_byte.h5 f835e0148df4618a81f67febfda2b4d0)
    flann_download_test_data(sift100K.h5 ae2b08f93f3d9f89f5d68566b0406102)
    flann_download_test_data(sift100K_byte.h5 b772255fd2044e9d2a5a0183953e4705)
    flann_download_test_data(cloud.h5 dfc77bad391a3ae739a9874f4a5dc0d7)
    flann_download_test_data(brief100K.h5 e1e781c0955917bc2f0a27b6344c2342)
endif()

if (GTEST_FOUND AND HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})

    set(TEST_LIBRARIES "${HDF5_LIBRARIES}")
    if (HDF5_IS_PARALLEL)
        set(TEST_LIBRARIES "${TEST_LIBRARIES};${MPI_LIBRARIES}")
    endif()

    flann_add_gtest(flann_linear_test flann_linear_test.cpp)
    target_link_libraries(flann_linear_test flann_cpp ${TEST_LIBRARIES})

    flann_add_gtest(flann_kdtree_test flann_kdtree_test.cpp)
    target_link_libraries(flann_kdtree_test flann_cpp ${TEST_LIBRARIES})

    flann_add_gtest(flann_kmeans_test flann_kmeans_test.cpp)
    target_link_libraries(flann_kmeans_test flann_cpp ${TEST_LIBRARIES})

    if (BUILD_OpenCL_LIB)
        flann_add_gtest(flann_kmeans_opencl_test flann_kmeans_opencl_test.cpp)
        target_link_libraries(flann_kmeans_opencl_test flann_cpp ${TEST_LIBRARIES})
        target_link_libraries(flann_kmeans_opencl_test ${OpenCL_LIBRARIES})

        flann_add_gtest(flann_hierarchical_opencl_test flann_hierarchical_opencl_test.cpp)
        target_link_libraries(flann_hierarchical_opencl_test flann_cpp ${TEST_LIBRARIES})
        target_link_libraries(flann_hierarchical_opencl_test ${OpenCL_LIBRARIES})
    endif()

    flann_add_gtest(flann_kdtree_single_test flann_kdtree_single_test.cpp)
    target_link_libraries(flann_kdtree_single_test flann_cpp ${TEST_LIBRARIES})

    flann_add_gtest(flann_hierarchical_test flann_hierarchical_test.cpp)
    target_link_libraries(flann_hierarchical_test flann_cpp ${TEST_LIBRARIES})

    flann_add_gtest(flann_lsh_test flann_lsh_test.cpp)
    target_link_libraries(flann_lsh_test flann_cpp ${TEST_LIBRARIES})

    flann_add_gtest(flann_autotuned_test flann_autotuned_test.cpp)
    target_link_libraries(flann_autotuned_test flann_cpp ${TEST_LIBRARIES})

    if (OPENMP_FOUND)
        flann_add_gtest(flann_multithreaded_test flann_multithreaded_test.cpp)
        target_link_libraries(flann_multithreaded_test flann_cpp ${TEST_LIBRARIES})
    endif()

endif()

if (GTEST_FOUND AND HDF5_FOUND AND BUILD_CUDA_LIB)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC;-arch=sm_13" )
    if (NVCC_COMPILER_BINDIR)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--compiler-bindir=${NVCC_COMPILER_BINDIR}")
    endif()
    flann_add_cuda_gtest(flann_cuda_test flann_cuda_test.cu)
    target_link_libraries(flann_cuda_test flann_cpp ${HDF5_LIBRARIES} flann_cuda)
endif()

#---------- pyunit tests --------------
if (BUILD_PYTHON_BINDINGS)
    flann_add_pyunit(test_nn.py)
    flann_add_pyunit(test_nn_index.py)
    flann_add_pyunit(test_index_save.py)
    flann_add_pyunit(test_nn_autotune.py)
    flann_add_pyunit(test_clustering.py)
endif()

#---------- ruby spec ----------------
if (BUILD_C_BINDINGS)
    add_custom_target(flann_ruby_spec
                    COMMAND bundle exec rake spec
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/ruby)
    add_dependencies(test flann_ruby_spec)
endif()
