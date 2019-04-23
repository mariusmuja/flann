#include_directories(${CMAKE_SOURCE_DIR}/include algorithms ext util nn .)

add_definitions(-D_FLANN_VERSION=${FLANN_VERSION})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/flann/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/flann/config.h)

file(GLOB_RECURSE C_SOURCES flann.cpp)
file(GLOB_RECURSE CPP_SOURCES flann_cpp.cpp)
file(GLOB_RECURSE CU_SOURCES *.cu)

add_library(flann_cpp_s STATIC ${CPP_SOURCES})
target_link_libraries(flann_cpp_s ${LZ4_LIBRARIES})
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    set_target_properties(flann_cpp_s PROPERTIES COMPILE_FLAGS -fPIC)
endif()
set_property(TARGET flann_cpp_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC FLANN_USE_CUDA)

if (BUILD_CUDA_LIB)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DFLANN_USE_CUDA")
    if(CMAKE_COMPILER_IS_GNUCC)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC;" )
        if (NVCC_COMPILER_BINDIR)
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--compiler-bindir=${NVCC_COMPILER_BINDIR}")
        endif()
    else()
	    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};" )
    endif()
    cuda_add_library(flann_cuda_s STATIC ${CU_SOURCES})
    set_property(TARGET flann_cuda_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_COMPILER_IS_GNUCC)
    add_library(flann_cpp SHARED "")
    set_target_properties(flann_cpp PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(flann_cpp -Wl,-whole-archive flann_cpp_s -Wl,-no-whole-archive)

    if (BUILD_CUDA_LIB)
	    cuda_add_library(flann_cuda SHARED "")
        set_target_properties(flann_cuda PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(flann_cuda -Wl,-whole-archive flann_cuda_s -Wl,-no-whole-archive)
        set_property(TARGET flann_cpp_s PROPERTY COMPILE_DEFINITIONS FLANN_USE_CUDA)
    # 	target_link_libraries(flann_cuda cudpp_x86_64)
    endif()
else()
    add_library(flann_cpp SHARED ${CPP_SOURCES})
    target_link_libraries(flann_cpp -Wl,--push-state,--no-as-needed ${LZ4_LIBRARIES} -Wl,--pop-state)
    # export lz4 headers, so that MSVC to creates flann_cpp.lib
    set_target_properties(flann_cpp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS YES)
    if (BUILD_CUDA_LIB)
		cuda_add_library(flann_cuda SHARED ${CPP_SOURCES})
        set_property(TARGET flann_cpp PROPERTY COMPILE_DEFINITIONS FLANN_USE_CUDA)
    endif()
endif()

set_target_properties(flann_cpp PROPERTIES
   VERSION ${FLANN_VERSION}
   SOVERSION ${FLANN_SOVERSION}
   DEFINE_SYMBOL FLANN_EXPORTS
)

if (BUILD_CUDA_LIB)
    set_target_properties(flann_cuda PROPERTIES
       VERSION ${FLANN_VERSION}
       SOVERSION ${FLANN_SOVERSION}
       DEFINE_SYMBOL FLANN_EXPORTS
    )
endif()


if (USE_MPI AND HDF5_IS_PARALLEL)
    add_executable(flann_mpi_server flann/mpi/flann_mpi_server.cpp)
    target_link_libraries(flann_mpi_server flann_cpp ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})

    add_executable(flann_mpi_client flann/mpi/flann_mpi_client.cpp)
    target_link_libraries(flann_mpi_client flann_cpp ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})

    install (TARGETS flann_mpi_client flann_mpi_server
        DESTINATION bin)
endif()


if (BUILD_C_BINDINGS)
    add_library(flann_s STATIC ${C_SOURCES})
    target_link_libraries(flann_s -Wl,--push-state,--no-as-needed ${LZ4_LIBRARIES} -Wl,--pop-state)
    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
        set_target_properties(flann_s PROPERTIES COMPILE_FLAGS -fPIC)
    endif()
    set_property(TARGET flann_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_COMPILER_IS_GNUCC)
        add_library(flann SHARED "")
        set_target_properties(flann PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(flann -Wl,-whole-archive flann_s -Wl,-no-whole-archive)
    else()
        add_library(flann SHARED ${C_SOURCES})
        target_link_libraries(flann -Wl,--push-state,--no-as-needed ${LZ4_LIBRARIES} -Wl,--pop-state)

        if(MINGW AND OPENMP_FOUND)
          target_link_libraries(flann gomp)
        endif()
    endif()

    set_target_properties(flann PROPERTIES
       VERSION ${FLANN_VERSION}
       SOVERSION ${FLANN_SOVERSION}
       DEFINE_SYMBOL FLANN_EXPORTS
    )
endif()

if (BUILD_OpenCL_LIB)
    set_property(TARGET flann_cpp PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    set_property(TARGET flann_cpp_s PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    set_property(TARGET flann PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    set_property(TARGET flann_s PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    target_link_libraries(flann_cpp ${OpenCL_LIBRARIES})
    target_link_libraries(flann_cpp_s ${OpenCL_LIBRARIES})
    target_link_libraries(flann ${OpenCL_LIBRARIES})
    target_link_libraries(flann_s ${OpenCL_LIBRARIES})
endif()

if(WIN32)
if (BUILD_C_BINDINGS AND BUILD_MATLAB_BINDINGS)
    install (
        TARGETS flann
        RUNTIME DESTINATION share/flann/matlab
    )
endif()
endif(WIN32)


install (
    TARGETS flann_cpp flann_cpp_s
    EXPORT ${targets_export_name}
    INCLUDES DESTINATION include
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION ${FLANN_LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${FLANN_LIB_INSTALL_DIR}
)

if (BUILD_CUDA_LIB)
    install (
        TARGETS flann_cuda flann_cuda_s
        EXPORT ${targets_export_name}
        INCLUDES DESTINATION include
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION ${FLANN_LIB_INSTALL_DIR}
        ARCHIVE DESTINATION ${FLANN_LIB_INSTALL_DIR}
    )
endif()

if (BUILD_C_BINDINGS)
    install (
        TARGETS flann flann_s
        EXPORT ${targets_export_name}
        INCLUDES DESTINATION include
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION ${FLANN_LIB_INSTALL_DIR}
        ARCHIVE DESTINATION ${FLANN_LIB_INSTALL_DIR}
    )
endif()

install (
    DIRECTORY flann
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
