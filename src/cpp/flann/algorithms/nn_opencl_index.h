/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2017  Seth Price (seth@planet.com). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/


#ifdef FLANN_USE_OPENCL
#ifndef FLANN_NN_OPENCL_INDEX_H_
#define FLANN_NN_OPENCL_INDEX_H_

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/opencl.h>
#endif

#include <cassert>
#include <limits.h>
#include <float.h>

namespace flann
{

template <typename T>
class TypeInfo {
public:
    static std::string clName() { return "unknown"; }
    static size_t size() { return 0; }
    static double max() { return 0; } \
};
#define TYPE_NAME_AND_SIZE(T, TN, TM) template <> \
class TypeInfo<T> { \
public: \
    static std::string clName() { return TN; } \
    static size_t size() { return sizeof(T); } \
    static double max() { return TM; } \
};

TYPE_NAME_AND_SIZE(float, "float", FLT_MAX);
TYPE_NAME_AND_SIZE(double, "double", DBL_MAX);
TYPE_NAME_AND_SIZE(bool, "bool", 1);
TYPE_NAME_AND_SIZE(char, "char", CHAR_MAX);
TYPE_NAME_AND_SIZE(unsigned char, "uchar", UCHAR_MAX);
TYPE_NAME_AND_SIZE(short, "short", SHRT_MAX);
TYPE_NAME_AND_SIZE(unsigned short, "ushort", USHRT_MAX);
TYPE_NAME_AND_SIZE(int, "int", INT_MAX);
TYPE_NAME_AND_SIZE(unsigned int, "uint", UINT_MAX);
TYPE_NAME_AND_SIZE(long, "long", LONG_MAX);
TYPE_NAME_AND_SIZE(unsigned long, "ulong", ULONG_MAX);
#undef TYPE_NAME_AND_SIZE

#define HandleFLANNErr(err) if ((err) != CL_SUCCESS) { \
    throw FLANNException("OpenCL error."); \
}

class OpenCLIndex
{
public:
    OpenCLIndex()
    {
        own_cq_ = false;
        cl_knn_search_kern_ = NULL;
        cl_cmd_queue_ = NULL;
    }

    virtual ~OpenCLIndex()
    {
        if (cl_knn_search_kern_)
        {
            clReleaseKernel(cl_knn_search_kern_);
            cl_knn_search_kern_ = NULL;
        }

        if (cl_cmd_queue_ && own_cq_)
        {
            cl_context context = NULL;
            cl_int err = CL_SUCCESS;
            // Get the context so we can release it later
            err = clGetCommandQueueInfo(cl_cmd_queue_, CL_QUEUE_CONTEXT,
                                        sizeof(context), &context, NULL);
            HandleFLANNErr(err);

            clReleaseCommandQueue(cl_cmd_queue_);
            cl_cmd_queue_ = NULL;

            clReleaseContext(context);
        }
    }

    void swap(OpenCLIndex& other)
    {
        std::swap(own_cq_, other.own_cq_);
        std::swap(cl_knn_search_kern_, other.cl_knn_search_kern_);
        std::swap(cl_kern_heap_size_, other.cl_kern_heap_size_);
        std::swap(cl_cmd_queue_, other.cl_cmd_queue_);
    }

    /**
     *
     */
    void buildCLKnnSearch(size_t knn,
                          const SearchParams& params,
                          cl_command_queue cq = NULL)
    {
        cl_int err = CL_SUCCESS;
        if (!cq && !cl_cmd_queue_)
        {
            own_cq_ = true;

            // Make a default command queue
            cl_device_id device_id;
            err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
            HandleFLANNErr(err);
            cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
            HandleFLANNErr(err);
#ifdef CL_API_SUFFIX__VERSION_2_0
            cq = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
#else // CL_API_SUFFIX__VERSION_2_0
            cq = clCreateCommandQueue(context, device_id, 0, &err);
#endif // CL_API_SUFFIX__VERSION_2_0
            HandleFLANNErr(err);
        }

        if (cl_cmd_queue_ != NULL && cq != NULL && cl_cmd_queue_ != cq) {
            // Using a new command queue, reset everything!
            freeCLIndexMem();
            if (cl_knn_search_kern_)
                clReleaseKernel(cl_knn_search_kern_);
            cl_cmd_queue_ = NULL;
        }

        if (!cl_cmd_queue_)
            cl_cmd_queue_ = cq;

        // Get the useful OpenCL environment info
        cl_context context;
        assert(cl_cmd_queue_);
        err = clGetCommandQueueInfo(cl_cmd_queue_, CL_QUEUE_CONTEXT, sizeof(context), &context, NULL);
        HandleFLANNErr(err);
        cl_device_id dev;
        err = clGetCommandQueueInfo(cl_cmd_queue_, CL_QUEUE_DEVICE, sizeof(dev), &dev, NULL);
        HandleFLANNErr(err);

        // Init the cl_mem index
        initCLIndexMem(context);

        // Update kernel if needed
        if (!cl_knn_search_kern_ || !clParamsMatch(knn, params)) {
            // Free old kernel first
            if (cl_knn_search_kern_)
                clReleaseKernel(cl_knn_search_kern_);

            // Build new one
            cl_knn_search_kern_ = buildCLknnSearchKernel(
                context, dev, knn, params.checks, &cl_kern_heap_size_);
            cl_kern_knn_ = knn;
            cl_kern_max_checks_ = params.checks;
        }
    }

protected:

    virtual bool clParamsMatch(size_t knn, const SearchParams& params) const
    {
        // unimplemented in nn_index
        printf("in unimplemented clParamsMatch()!\n");
        return false;
    }

    /**
     * Format the index for, and copy the index into, OpenCL device memory.
     * The device memory format is not intended to be changeable, so if the
     * index changes, then the index device memory should be freed and rebuilt.
     *
     * @param[in] context The OpenCL context to use.
     */
    virtual void initCLIndexMem(cl_context context)
    {
        // unimplemented in nn_index
        printf("in unimplemented initCLIndexMem()!\n");
    }

    /**
     * Build the kernel for knnSearch() using the given params and save it.
     * Future calls to knnSearch() that use those params are elgible for OpenCL
     * accelleration.
     *
     * @param[in] context The OpenCL context to use.
     * @param[in] dev The device we're compiling for.
     * @param[in] knn The number of nearest neighbors we're expecting to retrieve.
     * @param[in] maxChecks The maximum number of checks we're expecting to make.
     * @return The compiled OpenCL kernel.
     */
    virtual cl_kernel buildCLknnSearchKernel(cl_context context, cl_device_id dev,
                                             size_t knn, int maxChecks, int *heapSize)
    {
        // unimplemented in nn_index
        printf("in unimplemented buildCLknnSearchKernel()!\n");
        return NULL;
    }

    /**
     * Frees any OpenCL memory that has been allocated, and dependencies such
     * as kernel(s). To be called when the index changes, and thus must be
     * rebuilt or as a destructor.
     */
    virtual void freeCLIndexMem()
    {
        // unimplemented in nn_index
        printf("in unimplemented freeCLIndexMem()!\n");
    }

    /**
     * The number of threads and local size to use for the number of queries.
     * Should be a power of two, minimum of 32 on GPU, and within work group size.
     *
     * @param[in] dev The device we're targeting.
     * @param[in] numQueries The number of queries we expect to make.
     * @param[in] minLocSize The requested minimum local group size.
     * @param[out] numThreads The number of threads we'll want for the given number of queries and group size.
     * @param[out] locSize The number of threads per thread group. May actually be smaller than minLocSize depending on the device.
     */
    void getCLNumThreads(cl_device_id dev, int numQueries, int minLocSz,
                         size_t *numThreads, size_t *locSize) const
    {
        cl_int err = CL_SUCCESS;
        // Get the device type
        cl_device_type dev_type;
        err = clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
        HandleFLANNErr(err);

        // Get the maximum group size
        size_t max_work_group_size;
        err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                              sizeof(size_t), &max_work_group_size, NULL);
        HandleFLANNErr(err);

        // Maximum work group size per dimension
        size_t max_work_item_sizes[3];
        err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                              sizeof(max_work_item_sizes), &max_work_item_sizes, NULL);
        HandleFLANNErr(err);

        if (max_work_group_size < max_work_item_sizes[0])
            max_work_group_size = max_work_item_sizes[0];

        // Doesn't seem to run on my CPU with a work group over 1, even though
        // max_work_group_size == 1024. Bug in Apple's OpenCL implementation?
        if ((minLocSz <= 1 && dev_type != CL_DEVICE_TYPE_GPU)
             || dev_type == CL_DEVICE_TYPE_CPU || max_work_group_size < 2) {
            (*locSize) = 1;
        } else if ((minLocSz <= 2 && dev_type != CL_DEVICE_TYPE_GPU)
                   || max_work_group_size < 4) {
            (*locSize) = 2;
        } else if ((minLocSz <= 4 && dev_type != CL_DEVICE_TYPE_GPU)
                    || max_work_group_size < 8) {
            (*locSize) = 4;
        } else if ((minLocSz <= 8 && dev_type != CL_DEVICE_TYPE_GPU)
                    || max_work_group_size < 16) {
            (*locSize) = 8;
        } else if ((minLocSz <= 16 && dev_type != CL_DEVICE_TYPE_GPU)
                    || max_work_group_size < 32) {
            (*locSize) = 16;
            // Should be at least 32 on GPU for hardware thread scheduler reasons
        } else if (minLocSz <= 32 || max_work_group_size < 64) {
            (*locSize) = 32;
        } else if (minLocSz <= 64 || max_work_group_size < 128) {
            (*locSize) = 64;
        } else if (minLocSz <= 128 || max_work_group_size < 256) {
            (*locSize) = 128;
        } else if (minLocSz <= 256 || max_work_group_size < 512) {
            (*locSize) = 256;
        } else if (minLocSz <= 512 || max_work_group_size < 1024) {
            (*locSize) = 512;
        } else if (minLocSz <= 1024 || max_work_group_size < 2048) {
            (*locSize) = 1024;
        }

        (*numThreads) = numQueries*(*locSize);
    }

    /**
     * Build the given program from the given source with the given flags.
     * Handle the multitude of kernel building compile errors with some
     * useful message.
     *
     * @param[in] context OpenCL context to use.
     * @param[in] dev OpenCL device to compile for.
     * @param[in] programSrc The source code for the OpenCL kernel.
     * @param[in] progName The name of the kernel to compile in the source code.
     * @param[in] buildStr The string to use to compile the kernel. May contain flags (such as -cl-fast-relaxed-math -Werror) or defined invarants.
     * @return The compiled OpenCL kernel.
     */
    cl_kernel buildCLKernel(cl_context context, cl_device_id dev, const char *programSrc,
                            const char *progName, char *buildStr) const
    {
        cl_program program;
        cl_kernel kernel;
        cl_int err = 0;

        program = clCreateProgramWithSource(context, 1, &programSrc, NULL, &err);

        err = clBuildProgram(program, 1, &dev, buildStr, NULL, NULL);

        //Detailed debugging info
        if (err != CL_SUCCESS)
        {
            size_t len;
            char *buffer = (char *)calloc(128000, sizeof(char));

            printf("Error: Failed to build program executable!\n");
            printf("clBuildProgram return:\n");
            if(err == CL_INVALID_PROGRAM)
                printf("CL_INVALID_PROGRAM\n");
            else if(err == CL_INVALID_VALUE)
                printf("CL_INVALID_VALUE\n");
            else if(err == CL_INVALID_BINARY)
                printf("CL_INVALID_BINARY\n");
            else if(err == CL_INVALID_BUILD_OPTIONS)
                printf("CL_INVALID_BUILD_OPTIONS\n");
            else if(err == CL_INVALID_OPERATION)
                printf("CL_INVALID_OPERATION\n");
            else if(err == CL_COMPILER_NOT_AVAILABLE)
                printf("CL_COMPILER_NOT_AVAILABLE\n");
            else if(err == CL_BUILD_PROGRAM_FAILURE)
                printf("CL_BUILD_PROGRAM_FAILURE\n");
            else if(err == CL_OUT_OF_HOST_MEMORY)
                printf("CL_OUT_OF_HOST_MEMORY\n");
            else
                printf("unrecognized: %d\n", err);

	                err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_STATUS, 128000*sizeof(char), buffer, &len);
            HandleFLANNErr(err);

            printf("Build Status:\n");
            if(buffer[0] == CL_BUILD_NONE)
                printf("CL_BUILD_NONE\n");
            else if(buffer[0] == CL_BUILD_ERROR)
                printf("CL_BUILD_ERROR\n");
            else if(buffer[0] == CL_BUILD_SUCCESS)
                printf("CL_BUILD_SUCCESS\n");
            else if(buffer[0] == CL_BUILD_IN_PROGRESS)
                printf("CL_BUILD_IN_PROGRESS\n");
            else
                printf("unrecognized\n");

            err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 128000*sizeof(char), buffer, &len);
            printf("Get Build Info:\n");
            switch (err) {
                case CL_INVALID_DEVICE:
                    printf("CL_INVALID_DEVICE\n"); break;
                case CL_INVALID_VALUE:
                    printf("CL_INVALID_VALUE\n"); break;
                case CL_INVALID_PROGRAM:
                    printf("CL_INVALID_PROGRAM\n"); break;
            }

            printf("Build Info:\n%s\nProgram Source:\n%s\n", buffer, programSrc);

            free(buffer);
            exit(1);
        }

        kernel = clCreateKernel(program, progName, &err);
        HandleFLANNErr(err);

        err = clReleaseProgram(program);
        HandleFLANNErr(err);

        return kernel;
    }

    /**
     * Run the kernel with a default group size and print debug info if
     * profiling is enabled in the queue.
     *
     * @param[in] kern The compiled OpenCL kernel to run.
     * @param[in] numThreads Number of threads to run.
     * @param[in] locSize Size of the local thread group.
     * @return CL_SUCCESS
     */
    cl_int runKern(cl_kernel kern, size_t numThreads, size_t locSize) const
    {
        cl_int err = CL_SUCCESS;
        cl_event ev;
        cl_command_queue cmd_queue = this->cl_cmd_queue_;
        size_t loc_size_arr[3] = {locSize,1,1};
        size_t *loc_size_ptr;

        if (locSize != 0)
            loc_size_ptr = loc_size_arr;
        else
            loc_size_ptr = NULL;

        // Run the calculation by enqueuing it and forcing the
        // command queue to complete the task
        err = clEnqueueNDRangeKernel(cmd_queue, kern, 1, NULL,
                                     &numThreads, loc_size_ptr, 0, NULL, &ev);
        if (err == CL_INVALID_WORK_ITEM_SIZE)
            printf("CL_INVALID_WORK_ITEM_SIZE\n");
        else if (err == CL_INVALID_WORK_GROUP_SIZE)
            printf("CL_INVALID_WORK_GROUP_SIZE\n");
        else if (err == CL_INVALID_GLOBAL_WORK_SIZE)
            printf("CL_INVALID_GLOBAL_WORK_SIZE\n");
        else if (err == CL_INVALID_KERNEL_ARGS)
            printf("CL_INVALID_KERNEL_ARGS\n");
        else if (err == CL_INVALID_WORK_DIMENSION)
            printf("CL_INVALID_WORK_DIMENSION\n");
        else if (err == CL_INVALID_PROGRAM_EXECUTABLE)
            printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
        else if (err == CL_INVALID_KERNEL)
            printf("CL_INVALID_KERNEL\n");
        HandleFLANNErr(err);
        err = clFinish(cmd_queue);
        HandleFLANNErr(err);

        cl_command_queue_properties prop;
        err = clGetCommandQueueInfo(cmd_queue, CL_QUEUE_PROPERTIES, sizeof(prop), &prop, NULL);
        HandleFLANNErr(err);

        // Conditional on profiling enabled in the queue
        if (prop & CL_QUEUE_PROFILING_ENABLE) {
            size_t start_time = 0, end_time;

            err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                          sizeof(size_t), &start_time, NULL);
            HandleFLANNErr(err);
            err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                          sizeof(size_t), &end_time, NULL);
            HandleFLANNErr(err);

            err = clReleaseEvent(ev);
            HandleFLANNErr(err);
            double totTime = (end_time-start_time)/1000000000.0;

            printf("NNIndex OpenCL Total Kernel Time:%12.4f\n", totTime);
        }

        return CL_SUCCESS;
    }

    /**
     * Allocate a block of OpenCL device memory to a location, and also allocate
     * the same sized block of pinned memory on the host. The pinned memory
     * makes for good temp memory to format data structures and for faster
     * device-to-host data transfer.
     *
     * @param[in] context The context to use for the allocation.
     * @param[in] sz The size of the allocation in bytes.
     * @param[in] mem_flags Allocate the device memory with these flags. (CL_MEM_READ_ONLY, etc)
     * @param[out] clPtr Pointer to where to store the device cl_mem.
     * @param[out] pinnedWorkCLPtr Pointer to the pinned host cl_mem.
     * @param[out] pinnedWorkPtr Pointer to the pinned host memory region. Cannot be used until the queue is finished.
     */
    template <typename CLAllocType>
    void allocCLDevPinnedWorkMem(cl_context context, size_t sz, cl_mem_flags mem_flags,
                                 cl_mem *clPtr, cl_mem *pinnedWorkCLPtr, CLAllocType **pinnedWorkPtr) const
    {
        cl_int err = CL_SUCCESS;
        assert(sz >= 0);
        assert((*clPtr) == NULL);

        (*clPtr) = clCreateBuffer(context, mem_flags, sz, NULL, &err);
        HandleFLANNErr(err);
        (*pinnedWorkCLPtr) = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
        HandleFLANNErr(err);
        (*pinnedWorkPtr) = (CLAllocType *)clEnqueueMapBuffer(this->cl_cmd_queue_, (*pinnedWorkCLPtr), CL_FALSE,
                                                             CL_MAP_READ|CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
        HandleFLANNErr(err);
    }

    /**
     * Free a block of OpenCL memory, if it isn't NULL. Then
     * set it as NULL so no dangling pointers.
     *
     * @param[in] toFree Pointer to the cl_mem region to free.
     */
    inline void freeCLMem(cl_mem *toFree) const
    {
        if (*toFree) {
            cl_int err = clReleaseMemObject(*toFree);
            HandleFLANNErr(err);
            (*toFree) = NULL;
        }
    }

    /**
     * Calculate the adjusted number of columns of in the knn result array. Make a size that
     * works well with vector instructions.
     *
     * @param[in] knn The requested number of nearest neighbors to retrieve.
     * @return The number of nearest neighbors to actually use.
     */
    int getCLknn(size_t knn) const
    {
        // Each full set of results is a multiple of four plus one result to be replaced
        // when a new result is added
        return 4*((knn+3)/4)+1;
    }

    /**
     * Returns source code for the kernel. Code should be appropriate for the
     * algorithm (hash vs tree) and contain the correct distance calculation.
     * Caller compiles the string with the correct defines ("-D")
     *
     * Caller must call delete[] after use.
     *
     * @param  isLocal [description]
     * @param  type    [description]
     * @return         [description]
     */
    char *getCLSrc(bool isLocal, const char *tooFarSrc, flann_distance_t type) const
    {
        char *src = new char[100000];
        const char *distSrc = NULL;

        if (isLocal)
        {
            switch(type)
            {
                case FLANN_DIST_L2:
                    distSrc =
// Compare the local query point against a global point and return the euclidian distance
// between them.
"DISTANCE_TYPE vecDistLoc (const __local ELEMENT_TYPE *v1, const __global ELEMENT_TYPE *v2, const int off2)\n"
"{\n"
    "DISTANCE_TYPE_VEC sum = (DISTANCE_TYPE_VEC)0.0f;\n"
    "for (int i = 0; i < N_VECLEN; i += 4) {\n"
        // Use vector instructions because, why not? (And make better use of global latency.)
        "DISTANCE_TYPE_VEC diff = CONVERT_DISTANCE_TYPE_VEC(vload4(0,v1+i)) - CONVERT_DISTANCE_TYPE_VEC(vload4(0,v2+i+off2));\n"
        "sum = mad(diff, diff, sum);\n"
    "}\n"

    // return the sum of all vector components
    "return sum.s0+sum.s1+sum.s2+sum.s3;\n"
"}\n";
                    break;
                case FLANN_DIST_HAMMING:
                case FLANN_DIST_HAMMING_LUT:
                case FLANN_DIST_HAMMING_POPCNT:
                    distSrc =
// Compare the local query point against a global point and return the euclidian distance
// between them.
"DISTANCE_TYPE vecDistLoc (const __local ELEMENT_TYPE *v1, const __global ELEMENT_TYPE *v2, const int off2)\n"
"{\n"
    "DISTANCE_TYPE_VEC sum = (DISTANCE_TYPE_VEC)0;\n"
    "for (int i = 0; i < (int)(N_VECLEN*sizeof(ELEMENT_TYPE)/sizeof(DISTANCE_TYPE)); i += 4) {\n"
        // Use vector instructions because, why not? (And make better use of global latency.)
        "sum += popcount(vload4(0,i+(__local DISTANCE_TYPE *)(v1)) ^ vload4(0,i+(__global DISTANCE_TYPE *)(v2+off2)));\n"
    "}\n"

    // return the sum of all vector components
    "return sum.s0+sum.s1+sum.s2+sum.s3;\n"
"}\n";
                    break;
                case FLANN_DIST_L1:
                case FLANN_DIST_MINKOWSKI:
                case FLANN_DIST_MAX:
                case FLANN_DIST_HIST_INTERSECT:
                case FLANN_DIST_HELLINGER:
                case FLANN_DIST_CHI_SQUARE:
                case FLANN_DIST_KULLBACK_LEIBLER:
                case FLANN_DIST_L2_SIMPLE:
                    distSrc = "vecDistLoc unimplemented for this distance measure!";
                    break;
            }
            
// **************** Implementation that takes advantage of thread groups on GPUs ****************
			const char *queryTreeSrcLocal =
"void findLeaves(const __global int *nodeIndex,\n"
                "const __local ELEMENT_TYPE *query,\n"
                "const __global ELEMENT_TYPE *dataset,\n"
                "__local int *locDone, __local int *locPtr,\n"
                "__local DISTANCE_TYPE *heapDist, __local int *heapId,\n"
                "const int nNodes);\n"

"void findNodes(const __global int *nodeIndex,\n"
               "const __global ELEMENT_TYPE *nodePivots,\n"
               "const __global DISTANCE_TYPE *nodeVariance,\n"
               "__local int *locDone, const __local ELEMENT_TYPE *query,\n"
               "__local DISTANCE_TYPE *heapDist, __local int *heapId, const int nNodes);\n"

"void findNewNodeDist(const __global int *nodeIndex,\n"
                     "const __global ELEMENT_TYPE *nodePivots,\n"
                     "const __global DISTANCE_TYPE *nodeVariance,\n"
                     "const __local ELEMENT_TYPE *vec,\n"
                     "__local DISTANCE_TYPE *heapDist, __local int *heapId, const int nNodes);\n"

"void sortHeap(__local DISTANCE_TYPE *heapDist, __local int *heapId);\n"

"void bitonicMerge(__local DISTANCE_TYPE *heapDist, __local int *heapId,\n"
                  "int size, int dir);\n"

"void checkDone(int privDone, __local int *locDone);\n"

"void initLoc (__local DISTANCE_TYPE *heapDist, __local int *heapId, const int nNodes);\n"

"void storeResult (__global DISTANCE_TYPE *resultDistArr, __global int *resultIdArr,\n"
                  "__local DISTANCE_TYPE *heapDist, __local int *heapId, const int queryOff);\n"

"DISTANCE_TYPE vecDistLoc (const __local ELEMENT_TYPE *v1,\n"
                          "const __global ELEMENT_TYPE *v2,\n"
                          "const int off2);\n"

// Kernel for finding k-nearest-neighbors using thread groups and local memory. This makes it work
// well for execution on GPU processors. Thus it has been optimized for fewer branches. There
// is one query performed per thread group, each thread group is large enough to check pointers
// from the first half of the heap, save the reults in the other half, and then sort them all for
// the next pass. This algorithm uses an index of the same structure, but the logic is completely
// different.
"#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n"
"__kernel __attribute__((reqd_work_group_size(LOC_SIZE, 1, 1)))\n"
"void findNeighborsLocal (const __global int *nodeIndex,\n"
                         "const __global ELEMENT_TYPE *nodePivots,\n"
                         "const __global DISTANCE_TYPE *nodeVariance,\n"
                         "const __global DISTANCE_TYPE *nodeRadii,\n"
                         "const __global ELEMENT_TYPE *dataset,\n"
                         "__global DISTANCE_TYPE *resultDistArr,\n"
                         "__global int *resultIdArr,\n"
                         "const __global ELEMENT_TYPE *queryArr,\n"
                         "const int nNodes,\n"
                         "const int numQueries,\n"
                         "const int queryOff )\n"
"{\n"
    // N_HEAP == get_local_size(0) *2 == (avg checks per node needed for MAX_CHECKS) *2
    // == LOC_SIZE *2 >= N_RESULT *2 >= 32 (min GPU hardware thread group size, I think) *2
    "__local int heapId[N_HEAP];\n"
    "__local DISTANCE_TYPE heapDist[N_HEAP];\n"
    "__local ELEMENT_TYPE query[N_VECLEN];\n"
    "__local int locPtr;\n"
    "__local int locDone;\n"

    // All threads in a group preform (approx) MAX_CHECKS in parallel
    // for one query (one query per thread group)
    // Copy vec to __local space because all threads are using it.
    "for (int i = get_local_id(0); i < N_VECLEN; i += LOC_SIZE)\n"
        "query[i] = queryArr[i + (get_group_id(0)+queryOff)*N_VECLEN];\n"

    // Find the child nodes that are closest to the query point
    "findNodes(nodeIndex, nodePivots, nodeVariance, &locDone, query, heapDist, heapId, nNodes);\n"

    // Go from leaf pointers in the heap to a sorted set of leaves
    "findLeaves(nodeIndex, query, dataset, &locDone, &locPtr, heapDist, heapId, nNodes);\n"

    // After inserting, sorting, and checking leaves, the results are in the lower part of the heap
    "storeResult(resultDistArr, resultIdArr, heapDist, heapId, queryOff);\n"
"}\n"

// Start with the root node, search through all parent nodes in the lower half of the
// heap and add their children to the top half of the heap. Then sort the heap by priority
// (adjusted distance). After each pass, check if there are any parent nodes left in the bottom
// half of the heap. (Otherwise they are all nodes pointing to groups of leaves.)
"void findNodes(const __global int *nodeIndex,\n"
               "const __global ELEMENT_TYPE *nodePivots,\n"
               "const __global DISTANCE_TYPE *nodeVariance,\n"
               "__local int *locDone, const __local ELEMENT_TYPE *query,\n"
               "__local DISTANCE_TYPE *heapDist, __local int *heapId, const int nNodes)\n"
"{\n"
    "initLoc(heapDist, heapId, nNodes);\n"

    // Init the node query array with a pointer to root_'s children
    // Ensure that all trees are decended
    "for (int i = 0; i < N_TREES; i++) {\n"
        "heapId[i] = i*BRANCHING;\n"
        "heapDist[i] = 0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"

    // Find the closest children to the root
    "findNewNodeDist(nodeIndex, nodePivots, nodeVariance, query, heapDist, heapId, nNodes);\n"

    // Sort to mix the new nodes into the correct positions in the heap
    "sortHeap(heapDist, heapId);\n"
    "do {\n"
        // Use the closest parent nodes to discover additional nodes
        "findNewNodeDist(nodeIndex, nodePivots, nodeVariance, query, heapDist, heapId, nNodes);\n"

        // Init shared vars
        "if (get_local_id(0) == 0) {\n"
            // Set this as if it's our last pass, will be flipped back if not done
            "(*locDone) = 1;\n"
        "}\n"

        // Sort to mix the new nodes into the correct positions in the heap
        "sortHeap(heapDist, heapId);\n"

        // While there is a pointer to a parent node in the heap, keep going
        "checkDone((heapId[get_local_id(0)] >= nNodes), locDone);\n"
    "} while (!(*locDone));\n"
"}\n"

// The bottom half of the heap are pointers to groups of leaves. We have one thread per leaf
// group, so store the pointer in private memory and fill the heap with candidate leaf points.
// After each pass, sort the leaves. After all leaf groups are checked and sorted, we have
// preformed approximately MAX_CHECKS and the bottom of the heap are the closest points.
"void findLeaves(const __global int *nodeIndex,\n"
                "const __local ELEMENT_TYPE *query,\n"
                "const __global ELEMENT_TYPE *dataset,\n"
                "__local int *locDone, __local int *locPtr,\n"
                "__local DISTANCE_TYPE *heapDist, __local int *heapId,\n"
                "const int nNodes)\n"
"{\n"
    // Store pointer to list before init local mem
    "int leafPtr = heapId[get_local_id(0)];\n"
    "int lastPtr = (leafPtr < INT_MAX) ? (nodeIndex[leafPtr] + leafPtr) : 0;\n"

    // Reset heap
    "initLoc(heapDist, heapId, nNodes);\n"

    // 'Unroll' this ahead of the main loop to fill both halves of the heap before
    // first sort, and because we don't really need to checkDone() so soon.
    // Init shared vars
    "if (get_local_id(0) == 0)\n"
        // Pointer starts at the beginning of the heap
        "(*locPtr) = 0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"

    // Find the next dataset index to check
    "int locI = 0;\n"
    "while (leafPtr < lastPtr && (locI = atomic_inc(locPtr)) < N_HEAP)\n"
        "heapId[locI] = nodeIndex[++leafPtr];\n"

    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    // Note that the heapId may be uninitialized at this location if we're
    // only decending one tree or it's a small/wide tree.
    "if (heapId[get_local_id(0)] < INT_MAX)\n"
        "heapDist[get_local_id(0)] = vecDistLoc(query, dataset, heapId[get_local_id(0)]*N_VECLEN);\n"

    "do {\n"
        // Set the leaf distances for the given IDs.
        "if (heapId[LOC_SIZE + get_local_id(0)] < INT_MAX)\n"
            "heapDist[LOC_SIZE + get_local_id(0)] = vecDistLoc(query, dataset, heapId[LOC_SIZE + get_local_id(0)]*N_VECLEN);\n"

        // Sort heap to leave the closest distances in the bottom of the heap
        "sortHeap(heapDist, heapId);\n"
        // Init shared vars
        "if (get_local_id(0) == 0) {\n"
            // Pointer starts in the middle of the heap
            "(*locPtr) = LOC_SIZE;\n"

            // Assume we're done until proven otherwise
            "(*locDone) = 1;\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"

        // Find the next dataset index to check
        "locI = 0;\n"
        "while (leafPtr < lastPtr && (locI = atomic_inc(locPtr)) < N_HEAP)\n"
            "heapId[locI] = nodeIndex[++leafPtr];\n"

        // We're done if there's no more leaf indicies to search
        "checkDone(locI == 0, locDone);\n"
    "} while (!(*locDone));\n"
"}\n"

// Find a new set of nodes given the parent nodes already in the heap.
"void findNewNodeDist(const __global int *nodeIndex,\n"
                     "const __global ELEMENT_TYPE *nodePivots,\n"
                     "const __global DISTANCE_TYPE *nodeVariance,\n"
                     "const __local ELEMENT_TYPE *vec,\n"
                     "__local DISTANCE_TYPE *heapDist, __local int *heapId, const int nNodes)\n"
"{\n"
    // Find the next valid node pointer
    "unsigned int locId = get_local_id(0) / BRANCHING;\n"

    // Skip ids marked as pointers to leaves
    "while (locId < N_HEAP && heapId[locId] >= nNodes)\n"
        "locId += LOC_SIZE / BRANCHING;\n"

    // Check if we were able to find a valid node ptr
    // Go from pointer-in-heap to actual node ID
    "int nodeId = (locId < N_HEAP) ? (heapId[locId] + get_local_id(0) % BRANCHING) : 0;\n"

    // Ensure all node IDs are retrieved before adjusting the heap
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "if (locId < N_HEAP) {\n"
        // Invalidate this pointer if it's been used
        "heapDist[locId] = MAX_DIST;\n"
        "heapId[locId] = INT_MAX;\n"

        // Store adjusted distance to node and its pointer
        "int i = LOC_SIZE + get_local_id(0);\n"
        "heapDist[i] = vecDistLoc(vec, nodePivots, nodeId*N_VECLEN)\n"
"#ifdef CB_INDEX\n"
            " - CB_INDEX*nodeVariance[nodeId]\n"
"#endif\n"
            ";\n"
        "heapId[i] = nodeIndex[nodeId];\n"
    "}\n"
"}\n"

// Sort the local heap by ascending distance using a bitonic sort
"void sortHeap(__local DISTANCE_TYPE *heapDist, __local int *heapId)\n"
"{\n"
    // Iterate from small sizes to N_HEAP
    "for (int size = 2; size < N_HEAP; size <<= 1) {\n"
        //Bitonic merge
        "int ddd = (get_local_id(0) & (size / 2)) != 0;\n"
        "bitonicMerge(heapDist, heapId, size, ddd);\n"
    "}\n"
    "bitonicMerge(heapDist, heapId, N_HEAP, 0);\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
"}\n"

"void bitonicMerge(__local DISTANCE_TYPE *heapDist, __local int *heapId,\n"
                  "int size, int dir)\n"
"{\n"
    "for (int stride = size / 2; stride > 0; stride >>= 1)\n"
    "{\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));\n"
        "DISTANCE_TYPE keyA = heapDist[pos];\n"
        "DISTANCE_TYPE keyB = heapDist[pos + stride];\n"
        // Swap items if out of order
        "if ((keyA < keyB) == dir)\n"
        "{\n"
            "int valA = heapId[pos];\n"
            "int valB = heapId[pos + stride];\n"
            "heapDist[pos] = keyB;\n"
            "heapDist[pos + stride] = keyA;\n"
            "heapId[pos] = valB;\n"
            "heapId[pos + stride] = valA;\n"
        "}\n"
    "}\n"
"}\n"

// Check if all threads are done (as signaled by the 'done' parameter)
"void checkDone(int done, __local int *locDone)\n"
"{\n"
    // Assume all threads are done in local mem and the bit has already been set...
    // Until proven otherwise by private mem
    "if (!done)\n"
        "(*locDone) = 0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
"}\n"

// Init both halves of the local heap to be invalid (maximum distance and a node ID of a leaf node)
"void initLoc (__local DISTANCE_TYPE *heapDist, __local int *heapId, const int nNodes)\n"
"{\n"
    // Init our position
    "heapDist[get_local_id(0)] = MAX_DIST;\n"
    "heapId[get_local_id(0)] = INT_MAX;\n"
    "heapDist[get_local_id(0)+LOC_SIZE] = MAX_DIST;\n"
    "heapId[get_local_id(0)+LOC_SIZE] = INT_MAX;\n"
"}\n"

// Store the bottom points of the heap as results in global memory
"void storeResult (__global DISTANCE_TYPE *resultDistArr, __global int *resultIdArr,\n"
                  "__local DISTANCE_TYPE *heapDist, __local int *heapId, const int queryOff)\n"
"{\n"
    "__global DISTANCE_TYPE *resultDist = &resultDistArr[(get_group_id(0)+queryOff)*N_RESULT];\n"
    "__global int *resultId = &resultIdArr[(get_group_id(0)+queryOff)*N_RESULT];\n"

    // Get rid of duplicates if there are multiple trees searched
    "if (N_TREES > 1) {\n"
        "int thisHeapId = heapId[get_local_id(0)];\n"
        "for (int i = get_local_id(0)+1; i < N_HEAP; i++)\n"
            "if (heapId[i] == thisHeapId)\n"
                "heapDist[i] = MAX_DIST;\n"

        // Send duplicates to the back of the heap
        "sortHeap(heapDist, heapId);\n"
    "}\n"

    // Copy from heap to global result mem
    "for (int i = get_local_id(0); i < N_RESULT; i += LOC_SIZE) {\n"
        "resultDist[i] = heapDist[i];\n"
        "resultId[i] = heapId[i];\n"
    "}\n"
"}\n";
            sprintf(src, "%s\n%s\n%s", queryTreeSrcLocal, distSrc, tooFarSrc);
        }
        else
        {
            switch(type)
            {
                case FLANN_DIST_L2:
                    distSrc =
// Calculates the square of the euclidian distance between two vectors
"DISTANCE_TYPE vecDist (const __global ELEMENT_TYPE *v1, const __global ELEMENT_TYPE *v2, const int off2)\n"
"{\n"
    "DISTANCE_TYPE_VEC sum = (DISTANCE_TYPE_VEC)0.0f;\n"
    "for (int i = 0; i < N_VECLEN; i += 4) {\n"
        // Vector instructions FTW!
        "DISTANCE_TYPE_VEC diff = CONVERT_DISTANCE_TYPE_VEC(vload4(0,v1+i)) - CONVERT_DISTANCE_TYPE_VEC(vload4(0,v2+i+off2));\n"
        "sum = mad(diff, diff, sum);\n"
    "}\n"
    "return sum.s0+sum.s1+sum.s2+sum.s3;\n"
"}\n";
                    break;
                case FLANN_DIST_HAMMING:
                case FLANN_DIST_HAMMING_LUT:
                case FLANN_DIST_HAMMING_POPCNT:
                    distSrc =
// Calculates the square of the euclidian distance between two vectors
"DISTANCE_TYPE vecDist (const __global ELEMENT_TYPE *v1, const __global ELEMENT_TYPE *v2, const int off2)\n"
"{\n"
    "DISTANCE_TYPE_VEC sum = (DISTANCE_TYPE_VEC)0;\n"
    "for (int i = 0; i < (int)(N_VECLEN*sizeof(ELEMENT_TYPE)/sizeof(DISTANCE_TYPE)); i += 4) {\n"
        // Vector instructions FTW!
        "sum += popcount(vload4(0,i+(__global DISTANCE_TYPE *)(v1)) ^ vload4(0,i+(__global DISTANCE_TYPE *)(v2+off2)));\n"
    "}\n"

    "return sum.s0+sum.s1+sum.s2+sum.s3;\n"
"}\n";
                    break;
                case FLANN_DIST_L1:
                case FLANN_DIST_MINKOWSKI:
                case FLANN_DIST_MAX:
                case FLANN_DIST_HIST_INTERSECT:
                case FLANN_DIST_HELLINGER:
                case FLANN_DIST_CHI_SQUARE:
                case FLANN_DIST_KULLBACK_LEIBLER:
                case FLANN_DIST_L2_SIMPLE:
                    distSrc = "vecDist unimplemented for this distance measure!";
                    break;
            }
			
			const char *queryTreeSrcGeneral =
"int findNN(const __global int *nodeIndexArr,\n"
           "const __global ELEMENT_TYPE *nodePivots,\n"
           "const __global DISTANCE_TYPE *nodeRadii,\n"
           "const __global DISTANCE_TYPE *nodeVariance,\n"
           "const __global ELEMENT_TYPE *dataset,\n"
           "const int idx,\n"
           "__global DISTANCE_TYPE *resultDist, __global int *resultId,\n"
           "const __global ELEMENT_TYPE *vec, int *checks,\n"
           "__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
           "int *heapStart, int *heapEnd, const int nNodes);\n"

"int tooFar(const __global ELEMENT_TYPE *nodePivots, const __global DISTANCE_TYPE *nodeRadii,\n"
           "const int nodeId, const __global ELEMENT_TYPE *vec, const int *checks,\n"
           "const __global DISTANCE_TYPE *resultDist, const __global int *resultId );\n"

"int exploreNodeBranches (const __global int *nodeIndexArr,\n"
                         "const __global ELEMENT_TYPE *nodePivots,\n"
                         "const __global DISTANCE_TYPE *nodeVariance,\n"
                         "const int nodeId,\n"
                         "const __global ELEMENT_TYPE *q,\n"
                         "__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
                         "int *heapStart, int *heapEnd);\n"

"DISTANCE_TYPE vecDist (const __global ELEMENT_TYPE *v1, const __global ELEMENT_TYPE *v2, const int off);\n"

"void heapInsert(DISTANCE_TYPE nodeDist, int nodeId,\n"
                "__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
                "int *heapStart, int *heapEnd);\n"

"int heapPopMin(__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
               "int *heapStart, int *heapEnd);\n"

"void resultAddPoint(DISTANCE_TYPE rsDist, int rsId, int *checks,\n"
                    "__global DISTANCE_TYPE *resultDist, __global int *resultId );\n"

// Kernel intended for CPU operation. It plays a bit more fast and loose with branches, random
// accesses to global memory, and amount of memory used. Operations are vectorized when
// possible. The actual logic is very similar to the non-OpenCL code. One thread per query.
"__kernel void findNeighbors (const __global int *nodeIndexArr,\n"
                             "const __global ELEMENT_TYPE *nodePivots,\n"
                             "const __global DISTANCE_TYPE *nodeVariance,\n"
                             "const __global DISTANCE_TYPE *nodeRadii,\n"
                             "const __global ELEMENT_TYPE *dataset,\n"
                             "__global DISTANCE_TYPE *resultDistArr, __global int *resultIdArr,\n"
                             "__global ELEMENT_TYPE *vecArr,\n"
                             "const int nNodes,\n"
                             "const int numQueries,\n"
                             "const int queryOff )\n"
"{\n"
    "int gid = get_global_id(0);\n"
    // In case we are operating with a group size
    "if (gid >= numQueries)\n"
        "return;\n"
    "gid += queryOff;\n"

    // Set pointers to global memory that are constant for this kernel
    "__global ELEMENT_TYPE *vec = &vecArr[gid*N_VECLEN];\n"
    "__global DISTANCE_TYPE *resultDist = &resultDistArr[gid*N_RESULT];\n"
    "__global int *resultId = &resultIdArr[gid*N_RESULT];\n"

    "__private DISTANCE_TYPE heapDist[N_HEAP];\n"
    "__private int heapId[N_HEAP];\n"

    // Set up initial variables so we have something to compare against (one less edge case)
    "resultDist[0] = MAX_DIST;\n"
    "resultId[0] = 0;\n"
    "int checks = 1;\n"
    "int heapStart = 0;\n"
    "int heapEnd = 0;\n"
    "int tree = 1;\n"

    // Find the first node to search starting from the root
    "int nextNode = exploreNodeBranches(nodeIndexArr, nodePivots, nodeVariance,\n"
                                       "0, vec, heapDist, heapId, &heapStart, &heapEnd);\n"

    // Iterate through all suggested nodes until we're done or have nothing else to search
    "do {\n"
        // Find the next nearest neighbor
        "nextNode = findNN(nodeIndexArr, nodePivots, nodeRadii, nodeVariance, dataset,\n"
                          "nextNode, resultDist, resultId, vec,\n"
                          "&checks, heapDist, heapId, &heapStart, &heapEnd, nNodes);\n"

        // If we've hit the end of this branch, pop the next one off the heap
        "if (nextNode == -1) {\n"
            "if (tree < N_TREES) {\n"
                // Find the first node to search starting from the root
                "nextNode = exploreNodeBranches(nodeIndexArr, nodePivots, nodeVariance,\n"
                                               "tree*BRANCHING, vec, heapDist, heapId, &heapStart, &heapEnd);\n"
                "tree++;"
            "} else {\n"
                "nextNode = heapPopMin(heapDist, heapId, &heapStart, &heapEnd);\n"
            "}\n"
        "}\n"

    "} while ((nextNode != -1) && (checks < MAX_CHECKS || checks < N_RESULT));\n"
"}\n"

// Kernel intended for finding the exact set of neighbors. As of writing this comment, it hasn't
// been tested. Uses vector commands when possible, and likely optimized for CPU.
"__kernel void findNeighborsExact (const __global int *nodeIndexArr,\n"
                                  "const __global ELEMENT_TYPE *nodePivots,\n"
                                  "const __global DISTANCE_TYPE *nodeRadii,\n"
                                  "const __global ELEMENT_TYPE *dataset,\n"
                                  "__global DISTANCE_TYPE *resultDistArr,\n"
                                  "__global int *resultIdArr,\n"
                                  "__global ELEMENT_TYPE *vecArr,\n"
                                  "const int nNodes,\n"
                                  "const int numQueries,\n"
                                  "const int queryOff )\n"
"{\n"
    "int gid = get_global_id(0);\n"
    "if (gid >= numQueries)\n"
        "return;\n"
    "gid += queryOff;\n"

    "__global ELEMENT_TYPE *vec = &vecArr[gid*N_VECLEN];\n"

    // Each search has its own result array
    "__global DISTANCE_TYPE *resultDist = &resultDistArr[gid*N_RESULT];\n"
    "__global int *resultId = &resultIdArr[gid*N_RESULT];\n"

    // Add the first point now to remove the case of not having an
    // existing point to compare against (less flow control overall)
    "resultDist[0] = vecDist(vec, dataset, 0);\n"
    "resultId[0] = 0;\n"
    "int checks = 1;\n"

    // Iterate through all nodes that reference leaves
    "for (int nodeId = 0; nodeId < nNodes; ++nodeId) {\n"
        "int nodePtr = nodeIndexArr[nodeId];\n"

        // Check that the current node references leaves and it's not too far away to be valid
        "if (nodePtr < nNodes ||\n"
            "tooFar(nodePivots, nodeRadii, nodeId, vec, &checks, resultDist, resultId))\n"
            "continue;\n"

        // Iterate through leaves of this node
        "int lastPtr = nodeIndexArr[nodePtr] + nodePtr;\n"
        "do {\n"
            "int datasetI = nodeIndexArr[++nodePtr];\n"
            // Add them all. They will be sorted within the point adding code
            "resultAddPoint(vecDist(vec, dataset, datasetI*N_VECLEN), datasetI,\n"
                           "&checks, resultDist, resultId);\n"
        "} while (nodePtr < lastPtr);\n"
    "}\n"
"}\n"

// Finds the nearest neighbor from the given node
"int findNN(const __global int *nodeIndexArr,\n"
           "const __global ELEMENT_TYPE *nodePivots,\n"
           "const __global DISTANCE_TYPE *nodeRadii,\n"
           "const __global DISTANCE_TYPE *nodeVariance,\n"
           "const __global ELEMENT_TYPE *dataset,\n"
           "const int nodeId,\n"
           "__global DISTANCE_TYPE *resultDist, __global int *resultId,\n"
           "const __global ELEMENT_TYPE *vec, int *checks,\n"
           "__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
           "int *heapStart, int *heapEnd, int nNodes)\n"
"{\n"
    // Check if the point is too far to be useful
    "if (tooFar(nodePivots, nodeRadii, nodeId, vec, checks, resultDist, resultId))\n"
        "return -1;\n"
    "int nodePtr = nodeIndexArr[nodeId];\n"

    // Either the node references children or other nodes
    "if (nodePtr >= nNodes) {\n"
        // Iterate through all child leaves
        "int lastPtr = nodeIndexArr[nodePtr] + nodePtr;\n"
        "while (lastPtr > nodePtr) {\n"
            "int datasetI = nodeIndexArr[++nodePtr];\n"

            // (Attempt to) add to the result array
            "resultAddPoint(vecDist(vec, dataset, datasetI*N_VECLEN), datasetI,\n"
                           "checks, resultDist, resultId);\n"
        "};\n"
        "return -1;\n"
    "} else {\n"
        // If we're a parent node, find the closest child node
        "return exploreNodeBranches(nodeIndexArr, nodePivots, nodeVariance,\n"
                                   "nodePtr, vec, heapDist, heapId, heapStart, heapEnd);\n"
    "}\n"
"}\n"

// Explore child nodes for the closest. Add the rest to the heap
"int exploreNodeBranches (const __global int *nodeIndexArr,\n"
                         "const __global ELEMENT_TYPE *nodePivots,\n"
                         "const __global DISTANCE_TYPE *nodeVariance,\n"
                         "const int nodePtr, const __global ELEMENT_TYPE *q,\n"
                         "__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
                         "int *heapStart, int *heapEnd )\n"
"{\n"
    "int bestI = 0;\n"
    "DISTANCE_TYPE distArr[BRANCHING];\n"
    "distArr[0] = vecDist(q, nodePivots, nodePtr*N_VECLEN);\n"

    // Find the best branch to take
    "for (int i = 1; i < BRANCHING; ++i) {\n"
        "DISTANCE_TYPE d = distArr[i] = vecDist(q, nodePivots, (nodePtr+i)*N_VECLEN);\n"
        "if (d < distArr[bestI])\n"
            "bestI = i;\n"
    "}\n"

    // Add the remaining branches to the heap
    "for (int i = 0; i < BRANCHING; ++i) {\n"
        "if (i == bestI)\n"
            "continue;\n"
        "heapInsert(distArr[i]\n"
"#ifdef CB_INDEX\n"
                   "- CB_INDEX*nodeVariance[nodePtr+i]\n"
"#endif\n"
                   ", nodePtr+i, heapDist, heapId, heapStart, heapEnd);\n"
    "}\n"

    // Follow best branch for now...
    "return nodePtr+bestI;\n"
"}\n"

// Adds a node to the heap array. The heap should be large enough that when it's full there
// are enough leaves to perform the expected number of checks (MAX_CHECKS). If the heap is
// full, the furthest distance is discarded.
"void heapInsert(DISTANCE_TYPE nodeDist, int nodeId,\n"
                "__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
                "int *heapStart, int *heapEnd )\n"
"{\n"
    // If not full, inc the number of elements in the array
    "if ((*heapEnd) < N_HEAP)\n"
        "++(*heapEnd);\n"
    // Don't add nodes that are already worse than the worst one (if full)
    "else if (nodeDist >= heapDist[N_HEAP-1])\n"
        "return;\n"

    // Move all larger distance nodes back
    "int i = (*heapEnd)-1;\n"
    "for (; i > (*heapStart)+3; i -= 4) {\n"
        "DISTANCE_TYPE_VEC hd = vload4(0, heapDist+i-4);\n"
        "if (any(islessequal(hd, nodeDist)))\n"
            "break;\n"
        "vstore4(hd, 0, heapDist+i-3);\n"
        "vstore4(vload4(0, heapId+i-4), 0, heapId+i-3);\n"
    "}\n"
    "for (; i > (*heapStart) && nodeDist < heapDist[i-1]; --i) {\n"
        "heapDist[i] = heapDist[i-1];\n"
        "heapId[i] = heapId[i-1];\n"
    "}\n"

    // Insert new result
    "heapDist[i] = nodeDist;\n"
    "heapId[i] = nodeId;\n"
"}\n"

// Pop the minimum value off of the front of the heap
"int heapPopMin(__private DISTANCE_TYPE *heapDist, __private int *heapId,\n"
               "int *heapStart, int *heapEnd)\n"
"{\n"
    "if ((*heapStart) == (*heapEnd))\n"
        "return -1;\n"

    // Check if we should shift numbers back...
    "if ((*heapStart) == 8) {\n"
        "(*heapEnd) = (*heapEnd)-8;\n"
        // Move one vector at a time
        "int i = 0;\n"
        "for (; i < (*heapEnd)-3; i += 4) {\n"
            "vstore4(vload4(0, heapDist+i+8), 0, heapDist+i);\n"
            "vstore4(vload4(0, heapId+i+8), 0, heapId+i);\n"
        "}\n"
        "for (; i < (*heapEnd); ++i) {\n"
            "heapDist[i] = heapDist[i+8];\n"
            "heapId[i] = heapId[i+8];\n"
        "}\n"
        "(*heapStart) = 0;\n"
    "}\n"

    "return heapId[(*heapStart)++];\n"
"}\n"

// Adds a point to the result array.
// If it's full, the furthest result distance is discarded.
"void resultAddPoint(DISTANCE_TYPE rsDist, int rsId, int *checks,\n"
                    "__global DISTANCE_TYPE *resultDist, __global int *resultId )\n"
"{\n"
    // Don't add results that are already worse than the worst one
    "if ((*checks) < N_RESULT || rsDist < resultDist[N_RESULT-1]) {\n"
        // Skip duplicate entries if there are multiple trees
        "if (N_TREES > 1)\n"
            "for (int i = 0; i < (*checks) && i < N_RESULT-1; i++)\n"
                "if (resultId[i] == rsId && resultDist[i] == rsDist)\n"
                    "return;\n"

        "int i = min((*checks), N_RESULT-1);\n"
        // Move all larger distance results back
        "for (; i > 0 && rsDist < resultDist[i-1]; --i) {\n"
            "resultDist[i] = resultDist[i-1];\n"
            "resultId[i] = resultId[i-1];\n"
        "}\n"

        "resultDist[i] = rsDist;\n"
        "resultId[i] = rsId;\n"
    "}\n"

    // Inc the number of checks done. This is also the maximum number of elements in the array.
    "++(*checks);\n"
"}\n";
            sprintf(src, "%s\n%s\n%s", queryTreeSrcGeneral, distSrc, tooFarSrc);
        }
        return src;
    }

    /**
     * Enable OpenCL with this queue
     */
    cl_command_queue cl_cmd_queue_;

    /**
     * Save the changeable variables when the kernel is built so we know if we
     * need to recompile with new constants
     */
    int cl_kern_max_checks_;
    size_t cl_kern_knn_;

    /**
     * This needs to be known for the current compiled kernel so we have a
     * local group size as was compiled for.
     */
    int cl_kern_heap_size_;

    /**
     * OpenCL kernels are saved here when compiled
     */
    cl_kernel cl_knn_search_kern_;

    bool own_cq_;
};

}

#endif /* FLANN_NN_OPENCL_INDEX_H_ */
#endif /* FLANN_USE_OPENCL */
