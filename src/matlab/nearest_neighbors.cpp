/*
   Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
   Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.

   THE BSD LICENSE

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
   IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
   NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
   THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Workaround for MSVC 10, Matlab incompatibility */
#if (_MSC_VER >= 1600)
#include <yvals.h>
#define __STDC_UTF_16__
#endif
#include <mex.h>
#include <flann/flann.h>
#include <stdio.h>
#include <string.h>



struct TypedIndex
{
    flann_index_t index;
    flann_datatype_t type;
    void* dataset;
};


template <typename T>
static mxArray* to_mx_array(T value)
{
    mxArray* mat = mxCreateDoubleMatrix(1,1,mxREAL);
    double* ptr = mxGetPr(mat);
    *ptr = value;

    return mat;
}


static void matlabStructToFlannStruct( const mxArray* mexParams, FLANNParameters& flannParams )
{
    flannParams.algorithm = (flann_algorithm_t)(int)*(mxGetPr(mxGetField(mexParams, 0,"algorithm")));

    // kdtree
    flannParams.trees = (int)*(mxGetPr(mxGetField(mexParams, 0,"trees")));
    flannParams.leaf_max_size = (int)*(mxGetPr(mxGetField(mexParams, 0,"leaf_max_size")));

    // kmeans
    flannParams.branching = (int)*(mxGetPr(mxGetField(mexParams, 0,"branching")));
    flannParams.iterations = (int)*(mxGetPr(mxGetField(mexParams, 0,"iterations")));
    flannParams.centers_init = (flann_centers_init_t)(int)*(mxGetPr(mxGetField(mexParams, 0,"centers_init")));
    flannParams.cb_index = (float)*(mxGetPr(mxGetField(mexParams, 0,"cb_index")));

    // autotuned
    flannParams.target_precision = (float)*(mxGetPr(mxGetField(mexParams, 0,"target_precision")));
    flannParams.build_weight = (float)*(mxGetPr(mxGetField(mexParams, 0,"build_weight")));
    flannParams.memory_weight = (float)*(mxGetPr(mxGetField(mexParams, 0,"memory_weight")));
    flannParams.sample_fraction = (float)*(mxGetPr(mxGetField(mexParams, 0,"sample_fraction")));

    // misc
    flannParams.log_level = (flann_log_level_t)(int)*(mxGetPr(mxGetField(mexParams, 0,"log_level")));
    flannParams.random_seed = (int)*(mxGetPr(mxGetField(mexParams, 0,"random_seed")));

    // search
    flannParams.checks = (int)*(mxGetPr(mxGetField(mexParams, 0,"checks")));
    flannParams.eps = (float)*(mxGetPr(mxGetField(mexParams, 0,"eps")));
    flannParams.sorted = (int)*(mxGetPr(mxGetField(mexParams, 0,"sorted")));
    flannParams.max_neighbors = (int)*(mxGetPr(mxGetField(mexParams, 0,"max_neighbors")));
    flannParams.cores = (int)*(mxGetPr(mxGetField(mexParams, 0,"cores")));

    // lsh
    flannParams.table_number_ = (unsigned int)*(mxGetPr(mxGetField(mexParams, 0, "table_number")));
    flannParams.key_size_ = (unsigned int)*(mxGetPr(mxGetField(mexParams, 0, "key_size")));
    flannParams.multi_probe_level_ = (unsigned int)*(mxGetPr(mxGetField(mexParams, 0, "multi_probe_level")));
}

static mxArray* flannStructToMatlabStruct( const FLANNParameters& flannParams )
{
    const char* fieldnames[] = {"algorithm", "checks", "eps", "sorted", "max_neighbors", "cores", "trees", "leaf_max_size", "branching", "iterations", "centers_init", "cb_index", "table_number", "key_size", "multi_probe_level"};
    mxArray* mexParams = mxCreateStructMatrix(1, 1, sizeof(fieldnames)/sizeof(const char*), fieldnames);

    mxSetField(mexParams, 0, "algorithm", to_mx_array(flannParams.algorithm));
    mxSetField(mexParams, 0, "checks", to_mx_array(flannParams.checks));
    mxSetField(mexParams, 0, "eps", to_mx_array(flannParams.eps));
    mxSetField(mexParams, 0, "sorted", to_mx_array(flannParams.sorted));
    mxSetField(mexParams, 0, "max_neighbors", to_mx_array(flannParams.max_neighbors));
    mxSetField(mexParams, 0, "cores", to_mx_array(flannParams.cores));

    mxSetField(mexParams, 0, "trees", to_mx_array(flannParams.trees));
    mxSetField(mexParams, 0, "leaf_max_size", to_mx_array(flannParams.leaf_max_size));
    
    mxSetField(mexParams, 0, "branching", to_mx_array(flannParams.branching));
    mxSetField(mexParams, 0, "iterations", to_mx_array(flannParams.iterations));
    mxSetField(mexParams, 0, "centers_init", to_mx_array(flannParams.centers_init));
    mxSetField(mexParams, 0, "cb_index", to_mx_array(flannParams.cb_index));

    mxSetField(mexParams, 0, "table_number", to_mx_array(flannParams.table_number_));
    mxSetField(mexParams, 0, "key_size", to_mx_array(flannParams.key_size_));
    mxSetField(mexParams, 0, "multi_probe_level", to_mx_array(flannParams.multi_probe_level_));

    return mexParams;
}


static void check_allowed_type(const mxArray* datasetMat)
{
    if (!mxIsSingle(datasetMat) &&
        !mxIsDouble(datasetMat) &&
        !mxIsUint8(datasetMat) &&
        !mxIsInt32(datasetMat)) {
        mexErrMsgTxt("Data type must be floating point single precision, floating point double precision, "
                     "8 bit unsigned integer or 32 bit signed integer");
    }
}


/**
 * Input arguments: dataset (matrix), testset (matrix), n (int),  params (struct)
 * Output arguments: indices(matrix), dists(matrix)
 */
static void _find_nn(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    /* Check the number of input arguments */
    if(nInArray != 4) {
        mexErrMsgTxt("Incorrect number of input arguments, expecting:\n"
                     "dataset, testset, nearest_neighbors, params");
    }

    /* Check the number of output arguments */
    if(nOutArray > 2) {
        mexErrMsgTxt("One or two outputs required.");
    }
    const mxArray* datasetMat = InArray[0];
    const mxArray* testsetMat = InArray[1];
    check_allowed_type(datasetMat);
    check_allowed_type(testsetMat);

    int dcount = mxGetN(datasetMat);
    int length = mxGetM(datasetMat);
    int tcount = mxGetN(testsetMat);

    if (mxGetM(testsetMat) != length) {
        mexErrMsgTxt("Dataset and testset features should have the same size.");
    }

    const mxArray* nnMat = InArray[2];

    if ((mxGetM(nnMat)!=1)||(mxGetN(nnMat)!=1)|| !mxIsNumeric(nnMat)) {
        mexErrMsgTxt("Number of nearest neighbors should be a scalar.");
    }
    int nn = (int)(*mxGetPr(nnMat));

    const mxArray* pStruct = InArray[3];

    if (!mxIsStruct(pStruct)) {
        mexErrMsgTxt("Params must be a struct object.");
    }

    FLANNParameters p;
    matlabStructToFlannStruct(pStruct, p);

    int* result = (int*)malloc(tcount*nn*sizeof(int));
    float* dists = NULL;
    double* ddists = NULL;

    /* do the search */
    if (mxIsSingle(datasetMat)) {
        float* dataset = (float*) mxGetData(datasetMat);
        float* testset = (float*) mxGetData(testsetMat);
        dists = (float*)malloc(tcount*nn*sizeof(float));
        flann_find_nearest_neighbors_float(dataset,dcount,length,testset, tcount, result, dists, nn, &p);
    }
    else if (mxIsDouble(datasetMat)) {
        double* dataset = (double*) mxGetData(datasetMat);
        double* testset = (double*) mxGetData(testsetMat);
        ddists = (double*)malloc(tcount*nn*sizeof(double));
        flann_find_nearest_neighbors_double(dataset,dcount,length,testset, tcount, result, ddists, nn, &p);
    }
    else if (mxIsUint8(datasetMat)) {
        unsigned char* dataset = (unsigned char*) mxGetData(datasetMat);
        unsigned char* testset = (unsigned char*) mxGetData(testsetMat);
        dists = (float*)malloc(tcount*nn*sizeof(float));
        flann_find_nearest_neighbors_byte(dataset,dcount,length,testset, tcount, result, dists, nn, &p);
    }
    else if (mxIsInt32(datasetMat)) {
        int* dataset = (int*) mxGetData(datasetMat);
        int* testset = (int*) mxGetData(testsetMat);
        dists = (float*)malloc(tcount*nn*sizeof(float));
        flann_find_nearest_neighbors_int(dataset,dcount,length,testset, tcount, result, dists, nn, &p);
    }

    /* Allocate memory for Output Matrix */
    OutArray[0] = mxCreateDoubleMatrix(nn, tcount, mxREAL);

    /* Get pointer to Output matrix and store result */
    double* pOut = mxGetPr(OutArray[0]);
    for (int i=0; i<tcount*nn; ++i) {
        pOut[i] = result[i]+1; // matlab uses 1-based indexing
    }
    free(result);

    if (nOutArray > 1) {
        /* Allocate memory for Output Matrix */
        OutArray[1] = mxCreateDoubleMatrix(nn, tcount, mxREAL);

        /* Get pointer to Output matrix and store result*/
        double* pDists = mxGetPr(OutArray[1]);
        if (dists!=NULL) {
            for (int i=0; i<tcount*nn; ++i) {
                pDists[i] = dists[i];
            }
        }
        if (ddists!=NULL) {
            for (int i=0; i<tcount*nn; ++i) {
                pDists[i] = ddists[i];
            }
        }
    }
    if (dists!=NULL) free(dists);
    if (ddists!=NULL) free(ddists);
}

/**
 * Input arguments: index (pointer), testset (matrix), n (int),  params (struct)
 * Output arguments: indices(matrix), dists(matrix)
 */
static void _index_find_nn(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    /* Check the number of input arguments */
    if(nInArray != 4) {
        mexErrMsgTxt("Incorrect number of input arguments");
    }
    /* Check if there is one Output matrix */
    if(nOutArray > 2) {
        mexErrMsgTxt("One or two outputs required.");
    }

    const mxArray* indexMat = InArray[0];
    TypedIndex* typedIndex = *(TypedIndex**)mxGetData(indexMat);

    const mxArray* testsetMat = InArray[1];
    check_allowed_type(testsetMat);

    int tcount = mxGetN(testsetMat);

    const mxArray* nnMat = InArray[2];

    if ((mxGetM(nnMat)!=1)||(mxGetN(nnMat)!=1)) {
        mexErrMsgTxt("Number of nearest neighbors should be a scalar.");
    }
    int nn = (int)(*mxGetPr(nnMat));

    int* result = (int*)malloc(tcount*nn*sizeof(int));
    float* dists = NULL;
    double* ddists = NULL;

    const mxArray* pStruct = InArray[3];

    FLANNParameters p;
    matlabStructToFlannStruct(pStruct, p);

    if (mxIsSingle(testsetMat)) {
        if (typedIndex->type != FLANN_FLOAT32) {
            mexErrMsgTxt("Index type must match testset type");
        }
        float* testset = (float*) mxGetData(testsetMat);
        dists = (float*)malloc(tcount*nn*sizeof(float));
        flann_find_nearest_neighbors_index_float(typedIndex->index,testset, tcount, result, dists, nn, &p);
    }
    else if (mxIsDouble(testsetMat)) {
        if (typedIndex->type != FLANN_FLOAT64) {
            mexErrMsgTxt("Index type must match testset type");
        }
        double* testset = (double*) mxGetData(testsetMat);
        ddists = (double*)malloc(tcount*nn*sizeof(double));
        flann_find_nearest_neighbors_index_double(typedIndex->index,testset, tcount, result, ddists, nn, &p);
    }
    else if (mxIsUint8(testsetMat)) {
        if (typedIndex->type != FLANN_UINT8) {
            mexErrMsgTxt("Index type must match testset type");
        }
        unsigned char* testset = (unsigned char*) mxGetData(testsetMat);
        dists = (float*)malloc(tcount*nn*sizeof(float));
        flann_find_nearest_neighbors_index_byte(typedIndex->index,testset, tcount, result, dists, nn, &p);
    }
    else if (mxIsInt32(testsetMat)) {
        if (typedIndex->type != FLANN_INT32) {
            mexErrMsgTxt("Index type must match testset type");
        }
        int* testset = (int*) mxGetData(testsetMat);
        dists = (float*)malloc(tcount*nn*sizeof(float));
        flann_find_nearest_neighbors_index_int(typedIndex->index,testset, tcount, result, dists, nn, &p);
    }

    /* Allocate memory for Output Matrix */
    OutArray[0] = mxCreateDoubleMatrix(nn, tcount, mxREAL);

    /* Get pointer to Output matrix and store result*/
    double* pOut = mxGetPr(OutArray[0]);
    for (int i=0; i<tcount*nn; ++i) {
        pOut[i] = result[i]+1; // matlab uses 1-based indexing
    }
    free(result);
    if (nOutArray > 1) {
        /* Allocate memory for Output Matrix */
        OutArray[1] = mxCreateDoubleMatrix(nn, tcount, mxREAL);

        /* Get pointer to Output matrix and store result*/
        double* pDists = mxGetPr(OutArray[1]);
        if (dists!=NULL) {
            for (int i=0; i<tcount*nn; ++i) {
                pDists[i] = dists[i];
            }
        }
        if (ddists!=NULL) {
            for (int i=0; i<tcount*nn; ++i) {
                pDists[i] = ddists[i];
            }
        }
    }
    if (dists!=NULL) free(dists);
    if (ddists!=NULL) free(ddists);
}


template<typename T>
T* copy_array(const mxArray* array)
{
    size_t mem_size = mxGetN(array)*mxGetM(array)*sizeof(T);
    void* data = malloc(mem_size);
    memcpy(data,mxGetData(array),mem_size);
    return (T*)data;
}

/**
 * Input arguments: dataset (matrix), params (struct)
 * Output arguments: index (pointer to index), params (struct), speedup(double)
 */
static void _build_index(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    /* Check the number of input arguments */
    if(nInArray != 2) {
        mexErrMsgTxt("Incorrect number of input arguments");
    }
    /* Check the number of output arguments */
    if ((nOutArray == 0)||(nOutArray > 3)) {
        mexErrMsgTxt("Incorrect number of outputs.");
    }
    const mxArray* datasetMat = InArray[0];
    check_allowed_type(datasetMat);



    int dcount = mxGetN(datasetMat);
    int length = mxGetM(datasetMat);


    const mxArray* pStruct = InArray[1];

    /* get index parameters */
    FLANNParameters p;
    matlabStructToFlannStruct(pStruct, p);

    float speedup = -1;

    TypedIndex* typedIndex = new TypedIndex();

    if (mxIsSingle(datasetMat)) {
        float* dataset = copy_array<float>(datasetMat);
        typedIndex->index = flann_build_index_float(dataset,dcount,length, &speedup, &p);
        typedIndex->type = FLANN_FLOAT32;
        typedIndex->dataset = dataset;
    }
    else if (mxIsDouble(datasetMat)) {
        double* dataset = copy_array<double>(datasetMat);
        typedIndex->index = flann_build_index_double(dataset,dcount,length, &speedup, &p);
        typedIndex->type = FLANN_FLOAT64;
        typedIndex->dataset = dataset;
    }
    else if (mxIsUint8(datasetMat)) {
        unsigned char* dataset = copy_array<unsigned char>(datasetMat);
        typedIndex->index = flann_build_index_byte(dataset,dcount,length, &speedup, &p);
        typedIndex->type = FLANN_UINT8;
        typedIndex->dataset = dataset;
    }
    else if (mxIsInt32(datasetMat)) {
        int* dataset = copy_array<int>(datasetMat);
        typedIndex->index = flann_build_index_int(dataset,dcount,length, &speedup, &p);
        typedIndex->type = FLANN_INT32;
        typedIndex->dataset = dataset;
    }

    mxClassID classID;
    if (sizeof(flann_index_t)==4) {
        classID = mxUINT32_CLASS;
    }
    else if (sizeof(flann_index_t)==8) {
        classID = mxUINT64_CLASS;
    }

    /* Allocate memory for Output Matrix */
    OutArray[0] = mxCreateNumericMatrix(1, 1, classID, mxREAL);

    /* Get pointer to Output matrix and store result*/
    TypedIndex** pOut = (TypedIndex**)mxGetData(OutArray[0]);
    pOut[0] = typedIndex;

    if (nOutArray > 1) {
        OutArray[1] = flannStructToMatlabStruct(p);
    }
    if (nOutArray > 2) {
        OutArray[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
        double* pSpeedup = mxGetPr(OutArray[2]);

        *pSpeedup = speedup;
    }
}

/**
 * Inputs: index (index pointer)
 */
static void _free_index(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    /* Check the number of input arguments */
    if(!((nInArray == 1)&&((mxGetN(InArray[0])*mxGetM(InArray[0]))==1))) {
        mexErrMsgTxt("Expecting a single scalar argument: the index ID");
    }
    TypedIndex* typedIndex = *(TypedIndex**)mxGetData(InArray[0]);
    if (typedIndex->type==FLANN_FLOAT32) {
        flann_free_index_float(typedIndex->index, NULL);
    }
    else if (typedIndex->type==FLANN_FLOAT64) {
        flann_free_index_double(typedIndex->index, NULL);
    }
    else if (typedIndex->type==FLANN_UINT8) {
        flann_free_index_byte(typedIndex->index, NULL);
    }
    else if (typedIndex->type==FLANN_INT32) {
        flann_free_index_int(typedIndex->index, NULL);
    }
    free(typedIndex->dataset);
    delete typedIndex;
}

/**
 * Inputs: level
 */
static void _set_log_level(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    if (nInArray != 1) {
        mexErrMsgTxt("Incorrect number of input arguments: expecting log_level");
    }

    const mxArray* llMat = InArray[0];

    if ((mxGetM(llMat)!=1)||(mxGetN(llMat)!=1)|| !mxIsNumeric(llMat)) {
        mexErrMsgTxt("Log Level should be a scalar.");
    }
    int log_level = (int)(*mxGetPr(llMat));

    flann_log_verbosity(log_level);

}

/**
 * Inputs: type (flann_distance_t), order(int)
 */
static void _set_distance_type(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    if( ((nInArray != 1)&&(nInArray != 2))) {
        mexErrMsgTxt("Incorrect number of input arguments");
    }

    const mxArray* distMat = InArray[0];

    if ((mxGetM(distMat)!=1)||(mxGetN(distMat)!=1)|| !mxIsNumeric(distMat)) {
        mexErrMsgTxt("Distance type should be a scalar.");
    }
    int distance_type = (int)(*mxGetPr(distMat));

    int order = 0;
    if (nInArray==2) {
        const mxArray* ordMat = InArray[1];
        if ((mxGetM(ordMat)!=1)||(mxGetN(ordMat)!=1)|| !mxIsNumeric(ordMat)) {
            mexErrMsgTxt("Distance order should be a scalar.");
        }

        order = (int)(*mxGetPr(ordMat));
    }
    flann_set_distance_type((flann_distance_t)distance_type, order);
}


/**
 * Inputs: index (index pointer), filename (string)
 */
static void _save_index(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    /* Check the number of input arguments */
    if(nInArray != 2) {
        mexErrMsgTxt("Incorrect number of input arguments");
    }

    const mxArray* indexMat = InArray[0];
    TypedIndex* typedIndex = *(TypedIndex**)mxGetData(indexMat);

    // get the selector
    if(!mxIsChar(InArray[1])) {
        mexErrMsgTxt("'filename' should be a string");
    }
    char filename[128];
    mxGetString(InArray[1],filename,128);

    if (typedIndex->type==FLANN_FLOAT32) {
        flann_save_index_float(typedIndex->index, filename);
    }
    else if (typedIndex->type==FLANN_FLOAT64) {
        flann_save_index_double(typedIndex->index, filename);
    }
    else if (typedIndex->type==FLANN_UINT8) {
        flann_save_index_byte(typedIndex->index, filename);
    }
    else if (typedIndex->type==FLANN_INT32) {
        flann_save_index_int(typedIndex->index, filename);
    }
}


/**
 * Inputs: filename (string), matrix
 */
static void _load_index(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    if(nInArray != 2) {
        mexErrMsgTxt("Incorrect number of input arguments");
    }
    // get the selector
    if(!mxIsChar(InArray[0])) {
        mexErrMsgTxt("'filename' should be a string");
    }
    char filename[128];
    mxGetString(InArray[0],filename,128);

    const mxArray* datasetMat = InArray[1];
    check_allowed_type(datasetMat);

    int dcount = mxGetN(datasetMat);
    int length = mxGetM(datasetMat);

    TypedIndex* typedIndex = new TypedIndex();

    if (mxIsSingle(datasetMat)) {
        float* dataset = copy_array<float>(datasetMat);
        typedIndex->index = flann_load_index_float(filename, dataset,dcount,length);
        typedIndex->type = FLANN_FLOAT32;
        typedIndex->dataset = dataset;
    }
    else if (mxIsDouble(datasetMat)) {
        double* dataset = copy_array<double>(datasetMat);
        typedIndex->index = flann_load_index_double(filename, dataset,dcount,length);
        typedIndex->type = FLANN_FLOAT64;
        typedIndex->dataset = dataset;
    }
    else if (mxIsUint8(datasetMat)) {
        unsigned char* dataset = copy_array<unsigned char>(datasetMat);
        typedIndex->index = flann_load_index_byte(filename, dataset,dcount,length);
        typedIndex->type = FLANN_UINT8;
        typedIndex->dataset = dataset;
    }
    else if (mxIsInt32(datasetMat)) {
        int* dataset = copy_array<int>(datasetMat);
        typedIndex->index = flann_load_index_int(filename, dataset,dcount,length);
        typedIndex->type = FLANN_INT32;
        typedIndex->dataset = dataset;
    }

    mxClassID classID;
    if (sizeof(flann_index_t)==4) {
        classID = mxUINT32_CLASS;
    }
    else if (sizeof(flann_index_t)==8) {
        classID = mxUINT64_CLASS;
    }

    /* Allocate memory for Output Matrix */
    OutArray[0] = mxCreateNumericMatrix(1, 1, classID, mxREAL);

    /* Get pointer to Output matrix and store result*/
    TypedIndex** pOut = (TypedIndex**)mxGetData(OutArray[0]);
    pOut[0] = typedIndex;
}

flann::IndexParams _create_parameters_mex(FLANNParameters* p)
{
    flann::IndexParams params;

    params["algorithm"] = p->algorithm;

    params["checks"] = p->checks;
    params["cb_index"] = p->cb_index;
    params["eps"] = p->eps;

    if (p->algorithm == FLANN_INDEX_KDTREE) {
        params["trees"] = p->trees;
    }

    if (p->algorithm == FLANN_INDEX_KDTREE_SINGLE) {
        params["trees"] = p->trees;
        params["leaf_max_size"] = p->leaf_max_size;
    }

#ifdef FLANN_USE_CUDA
    if (p->algorithm == FLANN_INDEX_KDTREE_CUDA) {
        params["leaf_max_size"] = p->leaf_max_size;
    }
#endif

    if (p->algorithm == FLANN_INDEX_KMEANS) {
        params["branching"] = p->branching;
        params["iterations"] = p->iterations;
        params["centers_init"] = p->centers_init;
    }

    if (p->algorithm == FLANN_INDEX_AUTOTUNED) {
        params["target_precision"] = p->target_precision;
        params["build_weight"] = p->build_weight;
        params["memory_weight"] = p->memory_weight;
        params["sample_fraction"] = p->sample_fraction;
    }

    if (p->algorithm == FLANN_INDEX_HIERARCHICAL) {
        params["branching"] = p->branching;
        params["centers_init"] = p->centers_init;
        params["trees"] = p->trees;
        params["leaf_max_size"] = p->leaf_max_size;
    }

    if (p->algorithm == FLANN_INDEX_LSH) {
        params["table_number"] = p->table_number_;
        params["key_size"] = p->key_size_;
        params["multi_probe_level"] = p->multi_probe_level_;
    }

    params["log_level"] = p->log_level;
    params["random_seed"] = p->random_seed;

    return params;
}

flann::SearchParams _create_search_params_mex(FLANNParameters* p)
{
    flann::SearchParams params;
    params.checks = p->checks;
    params.eps = p->eps;
    params.sorted = p->sorted;
    params.max_neighbors = p->max_neighbors;
    params.cores = p->cores;

    return params;
}

template<typename Distance>
int __flann_radius_search_mex(typename Distance::ElementType* dataset,  int rows, int cols, 
                          typename Distance::ElementType* testset, int tcount,
                          std::vector<std::vector<int> >& indices,
                          std::vector<std::vector<typename Distance::ResultType> >& dists,
                          float radius,
                          FLANNParameters* flann_params, Distance d = Distance())
{
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    try {
        if (flann_params->random_seed>0) {
            seed_random(flann_params->random_seed);
        }

        flann::IndexParams params = _create_parameters_mex(flann_params);
        flann::Index<Distance>* index = new flann::Index<Distance>(Matrix<ElementType>(dataset,rows,cols), params, d);
        index->buildIndex();

        flann::SearchParams search_params = _create_search_params_mex(flann_params);
        int count = index->radiusSearch(flann::Matrix<ElementType>(testset, tcount, index->veclen()),
                                        indices,
                                        dists, radius, search_params );

        delete index;
        return count;
    }
    catch (std::runtime_error& e) {
        std::string estr(e.what());
        estr = "Caught exception: " + estr;
        mexErrMsgTxt(estr.c_str());
        return -1;
    }
}

template<typename Distance>
int __flann_radius_search_index_mex(flann_index_t index_ptr,
                          typename Distance::ElementType* testset, int tcount,
                          std::vector<std::vector<int> >& indices,
                          std::vector<std::vector<typename Distance::ResultType> >& dists,
                          float radius,
                          FLANNParameters* flann_params)
{
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    try {
        if (index_ptr==NULL) {
            mexErrMsgTxt("Invalid index");
            return -1;
        }
        if (flann_params->random_seed>0) {
            seed_random(flann_params->random_seed);
        }
        flann::Index<Distance>* index = (flann::Index<Distance>*)index_ptr;

        flann::SearchParams search_params = _create_search_params_mex(flann_params);
        int count = index->radiusSearch(flann::Matrix<ElementType>(testset, tcount, index->veclen()),
                                        indices,
                                        dists, radius, search_params );


        return count;
    }
    catch (std::runtime_error& e) {
        std::string estr(e.what());
        estr = "Caught exception: " + estr;
        mexErrMsgTxt(estr.c_str());
        return -1;
    }
}

extern flann_distance_t flann_distance_type;
extern int flann_distance_order;

template<typename T, typename R>
int _flann_radius_search_mex(T* dataset, int rows, int cols, 
                         T* testset,
                         int tcount,
                         std::vector<std::vector<int> >& indices,
                         std::vector<std::vector<R> >& dists,
                         float radius,
                         FLANNParameters* flann_params)
{
    if (flann_distance_type==FLANN_DIST_EUCLIDEAN) {
        return __flann_radius_search_mex<L2<T> >(dataset, rows, cols, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MANHATTAN) {
        return __flann_radius_search_mex<L1<T> >(dataset, rows, cols, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MINKOWSKI) {
        return __flann_radius_search_mex<MinkowskiDistance<T> >(dataset, rows, cols, testset, tcount, indices, dists, radius, flann_params, MinkowskiDistance<T>(flann_distance_order));
    }
    else if (flann_distance_type==FLANN_DIST_HIST_INTERSECT) {
        return __flann_radius_search_mex<HistIntersectionDistance<T> >(dataset, rows, cols, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_HELLINGER) {
        return __flann_radius_search_mex<HellingerDistance<T> >(dataset, rows, cols, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_CHI_SQUARE) {
        return __flann_radius_search_mex<ChiSquareDistance<T> >(dataset, rows, cols, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_KULLBACK_LEIBLER) {
        return __flann_radius_search_mex<KL_Divergence<T> >(dataset, rows, cols, testset, tcount, indices, dists, radius, flann_params);
    }
    else {
        mexErrMsgTxt( "Distance type unsupported in the C bindings, use the C++ bindings instead");
        return -1;
    }
}

template<typename T, typename R>
int _flann_radius_search_index_mex(flann_index_t index_ptr,
                         T* testset,
                         int tcount,
                         std::vector<std::vector<int> >& indices,
                         std::vector<std::vector<R> >& dists,
                         float radius,
                         FLANNParameters* flann_params)
{
    if (flann_distance_type==FLANN_DIST_EUCLIDEAN) {
        return __flann_radius_search_index_mex<L2<T> >(index_ptr, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MANHATTAN) {
        return __flann_radius_search_index_mex<L1<T> >(index_ptr, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MINKOWSKI) {
        return __flann_radius_search_index_mex<MinkowskiDistance<T> >(index_ptr, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_HIST_INTERSECT) {
        return __flann_radius_search_index_mex<HistIntersectionDistance<T> >(index_ptr, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_HELLINGER) {
        return __flann_radius_search_index_mex<HellingerDistance<T> >(index_ptr, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_CHI_SQUARE) {
        return __flann_radius_search_index_mex<ChiSquareDistance<T> >(index_ptr, testset, tcount, indices, dists, radius, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_KULLBACK_LEIBLER) {
        return __flann_radius_search_index_mex<KL_Divergence<T> >(index_ptr, testset, tcount, indices, dists, radius, flann_params);
    }
    else {
        mexErrMsgTxt( "Distance type unsupported in the C bindings, use the C++ bindings instead");
        return -1;
    }
}

/**
 * Input arguments: dataset (matrix), testset (matrix), radius (double),  params (struct)
 * Output arguments: indices(matrix), dists(matrix)
 */
static void _find_rn(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    /* Check the number of input arguments */
    if(nInArray != 4) {
        mexErrMsgTxt("Incorrect number of input arguments, expecting:\n"
                     "dataset, testset, nearest_neighbors, params");
    }

    /* Check the number of output arguments */
    if(nOutArray > 2) {
        mexErrMsgTxt("One or two outputs required.");
    }
    const mxArray* datasetMat = InArray[0];
    const mxArray* testsetMat = InArray[1];
    check_allowed_type(datasetMat);
    check_allowed_type(testsetMat);

    int dcount = mxGetN(datasetMat);
    int length = mxGetM(datasetMat);
    int tcount = mxGetN(testsetMat);

    if (mxGetM(testsetMat) != length) {
        mexErrMsgTxt("Dataset and testset features should have the same size.");
    }

    const mxArray* rMat = InArray[2];

    if ((mxGetM(rMat)!=1)||(mxGetN(rMat)!=1)|| !mxIsNumeric(rMat)) {
        mexErrMsgTxt("Radius should be a scalar.");
    }
    double radius = (double)(*mxGetPr(rMat));

    const mxArray* pStruct = InArray[3];

    if (!mxIsStruct(pStruct)) {
        mexErrMsgTxt("Params must be a struct object.");
    }

    FLANNParameters p;
    matlabStructToFlannStruct(pStruct, p);

    std::vector<std::vector<int> > result;
    std::vector<std::vector<float> > dists;
    std::vector<std::vector<double> > ddists;

    /* do the search */
    if (mxIsSingle(datasetMat)) {
        float* dataset = (float*) mxGetData(datasetMat);
        float* testset = (float*) mxGetData(testsetMat);
        _flann_radius_search_mex(dataset,dcount,length,testset, tcount, result, dists, radius, &p);
    }
    else if (mxIsDouble(datasetMat)) {
        double* dataset = (double*) mxGetData(datasetMat);
        double* testset = (double*) mxGetData(testsetMat);
        _flann_radius_search_mex(dataset,dcount,length,testset, tcount, result, ddists, radius, &p);
    }
    else if (mxIsUint8(datasetMat)) {
        unsigned char* dataset = (unsigned char*) mxGetData(datasetMat);
        unsigned char* testset = (unsigned char*) mxGetData(testsetMat);
        _flann_radius_search_mex(dataset,dcount,length,testset, tcount, result, dists, radius, &p);
    }
    else if (mxIsInt32(datasetMat)) {
        int* dataset = (int*) mxGetData(datasetMat);
        int* testset = (int*) mxGetData(testsetMat);
        _flann_radius_search_mex(dataset,dcount,length,testset, tcount, result, dists, radius, &p);
    }

    OutArray[0] = mxCreateCellMatrix(1, tcount);
    if (nOutArray > 1) {
        OutArray[1] = mxCreateCellMatrix(1, tcount);
    }
    for (size_t i=0; i<result.size(); ++i) {
        mxArray* idxs = mxCreateDoubleMatrix(result[i].size(), 1, mxREAL);
        double *pOut = mxGetPr(idxs);
        for (size_t j=0; j<result[i].size(); ++j)
            pOut[j] = result[i][j] + 1; // matlab uses 1-based indexing
        mxSetCell(OutArray[0], i, idxs);

        if (nOutArray > 1) {
            mxArray* mdists = mxCreateDoubleMatrix(result[i].size(), 1, mxREAL);
            double* pDists = mxGetPr(mdists);
            for (size_t j=0; j<result[i].size(); ++j) {
                if (!dists.empty()) {
                    pDists[j] = dists[i][j];
                }
                if (!ddists.empty()) {
                    pDists[j] = ddists[i][j];
                }
            }
            mxSetCell(OutArray[1], i, mdists);
        }
    }
}

/**
 * Input arguments: index (pointer), testset (matrix), radius (double),  params (struct)
 * Output arguments: indices(matrix), dists(matrix)
 */
static void _index_find_rn(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    /* Check the number of input arguments */
    if(nInArray != 4) {
        mexErrMsgTxt("Incorrect number of input arguments");
    }
    /* Check if there is one Output matrix */
    if(nOutArray > 2) {
        mexErrMsgTxt("One or two outputs required.");
    }

    const mxArray* indexMat = InArray[0];
    TypedIndex* typedIndex = *(TypedIndex**)mxGetData(indexMat);

    const mxArray* testsetMat = InArray[1];
    check_allowed_type(testsetMat);

    int tcount = mxGetN(testsetMat);

    const mxArray* rMat = InArray[2];

    if ((mxGetM(rMat)!=1)||(mxGetN(rMat)!=1)) {
        mexErrMsgTxt("Radius should be a scalar.");
    }
    double radius = (double)(*mxGetPr(rMat));

    const mxArray* pStruct = InArray[3];

    FLANNParameters p;
    matlabStructToFlannStruct(pStruct, p);

    std::vector<std::vector<int> > result;
    std::vector<std::vector<float> > dists;
    std::vector<std::vector<double> > ddists;

    if (mxIsSingle(testsetMat)) {
        if (typedIndex->type != FLANN_FLOAT32) {
            mexErrMsgTxt("Index type must match testset type");
        }
        float* testset = (float*) mxGetData(testsetMat);
        _flann_radius_search_index_mex(typedIndex->index, testset, tcount, result, dists, radius, &p);
    }
    else if (mxIsDouble(testsetMat)) {
        if (typedIndex->type != FLANN_FLOAT64) {
            mexErrMsgTxt("Index type must match testset type");
        }
        double* testset = (double*) mxGetData(testsetMat);
        _flann_radius_search_index_mex(typedIndex->index, testset, tcount, result, ddists, radius, &p);
    }
    else if (mxIsUint8(testsetMat)) {
        if (typedIndex->type != FLANN_UINT8) {
            mexErrMsgTxt("Index type must match testset type");
        }
        unsigned char* testset = (unsigned char*) mxGetData(testsetMat);
        _flann_radius_search_index_mex(typedIndex->index, testset, tcount, result, dists, radius, &p);
    }
    else if (mxIsInt32(testsetMat)) {
        if (typedIndex->type != FLANN_INT32) {
            mexErrMsgTxt("Index type must match testset type");
        }
        int* testset = (int*) mxGetData(testsetMat);
        _flann_radius_search_index_mex(typedIndex->index, testset, tcount, result, dists, radius, &p);
    }

    OutArray[0] = mxCreateCellMatrix(1, tcount);
    if (nOutArray > 1) {
        OutArray[1] = mxCreateCellMatrix(1, tcount);
    }
    for (size_t i=0; i<result.size(); ++i) {
        mxArray* idxs = mxCreateDoubleMatrix(result[i].size(), 1, mxREAL);
        double *pOut = mxGetPr(idxs);
        for (size_t j=0; j<result[i].size(); ++j)
            pOut[j] = result[i][j] + 1; // matlab uses 1-based indexing
        mxSetCell(OutArray[0], i, idxs);

        if (nOutArray > 1) {
            mxArray* mdists = mxCreateDoubleMatrix(result[i].size(), 1, mxREAL);
            double* pDists = mxGetPr(mdists);
            for (size_t j=0; j<result[i].size(); ++j) {
                if (!dists.empty()) {
                    pDists[j] = dists[i][j];
                }
                if (!ddists.empty()) {
                    pDists[j] = ddists[i][j];
                }
            }
            mxSetCell(OutArray[1], i, mdists);
        }
    }
}

struct mexFunctionEntry
{
    const char* name;
    void (* function)(int, mxArray**, int, const mxArray**);
};

static mexFunctionEntry __functionTable[] = {
    { "find_nn", &_find_nn},
    { "build_index", &_build_index},
    { "index_find_nn", &_index_find_nn},
    { "free_index", &_free_index},
    { "save_index", &_save_index},
    { "load_index", &_load_index},
    { "set_log_level", &_set_log_level},
    { "set_distance_type", &_set_distance_type},
    { "find_rn", &_find_rn},
    { "index_find_rn", &_index_find_rn}
};


static void print_selector_error()
{
    char buf[512];
    char* msg = buf;

    sprintf(msg, "%s", "Expecting first argument to be one of: ");
    msg = buf+strlen(buf);
    for (int i=0; i<sizeof(__functionTable)/sizeof(mexFunctionEntry); ++i) {
        if (i!=0) {
            sprintf(msg,", ");
            msg = buf+strlen(buf);
        }
        sprintf(msg, "%s", __functionTable[i].name);
        msg = buf+strlen(buf);
    }

    mexErrMsgTxt(buf);
}


void mexFunction(int nOutArray, mxArray* OutArray[], int nInArray, const mxArray* InArray[])
{
    // get the selector
    if((nInArray == 0)|| !mxIsChar(InArray[0])) {
        print_selector_error();
    }
    char selector[128];
    mxGetString(InArray[0],selector,128);

    // check if function with that name is present
    int idx = 0;
    for (idx = 0; idx<sizeof(__functionTable)/sizeof(mexFunctionEntry); ++idx) {
        if (strcmp(__functionTable[idx].name, selector)==0) {
            break;
        }
    }
    if (idx==sizeof(__functionTable)/sizeof(mexFunctionEntry)) {
        print_selector_error();
    }

    // now call the function
    __functionTable[idx].function(nOutArray,OutArray, nInArray-1, InArray+1);
}
