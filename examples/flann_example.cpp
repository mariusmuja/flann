
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <stdio.h>
#include <iostream>
#include <algorithm>

using namespace flann;

int main(int argc, char** argv)
{
    int nn = 3;

    float dataset_ptr[] = {1, 1,
			   			   3, 3,
			               3, 4,
			               7, 7,
			               7, 6};
	float* dataset_ptrh = new float[10];
	std::copy(dataset_ptr, dataset_ptr + 10, dataset_ptrh);

	float query_ptr[] = {3, 1};
	float* query_ptrh = new float[2];
	std::copy(query_ptr, query_ptr + 2, query_ptrh);

    Matrix<float> dataset (dataset_ptrh, 5, 2);
    Matrix<float> query (query_ptrh, 1, 2);

    Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
    Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

    // construct an randomized kd-tree index using 4 kd-trees
    Index<L2<float> > index(dataset, flann::KDTreeSingleIndexParams());
    index.buildIndex();                                                                                               

    // do a knn search, using 128 checks
    index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

	std::cout << indices[0][0] << " " << indices[0][1] << " " << indices[0][2] << std::endl;

    delete[] dataset.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
    
    return 0;
}
