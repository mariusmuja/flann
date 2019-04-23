/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2012  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2012  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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


#ifndef FLANN_TESTS_H_
#define FLANN_TESTS_H_

#include <flann/util/matrix.h>
#include <vector>
#include <set>

#define GTEST_PRECISION 0.0004


template<typename T>
float compute_precision(const flann::Matrix<T>& match, const flann::Matrix<T>& indices)
{
	int count = 0;

	assert(match.rows == indices.rows);
	size_t nn = std::min(match.cols, indices.cols);

	for (size_t i=0; i<match.rows; ++i) {
		for (size_t j=0;j<nn;++j) {
			for (size_t k=0;k<nn;++k) {
				if (match[i][j]==indices[i][k]) {
					count ++;
				}
			}
		}
	}

	return float(count)/(nn*match.rows);
}

/** @brief Compare the distances for match accuracies
 * This is more precise: e.g. when you ask for the top 10 neighbors and they all get the same distance,
 * you might have 100 other neighbors that are at the same distance and simply matching the indices is not the way to go
 * @param gt_dists the ground truth best distances
 * @param dists the distances of the computed nearest neighbors
 * @param tol tolerance at which distanceare considered equal
 * @return
 */
template<typename T>
float computePrecisionDiscrete(const flann::Matrix<T>& gt_dists, const flann::Matrix<T>& dists)
{
	int count = 0;

	assert(gt_dists.rows == dists.rows);
	size_t nn = std::min(gt_dists.cols, dists.cols);
	std::vector<T> gt_sorted_dists(nn), sorted_dists(nn), intersection(nn);

	for (size_t i = 0; i < gt_dists.rows; ++i)
	{
		std::copy(gt_dists[i], gt_dists[i] + nn, gt_sorted_dists.begin());
		std::sort(gt_sorted_dists.begin(), gt_sorted_dists.end());
		std::copy(dists[i], dists[i] + nn, sorted_dists.begin());
		std::sort(sorted_dists.begin(), sorted_dists.end());
		typename std::vector<T>::iterator end = std::set_intersection(gt_sorted_dists.begin(), gt_sorted_dists.end(),
				sorted_dists.begin(), sorted_dists.end(),
				intersection.begin());
		count += (end - intersection.begin());
	}

	return float(count) / (nn * gt_dists.rows);
}


const char* index_type_to_name(flann_algorithm_t index_type)
{
	switch (index_type) {
	case FLANN_INDEX_LINEAR: return "linear";
	case FLANN_INDEX_KDTREE: return "randomized kd-tree";
	case FLANN_INDEX_KMEANS: return "k-means";
	case FLANN_INDEX_COMPOSITE: return "composite";
	case FLANN_INDEX_KDTREE_SINGLE: return "single kd-tree";
	case FLANN_INDEX_HIERARCHICAL: return "hierarchical";
	case FLANN_INDEX_LSH: return "LSH";
#ifdef FLANN_USE_CUDA
	case FLANN_INDEX_KDTREE_CUDA: return "kd-tree CUDA";
#endif
#ifdef FLANN_USE_OPENCL
	case FLANN_INDEX_KMEANS_OPENCL: return "k-means OpenCL";
	case FLANN_INDEX_HIERARCHICAL_OPENCL: return "hierarchical OpenCL";
#endif
	case FLANN_INDEX_SAVED: return "saved";
	case FLANN_INDEX_AUTOTUNED: return "autotuned";
	default: return "(unknown)";
	}
}


class FLANNTestFixture : public ::testing::Test {
protected:
	clock_t start_time_;

	void start_timer(const std::string& message = "")
	{
		if (!message.empty()) {
			printf("%s", message.c_str());
			fflush(stdout);
		}
		start_time_ = clock();
	}

	double stop_timer()
	{
		return double(clock()-start_time_)/CLOCKS_PER_SEC;
	}


	template<typename Distance>
	void TestSearch(const flann::Matrix<typename Distance::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Distance::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Distance::ResultType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			float expected_precision,
			const flann::Matrix<size_t>& gt_indices,
			const flann::Matrix<typename Distance::ResultType>& gt_dists = flann::Matrix<typename Distance::ResultType>())
	{
		flann::seed_random(0);
		Index<Distance> index(data, index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex();
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision;
		if (gt_dists.ptr()==NULL) {
			precision = compute_precision(gt_indices, indices);
		}
		else {
			precision = computePrecisionDiscrete(gt_dists, dists);
		}
		EXPECT_GE(precision, expected_precision);
		printf("Precision: %g\n", precision);
	}

	template<typename Distance>
	void TestSearch2(const flann::Matrix<typename Distance::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Distance::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Distance::ResultType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			float expected_precision,
			const flann::Matrix<size_t>& gt_indices,
			const flann::Matrix<typename Distance::ResultType>& gt_dists = flann::Matrix<typename Distance::ResultType>())
	{
		flann::seed_random(0);
		Index<Distance> index(index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex(data);
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision;
		if (gt_dists.ptr()==NULL) {
			precision = compute_precision(gt_indices, indices);
		}
		else {
			precision = computePrecisionDiscrete(gt_dists, dists);
		}
		EXPECT_GE(precision, expected_precision);
		printf("Precision: %g\n", precision);
	}

	template<typename Distance>
	void TestAddIncremental(const flann::Matrix<typename Distance::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Distance::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Distance::ResultType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			float expected_precision,
			flann::Matrix<size_t>& gt_indices,
			const flann::Matrix<typename Distance::ResultType>& gt_dists = flann::Matrix<typename Distance::ResultType>(),
			bool retestDists = false)
	{
		flann::seed_random(0);
		size_t size1 = data.rows/2-1;
		size_t size2 = data.rows-size1;
		Matrix<typename Distance::ElementType> data1(data[0], size1, data.cols);
		Matrix<typename Distance::ElementType> data2(data[size1], size2, data.cols);
		Index<Distance> index(data1, index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex();
		EXPECT_EQ(index.size(), data1.rows);

		index.addPoints(data2);
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index.size(), data.rows);

		start_timer("Searching KNN...");
		index.knnSearch(query, indices, dists, knn, search_params);
		printf("done (%g seconds)\n", stop_timer());

		float precision;
		if (gt_dists.ptr()==NULL) {
			precision = compute_precision(gt_indices, indices);
		}
		else {
			precision = computePrecisionDiscrete(gt_dists, dists);
		}
		EXPECT_GE(precision, expected_precision);
		printf("Precision: %g\n", precision);

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				EXPECT_EQ(index.getPoint(indices[i][j]), data[indices[i][j]]);
			}
		}

		// save and re-load the index and make sure everything still checks out
		index.save("test_saved_index.idx");
		Index<Distance > index2(data, flann::SavedIndexParams("test_saved_index.idx"));
		index2.buildIndex();

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index2.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index2.size(), data.rows);

		flann::Matrix<size_t> indices2(new size_t[query.rows*knn], query.rows, knn);
		flann::Matrix<typename Distance::ResultType> dists2(new typename Distance::ResultType[query.rows*knn], query.rows, knn);
		start_timer("Searching KNN after saving and reloading index...");
		index2.knnSearch(query, indices2, dists2, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				if (retestDists) {
					EXPECT_NEAR(dists[i][j], dists2[i][j], 6);
				} else {
					EXPECT_EQ(indices[i][j], indices2[i][j]);
				}
			}
		}
		delete[] indices2.ptr();
		delete[] dists2.ptr();
	}

	template<typename Distance>
	void TestAddIncremental2(const flann::Matrix<typename Distance::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Distance::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Distance::ResultType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			float expected_precision,
			flann::Matrix<size_t>& gt_indices,
			const flann::Matrix<typename Distance::ResultType>& gt_dists = flann::Matrix<typename Distance::ResultType>(),
			bool retestDists = false)
	{
		flann::seed_random(0);
		size_t size1 = data.rows/2+1;
		size_t size2 = data.rows-size1;
		Matrix<typename Distance::ElementType> data1(data[0], size1, data.cols);
		Matrix<typename Distance::ElementType> data2(data[size1], size2, data.cols);
		Index<Distance> index(data1, index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex();
		EXPECT_EQ(index.size(), data1.rows);

		index.addPoints(data2);
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index.size(), data.rows);

		start_timer("Searching KNN...");
		index.knnSearch(query, indices, dists, knn, search_params);
		printf("done (%g seconds)\n", stop_timer());

		float precision;
		if (gt_dists.ptr()==NULL) {
			precision = compute_precision(gt_indices, indices);
		}
		else {
			precision = computePrecisionDiscrete(gt_dists, dists);
		}
		EXPECT_GE(precision, expected_precision);
		printf("Precision: %g\n", precision);

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				EXPECT_EQ(index.getPoint(indices[i][j]), data[indices[i][j]]);
			}
		}

		// save and re-load the index and make sure everything still checks out
		index.save("test_saved_index.idx");
		Index<Distance > index2(data, flann::SavedIndexParams("test_saved_index.idx"));
		index2.buildIndex();

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index2.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index2.size(), data.rows);

		flann::Matrix<size_t> indices2(new size_t[query.rows*knn], query.rows, knn);
		flann::Matrix<typename Distance::ResultType> dists2(new typename Distance::ResultType[query.rows*knn], query.rows, knn);
		start_timer("Searching KNN after saving and reloading index...");
		index2.knnSearch(query, indices2, dists2, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				if (retestDists) {
					EXPECT_NEAR(dists[i][j], dists2[i][j], 6);
				} else {
					EXPECT_EQ(indices[i][j], indices2[i][j]);
				}
			}
		}
		delete[] indices2.ptr();
		delete[] dists2.ptr();
	}


	template<typename Distance>
	void TestSave(const flann::Matrix<typename Distance::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Distance::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Distance::ResultType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			float expected_precision,
			flann::Matrix<size_t>& gt_indices,
			const flann::Matrix<typename Distance::ResultType>& gt_dists = flann::Matrix<typename Distance::ResultType>(),
			bool retestDists = false)
	{
		flann::seed_random(0);
		Index<Distance> index(data, index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex();
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index.size(), data.rows);

		start_timer("Searching KNN...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision;
		if (gt_dists.ptr()==NULL) {
			precision = compute_precision(gt_indices, indices);
		}
		else {
			precision = computePrecisionDiscrete(gt_dists, dists);
		}
		printf("Precision: %g\n", precision);
		EXPECT_GE(precision, expected_precision);

		printf("Saving index\n");
		index.save("test_saved_index.idx");

		printf("Loading index\n");
		Index<Distance > index2(data, flann::SavedIndexParams("test_saved_index.idx"));
		index2.buildIndex();

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index2.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index2.size(), data.rows);

		flann::Matrix<size_t> indices2(new size_t[query.rows*knn], query.rows, knn);
		flann::Matrix<typename Distance::ResultType> dists2(new typename Distance::ResultType[query.rows*knn], query.rows, knn);
		start_timer("Searching KNN after saving and reloading index...");
		index2.knnSearch(query, indices2, dists2, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision2;
		if (gt_dists.ptr()==NULL) {
			precision2 = compute_precision(gt_indices, indices2);
		}
		else {
			precision2 = computePrecisionDiscrete(gt_dists, dists2);
		}
		printf("Precision: %g\n", precision2);
		EXPECT_NEAR(precision, precision2, GTEST_PRECISION);

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				if (retestDists) {
					EXPECT_NEAR(dists[i][j], dists2[i][j], 6);
				} else {
					EXPECT_EQ(indices[i][j], indices2[i][j]);
				}
			}
		}
		delete[] indices2.ptr();
		delete[] dists2.ptr();
	}


	template<typename Distance>
	void TestCopy(const flann::Matrix<typename Distance::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Distance::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Distance::ResultType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			float expected_precision,
			flann::Matrix<size_t>& gt_indices,
			const flann::Matrix<typename Distance::ResultType>& gt_dists = flann::Matrix<typename Distance::ResultType>())
	{
		flann::seed_random(0);
		Index<Distance> index(data, index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex();
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision;
		if (gt_dists.ptr()==NULL) {
			precision = compute_precision(gt_indices, indices);
		}
		else {
			precision = computePrecisionDiscrete(gt_dists, dists);
		}
		printf("Precision: %g\n", precision);
		EXPECT_GE(precision, expected_precision);

		// test copy constructor
		Index<Distance> index2(index);

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index2.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index2.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision2;
		if (gt_dists.ptr()==NULL) {
			precision2 = compute_precision(gt_indices, indices);
		}
		else {
			precision2 = computePrecisionDiscrete(gt_dists, dists);
		}
		printf("Precision: %g\n", precision2);
		EXPECT_NEAR(precision, precision2, GTEST_PRECISION);

		// test assignment operator
		Index<Distance > index3(data, index_params);
		index3 = index;

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index3.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index3.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision3;
		if (gt_dists.ptr()==NULL) {
			precision3 = compute_precision(gt_indices, indices);
		}
		else {
			precision3 = computePrecisionDiscrete(gt_dists, dists);
		}
		printf("Precision: %g\n", precision3);
		EXPECT_NEAR(precision, precision3, GTEST_PRECISION);
	}


	template<typename Index>
	void TestCopy2(const flann::Matrix<typename Index::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Index::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Index::DistanceType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			float expected_precision,
			flann::Matrix<size_t>& gt_indices,
			const flann::Matrix<typename Index::DistanceType>& gt_dists = flann::Matrix<typename Index::DistanceType>())
	{
		flann::seed_random(0);
		Index index(data, index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex();
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision;
		if (gt_dists.ptr()==NULL) {
			precision = compute_precision(gt_indices, indices);
		}
		else {
			precision = computePrecisionDiscrete(gt_dists, dists);
		}
		printf("Precision: %g\n", precision);
		EXPECT_GE(precision, expected_precision);

		// test copy constructor
		Index index2(index);

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index2.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index2.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision2;
		if (gt_dists.ptr()==NULL) {
			precision2 = compute_precision(gt_indices, indices);
		}
		else {
			precision2 = computePrecisionDiscrete(gt_dists, dists);
		}
		printf("Precision: %g\n", precision2);
		EXPECT_NEAR(precision, precision2, GTEST_PRECISION);

		// test assignment operator
		Index index3(data, index_params);
		index3 = index;

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index3.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN...");
		index3.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		float precision3;
		if (gt_dists.ptr()==NULL) {
			precision3 = compute_precision(gt_indices, indices);
		}
		else {
			precision3 = computePrecisionDiscrete(gt_dists, dists);
		}
		printf("Precision: %g\n", precision3);
		EXPECT_NEAR(precision, precision3, GTEST_PRECISION);
	}

	template<typename Distance>
	void TestRemove(const flann::Matrix<typename Distance::ElementType>& data,
			const flann::IndexParams& index_params,
			const flann::Matrix<typename Distance::ElementType>& query,
			flann::Matrix<size_t>& indices,
			flann::Matrix<typename Distance::ResultType>& dists,
			size_t knn,
			const flann::SearchParams& search_params,
			bool retestDists = false)
	{
		flann::seed_random(0);
		Index< Distance > index(data, index_params);
		char message[256];
		const char* index_name = index_type_to_name(index.getType());
		sprintf(message, "Building %s index... ", index_name);
		start_timer( message );
		index.buildIndex();
		printf("done (%g seconds)\n", stop_timer());

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		start_timer("Searching KNN before removing points...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		EXPECT_EQ(index.size(), data.rows);

		// remove about 50% of neighbours found
		std::set<size_t> neighbors;
		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				if (rand_double()<0.5) {
					neighbors.insert(indices[i][j]);
				}
			}
		}

		flann::DynamicBitset removed(data.rows);

		for (std::set<size_t>::iterator it = neighbors.begin(); it!=neighbors.end();++it) {
			index.removePoint(*it);
			removed.set(*it);
		}

		// also remove 10% of the initial points
		size_t offset = data.rows/10;
		for (size_t i=0;i<offset;++i) {
			index.removePoint(i);
			removed.set(i);
		}

		size_t new_size = 0;
		for (size_t i=0;i<removed.size();++i) {
			if (!removed.test(i)) ++new_size;
		}

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index.size(), new_size);

		start_timer("Searching KNN after remove points...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				EXPECT_GE(indices[i][j], offset);
				EXPECT_TRUE(neighbors.find(indices[i][j])==neighbors.end());
				EXPECT_EQ(index.getPoint(indices[i][j]), data[indices[i][j]]);
			}
		}

		// save and re-load the index and make sure everything still checks out
		index.save("test_saved_index.idx");
		flann::seed_random(0);
		Index< Distance > index2(data, flann::SavedIndexParams("test_saved_index.idx"));
		index2.buildIndex();

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index2.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index2.size(), new_size);

		flann::Matrix<size_t> indices2(new size_t[query.rows*knn], query.rows, knn);
		flann::Matrix<typename Distance::ResultType> dists2(new typename Distance::ResultType[query.rows*knn], query.rows, knn);
		start_timer("Searching KNN after saving and reloading index...");
		index2.knnSearch(query, indices2, dists2, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				if (retestDists) {
					EXPECT_NEAR(dists[i][j], dists2[i][j], 6);
				} else {
					EXPECT_EQ(indices[i][j], indices2[i][j]);
				}
			}
		}
		delete[] indices2.ptr();
		delete[] dists2.ptr();


		// rebuild index
		index.buildIndex();

#ifdef FLANN_USE_OPENCL
		start_timer("Set up OpenCL KNN...");
    	index.buildCLKnnSearch(knn, search_params);
		printf("done (%g seconds)\n", stop_timer());
#endif

		EXPECT_EQ(index.size(), new_size);

		start_timer("Searching KNN after remove points and rebuild index...");
		index.knnSearch(query, indices, dists, knn, search_params );
		printf("done (%g seconds)\n", stop_timer());

		for (size_t i=0;i<indices.rows;++i) {
			for (size_t j=0;j<indices.cols;++j) {
				EXPECT_GE(indices[i][j], offset);
				EXPECT_TRUE(neighbors.find(indices[i][j])==neighbors.end());
				EXPECT_EQ(index.getPoint(indices[i][j]), data[indices[i][j]]);
			}
		}
	}

};


template<typename ElementType, typename DistanceType>
class DatasetTestFixture : public FLANNTestFixture
{
protected:

	std::string filename_;

	flann::Matrix<ElementType> data;
	flann::Matrix<ElementType> query;
	flann::Matrix<size_t> gt_indices;
	flann::Matrix<DistanceType> dists;
	flann::Matrix<size_t> indices;

	int knn;

	DatasetTestFixture(const std::string& filename) : filename_(filename), knn(5)
	{
	}


	void SetUp()
	{
		knn = 5;
		printf("Reading test data...");
		fflush(stdout);
		flann::load_from_file(data, filename_.c_str(), "dataset");
		flann::load_from_file(query, filename_.c_str(), "query");
		flann::load_from_file(gt_indices, filename_.c_str(), "match");

		dists = flann::Matrix<DistanceType>(new DistanceType[query.rows*knn], query.rows, knn);
		indices = flann::Matrix<size_t>(new size_t[query.rows*knn], query.rows, knn);

		printf("done\n");
	}

	void TearDown()
	{
		delete[] data.ptr();
		delete[] query.ptr();
		delete[] gt_indices.ptr();
		delete[] dists.ptr();
		delete[] indices.ptr();

	}
};

#endif /* FLANN_TESTS_H_ */

