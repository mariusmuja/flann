#define FLANN_USE_OPENCL
#include <gtest/gtest.h>
#include <time.h>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "flann_tests.h"

using namespace flann;

class HierarchicalOpenCLIndex_Brief100K : public FLANNTestFixture
{
protected:
	typedef flann::Hamming<unsigned char> Distance;
	typedef Distance::ElementType ElementType;
	typedef Distance::ResultType DistanceType;
	flann::Matrix<unsigned char> data;
	flann::Matrix<unsigned char> query;
	flann::Matrix<size_t> gt_indices;
	flann::Matrix<DistanceType> dists;
	flann::Matrix<DistanceType> gt_dists;
	flann::Matrix<size_t> indices;
	unsigned int k_nn_;

	void SetUp()
	{
		k_nn_ = 3;
		printf("Reading test data...");
		fflush(stdout);
		flann::load_from_file(data, "brief100K.h5", "dataset");
		flann::load_from_file(query, "brief100K.h5", "query");
		printf("done\n");

		flann::Index<Distance> index(data, flann::LinearIndexParams());
		index.buildIndex();

		start_timer("Searching KNN for ground truth...");
		gt_indices = flann::Matrix<size_t>(new size_t[query.rows * k_nn_], query.rows, k_nn_);
		gt_dists = flann::Matrix<DistanceType>(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
		index.knnSearch(query, gt_indices, gt_dists, k_nn_, flann::SearchParams(-1));
		printf("done (%g seconds)\n", stop_timer());

		dists = flann::Matrix<DistanceType>(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
		indices = flann::Matrix<size_t>(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	}

	void TearDown()
	{
		delete[] data.ptr();
		delete[] query.ptr();
		delete[] dists.ptr();
		delete[] indices.ptr();
		delete[] gt_indices.ptr();
		delete[] gt_dists.ptr();
	}
};


TEST_F(HierarchicalOpenCLIndex_Brief100K, TestSearch)
{
	TestSearch<Distance>(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.9, gt_indices, gt_dists);
}

TEST_F(HierarchicalOpenCLIndex_Brief100K, TestSearch2)
{
	TestSearch2<Distance>(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.9, gt_indices, gt_dists);
}


TEST_F(HierarchicalOpenCLIndex_Brief100K, TestAddIncremental)
{
	TestAddIncremental<Distance>(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists, true);
}

TEST_F(HierarchicalOpenCLIndex_Brief100K, TestAddIncremental2)
{
	TestAddIncremental2<Distance>(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists, true);
}

TEST_F(HierarchicalOpenCLIndex_Brief100K, TestRemove)
{
	TestRemove<Distance>(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), true);
}

TEST_F(HierarchicalOpenCLIndex_Brief100K, TestSave)
{
	TestSave<Distance>(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists, true);
}


TEST_F(HierarchicalOpenCLIndex_Brief100K, TestCopy)
{
	TestCopy<Distance>(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists);
}

TEST_F(HierarchicalOpenCLIndex_Brief100K, TestCopy2)
{
	TestCopy2<flann::HierarchicalClusteringOpenCLIndex<Distance> >(data, flann::HierarchicalClusteringOpenCLIndexParams(),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists);
}





int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
