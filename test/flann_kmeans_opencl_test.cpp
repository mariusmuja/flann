#define FLANN_USE_OPENCL
#include <gtest/gtest.h>
#include <time.h>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "flann_tests.h"

using namespace flann;

/**
 * Test fixture for SIFT 10K dataset
 */
class KMeansOpenCL_SIFT10K : public DatasetTestFixture<float, float> {
protected:
	KMeansOpenCL_SIFT10K() : DatasetTestFixture("sift10K.h5") {}
};




TEST_F(KMeansOpenCL_SIFT10K, TestSearch)
{
	TestSearch<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT10K, TestSearch2)
{
	TestSearch2<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT10K, TestAddIncremental)
{
	TestAddIncremental<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(110), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT10K, TestAddIncremental2)
{
	TestAddIncremental2<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(110), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT10K, TestRemove)
{
	TestRemove<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128));
}



TEST_F(KMeansOpenCL_SIFT10K, TestSave)
{
	TestSave<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT10K, TestCopy)
{
	TestCopy<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT10K, TestCopy2)
{
	TestCopy2<KMeansOpenCLIndex<flann::L2<float> > >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}

/**
 * Test fixture for SIFT 100K dataset
 */
class KMeansOpenCL_SIFT100K : public DatasetTestFixture<float, float> {
protected:
	KMeansOpenCL_SIFT100K() : DatasetTestFixture("sift100K.h5") {}
};


TEST_F(KMeansOpenCL_SIFT100K, TestSearch)
{
	TestSearch<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(96), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT100K, TestAddIncremental)
{
	TestAddIncremental<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT100K, TestAddIncremental2)
{
	TestAddIncremental2<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT100K, TestRemove)
{
	TestRemove<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4), query, indices,
			dists, knn, flann::SearchParams(128) );
}

TEST_F(KMeansOpenCL_SIFT100K, TestSave)
{
	TestSave<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(96), 0.75, gt_indices);
}



/**
 * Test fixture for SIFT 10K dataset with byte feature elements
 */
class KMeansOpenCL_SIFT10K_byte :  public DatasetTestFixture<unsigned char, float> {
protected:
	KMeansOpenCL_SIFT10K_byte() : DatasetTestFixture("sift10K_byte.h5") {}
};

TEST_F(KMeansOpenCL_SIFT10K_byte, TestSearch)
{
	TestSearch<flann::L2<unsigned char> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}



class KMeansOpenCL_SIFT100K_byte : public DatasetTestFixture<unsigned char, float> {
protected:
	KMeansOpenCL_SIFT100K_byte() : DatasetTestFixture("sift100K_byte.h5") {}
};

TEST_F(KMeansOpenCL_SIFT100K_byte, TestSearch)
{
	TestSearch<flann::L2<unsigned char> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(96), 0.75, gt_indices);
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
