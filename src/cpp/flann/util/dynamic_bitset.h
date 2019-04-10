/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
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

/***********************************************************************
 * Author: Vincent Rabaud
 *************************************************************************/

#ifndef FLANN_DYNAMIC_BITSET_H_
#define FLANN_DYNAMIC_BITSET_H_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <limits.h>

namespace flann {

/** Class re-implementing the boost version of it
 * This helps not depending on boost, it also does not do the bound checks
 * and has a way to reset a block for speed
 */
class DynamicBitset
{
public:
    /** @param default constructor
     */
    DynamicBitset() : size_(0)
    {
    }

    /** @param only constructor we use in our code
     * @param the size of the bitset (in bits)
     */
    DynamicBitset(size_t size)
    {
        resize(size);
        reset();
    }

    /** Sets all the bits to 0
     */
    void clear()
    {
        std::fill(bitset_.begin(), bitset_.end(), 0);
    }

    /** @brief checks if the bitset is empty
     * @return true if the bitset is empty
     */
    bool empty() const
    {
        return bitset_.empty();
    }

    /** @param set all the bits to 0
     */
    void reset()
    {
        std::fill(bitset_.begin(), bitset_.end(), 0);
    }

    /** @brief set one bit to 0
     * @param
     */
    void reset(size_t index)
    {
        bitset_[index / cell_bit_size_] &= ~(size_t(1) << (index % cell_bit_size_));
    }

    /** @brief sets a specific bit to 0, and more bits too
     * This function is useful when resetting a given set of bits so that the
     * whole bitset ends up being 0: if that's the case, we don't care about setting
     * other bits to 0
     * @param
     */
    void reset_block(size_t index)
    {
        bitset_[index / cell_bit_size_] = 0;
    }

    /** @param resize the bitset so that it contains at least size bits
     * @param size
     */
    void resize(size_t size)
    {
        size_ = size;
        bitset_.resize(size / cell_bit_size_ + 1);
    }

    /** @param set a bit to true
     * @param index the index of the bit to set to 1
     */
    void set(size_t index)
    {
        bitset_[index / cell_bit_size_] |= size_t(1) << (index % cell_bit_size_);
    }

    /** @param gives the number of contained bits
     */
    size_t size() const
    {
        return size_;
    }

    /** @param check if a bit is set
     * @param index the index of the bit to check
     * @return true if the bit is set
     */
    bool test(size_t index) const
    {
        return (bitset_[index / cell_bit_size_] & (size_t(1) << (index % cell_bit_size_))) != 0;
    }

private:
    template <typename Archive>
    void serialize(Archive& ar)
    {
    	ar & size_;
    	ar & bitset_;
    }
    friend struct serialization::access;

private:
    std::vector<size_t> bitset_;
    size_t size_;
    static const unsigned int cell_bit_size_ = CHAR_BIT * sizeof(size_t);
};


class SparseBitset
{
private:
	static const unsigned int MAX_SET_COUNT = 11000;
public:
    /** @param default constructor
     */
	SparseBitset() : size_(0)
    {
        setPointers_ = (size_t**)malloc(MAX_SET_COUNT*sizeof(size_t*));
        setCount_ = 0;
    }

    /** @param only constructor we use in our code
     * @param the size of the bitset (in bits)
     */
	SparseBitset(size_t size)
    {
        setPointers_ = (size_t**)malloc(MAX_SET_COUNT*sizeof(size_t*));
        resize(size);
    }

	~SparseBitset()
    {
		free(setPointers_);
    }

	/** Sets all the bits to 0
     */
    void clear()
    {
    	reset();
    }

    /** @brief checks if the bitset is empty
     * @return true if the bitset is empty
     */
    bool empty() const
    {
        return bitset_.empty();
    }

    /** @param set all the bits to 0
     */
    void reset()
    {
    	 for (unsigned int i = 0; i < setCount_; ++i) {
    		 *setPointers_[i] = 0;
    	 }
        setCount_ = 0;
    }

    /** @brief set one bit to 0
     * @param
     */
    void reset(size_t index)
    {
        bitset_[index / cell_bit_size_] &= ~(size_t(1) << (index % cell_bit_size_));
    }

    /** @brief sets a specific bit to 0, and more bits too
     * This function is useful when resetting a given set of bits so that the
     * whole bitset ends up being 0: if that's the case, we don't care about setting
     * other bits to 0
     * @param
     */
    void reset_block(size_t index)
    {
        bitset_[index / cell_bit_size_] = 0;
    }

    /** @param resize the bitset so that it contains at least size bits
     * @param size
     */
    void resize(size_t size)
    {
        size_ = size;
        bitset_.resize(size / cell_bit_size_ + 1);
        std::fill(bitset_.begin(), bitset_.end(), 0);
        setCount_ = 0;
    }

    /** @param set a bit to true
     * @param index the index of the bit to set to 1
     */
    void set(size_t index)
    {
    	assert(MAX_SET_COUNT > setCount_);

        int indexVec = index / cell_bit_size_;
        setPointers_[setCount_++] = &bitset_[indexVec];
        bitset_[indexVec] |= size_t(1) << (index % cell_bit_size_);
    }

    /** @param gives the number of contained bits
     */
    size_t size() const
    {
        return size_;
    }

    /** @param check if a bit is set
     * @param index the index of the bit to check
     * @return true if the bit is set
     */
    bool test(size_t index) const
    {
        return (bitset_[index / cell_bit_size_] & (size_t(1) << (index % cell_bit_size_))) != 0;
    }

private:
    std::vector<size_t> bitset_;
    size_t size_;
    static const unsigned int cell_bit_size_ = CHAR_BIT * sizeof(size_t);
    size_t** setPointers_;
    unsigned int setCount_;
};


/**
 * Pool of SparseBitset object class
 *
 * This class pools SparseBitset objects by their size.
 */
class BitsetPool
{
private:
	std::map<size_t, std::vector<SparseBitset*> > reservedBitset;
#ifdef _OPENMP
	omp_lock_t lock;
	BitsetPool() {
		omp_init_lock(&lock);
	}
	~BitsetPool() {
		omp_destroy_lock(&lock);
	}
#endif
public:
    /**
     * Returns: pool instance
     */
	static BitsetPool &getBitsetPool() {
		static BitsetPool pool;
		return pool;
	}

    /**
     * Returns: SparseBitset that has specified reserve size
     */
	SparseBitset* getBitset(size_t size) {
		SparseBitset *ret = NULL;
#ifdef _OPENMP
		omp_set_lock(&lock);
#endif
		if (reservedBitset[size].size() == 0) {
			ret = new SparseBitset(size);
		} else {
			ret = reservedBitset[size].back();
			reservedBitset[size].pop_back();
		}
#ifdef _OPENMP
		omp_unset_lock(&lock);
#endif

		return ret;
	}

    /**
     * After using SparseBitset, the SparseBitset have to be put back by this method.
     */
	void putBackBitset(SparseBitset *bitset) {
		size_t size = bitset->size();
		bitset->clear();
#ifdef _OPENMP
		omp_set_lock(&lock);
		reservedBitset[size].push_back(bitset);
		omp_unset_lock(&lock);
#else
		reservedBitset[size].push_back(bitset);
#endif
	}

    /**
     * Delete pooled bitsets these have specified reserve size.
     */
	void deletePool(size_t size) {
#ifdef _OPENMP
		omp_set_lock(&lock);
#endif

		while (reservedBitset[size].size() > 0) {
			SparseBitset* bitset = reservedBitset[size].back();
			reservedBitset[size].pop_back();
			delete bitset;
		}

#ifdef _OPENMP
		omp_unset_lock(&lock);
#endif
	}

};

} // namespace flann

#endif // FLANN_DYNAMIC_BITSET_H_
