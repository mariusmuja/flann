/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2013  Shiina, Hideaki (YNDRD Co.Ltd. shiina@yndrd.com). All rights reserved.
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

#ifndef FLANN_FREESIZEHEAP_H_
#define FLANN_FREESIZEHEAP_H_

#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace flann
{

/**
 * Priority Queue Implementation
 *
 * The priority queue is implemented with a heap.  A heap is a complete
 * (full) binary tree in which each parent is less than both of its
 * children, but the order of the children is unspecified.
 */
template <typename T>
class FreeSizeHeap
{

    /**
     * Storage array for the heap.
     * Type T must be comparable.
     */
    std::vector<T> heap;

    /**
     * Number of element in the heap
     */
    size_t count;



public:
    /**
     * Constructor.
     *
     * Params:
     *     size = heap size
     */

    FreeSizeHeap()
    {
        count = 0;
    }

    /**
     *
     * Returns: heap size
     */
    size_t size()
    {
        return count;
    }

    /**
     * Tests if the heap is empty
     *
     * Returns: true is heap empty, false otherwise
     */
    bool empty()
    {
        return count==0;
    }

    /**
     * Clears the heap.
     */
    void clear()
    {
        heap.clear();
        count = 0;
    }

    struct CompareT : public std::binary_function<T,T,bool>
    {
        bool operator()(const T& t_1, const T& t_2) const
        {
            return t_2 < t_1;
        }
    };

    /**
     * Insert a new element in the heap.
     *
     * We select the next empty leaf node, and then keep moving any larger
     * parents down until the right location is found to store this element.
     *
     * Params:
     *     value = the new element to be inserted in the heap
     */
    void insert(const T& value)
    {
        heap.push_back(value);
        static CompareT compareT;
        std::push_heap(heap.begin(), heap.end(), compareT);
        ++count;
    }



    /**
     * Returns the node of minimum value from the heap (top of the heap).
     *
     * Params:
     *     value = out parameter used to return the min element
     * Returns: false if heap empty
     */
    bool popMin(T& value)
    {
        if (count == 0) {
            return false;
        }

        value = heap[0];
        static CompareT compareT;
        std::pop_heap(heap.begin(), heap.end(), compareT);
        heap.pop_back();
        --count;

        return true;  /* Return old last node. */
    }
};


/**
 * Pool of heap object class
 *
 * This class pools heap objects
 */
template <typename T>
class HeapPool
{
private:
	std::vector<FreeSizeHeap<T>*> pool;
#ifdef _OPENMP
	omp_lock_t lock;
	HeapPool() {
		omp_init_lock(&lock);
	}
	~HeapPool() {
		omp_destroy_lock(&lock);
	}
#endif
public:
    /**
     * Returns: pool instance
     */
	static HeapPool<T> &getHeapPool() {
		static HeapPool<T> pool;
		return pool;
	}

    /**
     * Returns: heap that has specified reserve size
     */
	FreeSizeHeap<T>* getHeap() {
		FreeSizeHeap<T> *ret = NULL;
#ifdef _OPENMP
		omp_set_lock(&lock);
#endif
		if (pool.size() == 0) {
			ret = new FreeSizeHeap<T>();
		} else {
			ret = pool.back();
			pool.pop_back();
		}
#ifdef _OPENMP
		omp_unset_lock(&lock);
#endif

		return ret;
	}

    /**
     * After using heap, the heap have to be put back by this method.
     */
	void putBackHeap(FreeSizeHeap<T> *heap) {
		heap->clear();
#ifdef _OPENMP
		omp_set_lock(&lock);
		pool.push_back(heap);
		omp_unset_lock(&lock);
#else
		pool.push_back(heap);
#endif
	}

    /**
     * Delete pooled heaps these have specified reserve size.
     */
	void deletePool() {
#ifdef _OPENMP
		omp_set_lock(&lock);
#endif

		while (pool.size() > 0) {
			FreeSizeHeap<T>* heap = pool.back();
			pool.pop_back();
			delete heap;
		}

#ifdef _OPENMP
		omp_unset_lock(&lock);
#endif
	}

};

}

#endif //FLANN_HEAP_H_
