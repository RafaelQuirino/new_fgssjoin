#ifndef _FILTERING_CUH_
#define _FILTERING_CUH_

#include "util.hpp"
#include "util.cuh"
#include "index.cuh"
#include "compaction.cuh"
#include "similarity.cuh"



/*
 *  DOCUMENTATION
 */
__host__
void filtering_block (
	sets_t* sets, Index inv_index,
	short* d_partial_scores, unsigned int* d_comp_buckets, int* d_nres, 
	unsigned int bsize, float threshold, unsigned int q_offset, unsigned int s_offset, 
	unsigned int block_size, unsigned int n_queries,
	unsigned int** candidates, unsigned int* candidates_size,
	char verbose
);



/*
 *  DOCUMENTATION
 */
__global__
void filtering_kernel_block (
	Entry* inv_index, int* count, int* index,
	unsigned int* pos, unsigned int* len, unsigned int* tokens, short* scores, 
	float threshold, unsigned int q_offset, unsigned i_offset, 
	unsigned int block_size, unsigned int n
);



#endif /* _FILTERING_CUH_ */