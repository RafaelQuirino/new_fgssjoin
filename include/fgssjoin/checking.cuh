#ifndef _CHECKING_CUH
#define _CHECKING_CUH

#include "util.hpp"
#include "util.cuh"
#include "data.cuh"
#include "similarity.cuh"



/*
 *  DOCUMENTATION
 */
__host__
void checking_block (
	short* d_partial_scores,
	unsigned int* d_buckets, sets_t* sets, float threshold, unsigned int csize, 
	unsigned int q_offset, unsigned int i_offset, unsigned int block_size,
	unsigned int** similar_pairs, unsigned short** scores, unsigned int* size,
	int* d_nres, char verbose
);



/*
 *  DOCUMENTATION
 */
__global__
void checking_kernel_block (
	short* partial_scores,
	unsigned int* buckets, unsigned short* scores,
	unsigned int* pos, unsigned int* len, unsigned int* tokens, float threshold, 
	unsigned int q_offset, unsigned int i_offset, unsigned int block_size, unsigned int csize
);



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_1 (
    unsigned int *dst, 
    unsigned int *buckets, 
    const unsigned short *scores, 
    int *nres, 
    int n
);



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_2 (
    unsigned short *dst, 
    const unsigned short *scores, 
    int *nres, 
    int n
);



#endif /* _CHECKING_CUH */
