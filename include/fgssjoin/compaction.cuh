#ifndef _COMPACTION_CUH
#define _COMPACTION_CUH

#include "util.hpp"
#include "util.cuh"
#include "scan.cuh"



__global__ 
void filter_k (unsigned int *dst, const short *src, int *nres, int n);

__global__
void filter_k_int (unsigned int *dst, const int *src, int *nres, int n);

__global__
void warp_agg_filter_k (unsigned int *dst, const short *src, int* nres, int n);



#endif