#ifndef _COMPACTION_CUH
#define _COMPACTION_CUH

#include "util.hpp"
#include "util.cuh"
#include "scan.cuh"



/*
 *  DOCUMENTATION
 */
__global__ 
void filter_k (unsigned int *dst, const short *src, int *nres, int n);



/*
 *  DOCUMENTATION
 */
__global__
void filter_k_int (unsigned int *dst, const int *src, int *nres, int n);



// /*
//  *  DOCUMENTATION
//  */
// __global__ 
// void filter_k_ushort (
//     unsigned int *dst, 
//     unsigned int *src, 
//     const unsigned short *pred, 
//     int *nres, 
//     int n
// )



// /*
//  *  DOCUMENTATION
//  */
// __global__
// void warp_agg_filter_k (unsigned int *dst, const short *src, int* nres, int n);



#endif