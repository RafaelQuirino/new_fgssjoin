#ifndef _UTIL_CUH_
#define _UTIL_CUH_

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cstdio>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

extern int WARP_SIZE;

void get_grid_config(dim3 &grid, dim3 &threads);

void __gpuAssert(cudaError_t stat, int line, std::string file);

#define gpuAssert(value)  __gpuAssert((value),(__LINE__),(__FILE__))

#define gpu(value)  __gpuAssert((value),(__LINE__),(__FILE__))

#endif /* _UTIL_CUH_ */
