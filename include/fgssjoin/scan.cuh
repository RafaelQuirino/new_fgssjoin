#ifndef _SCAN_CUH
#define _SCAN_CUH

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "util.hpp"
#include "util.cuh"

using namespace std;

#define BLOCK_SIZE 1000000000

template <typename T>
__global__
void inc_block_kernel (T* d_scan, unsigned int* d_sum, unsigned int n)
{
    unsigned int sum = d_sum[0];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        d_scan[i] += sum;
    }
}



template <typename T> 
inline void exclusive_scan_impl (T *d_in, T *d_out, unsigned int first, unsigned int end)
{
    thrust::device_ptr<T> thrust_d_in(d_in);
    thrust::device_ptr<T> thrust_d_out(d_out);
    thrust::exclusive_scan(thrust_d_in + first, thrust_d_in + end, thrust_d_out + first);
    gpuAssert(cudaDeviceSynchronize());
}



template <typename T>
inline void exclusive_scan (T *d_in, T *d_out, unsigned int n)
{
    unsigned int b = BLOCK_SIZE;
    unsigned int n_blocks = n / b;
    unsigned int last_block = n % b;

    if (last_block == 0)
        last_block = b;
    else
        n_blocks += 1;

    if (n <= b) 
    {
        exclusive_scan_impl <T> (d_in, d_out, 0, n);
    } 
    else 
    {
        unsigned int pos[n_blocks];
        unsigned int sum[n_blocks];
        pos[0] = 0;
        
        for (unsigned i = 1; i < n_blocks; i++)
            pos[i] = pos[i-1] + b;



        for (unsigned i = 0; i < n_blocks-1; i++)
        {
            unsigned int start = pos[i];
            exclusive_scan_impl <T> (d_in, d_out, start, start + b);

            T last_in;
            T last_out;
            gpuAssert(cudaMemcpy(&last_in, d_in + start + (b-1), sizeof(T), cudaMemcpyDeviceToHost));
            gpuAssert(cudaMemcpy(&last_out, d_out + start+ (b-1), sizeof(T), cudaMemcpyDeviceToHost));
            sum[i] = last_out + last_in;
        }

        unsigned int start = pos[n_blocks-1];
        exclusive_scan_impl <T> (d_in, d_out, start, start + last_block);

        T last_in;
        T last_out;
        gpuAssert(cudaMemcpy(&last_in, d_in + start + (last_block-1), sizeof(T), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&last_out, d_out + start+ (last_block-1), sizeof(T), cudaMemcpyDeviceToHost));
        sum[n_blocks-1] = last_out + last_in;
        


        unsigned int* d_sum;
        gpuAssert(cudaMalloc(&d_sum, n_blocks * sizeof(unsigned int)));
        gpuAssert(cudaMemcpy(d_sum, sum, n_blocks * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        exclusive_scan_impl <unsigned int> (d_sum, d_sum, 0, n_blocks);
        
        dim3 grid, block;
        get_grid_config(grid,block);
        for (unsigned i = 0; i < n_blocks-1; i++)
        {
            inc_block_kernel <<<grid,block>>> (d_out + pos[i], d_sum + i, b);
        }

        inc_block_kernel <<<grid,block>>> (d_out + pos[n_blocks-1], d_sum + (n_blocks-1), last_block);
        gpuAssert(cudaDeviceSynchronize());
    }
}

#endif
