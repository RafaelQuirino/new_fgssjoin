#include "compaction.cuh"



/*
 *  DOCUMENTATION
 */
__global__ 
void filter_k (unsigned int *dst, const short *src, int *nres, int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        if (src[i] > 0)
            dst[atomicAdd(nres, 1)] = i;
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void filter_k_int (unsigned int *dst, const int *src, int *nres, int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        if (src[i] > 0)
            dst[atomicAdd(nres, 1)] = i;
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void filter_k_ushort (
    unsigned int *dst, 
    unsigned int *src, 
    const unsigned short *pred, 
    int *nres, 
    int n
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        if (pred[i] > 0)
            dst[atomicAdd(nres, 1)] = src[i];
    }
}



/*
 *  DOCUMENTATION
 */
// __global__ 
// void filter_k_2 (unsigned int *dst, const short *src, int *nres, int n) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
//     {
//         if (src[i] > 0)
//             dst[atomicAdd(nres, 1)] = i;
//     }
// }



// ONLY FOR CAPABILITY 3.X
//=========================
/*
#define WARP_SZ 32
__device__
inline int lane_id (void) { return threadIdx.x % WARP_SZ; }

// warp-aggregated atomic increment
__device__
int atomicAggInc (int *ctr)
{
    int mask = __ballot(1);
    // select the leader
    int leader = __ffs(mask) - 1;
    // leader does the update
    int res;
    if (lane_id() == leader)
        res = atomicAdd(ctr, __popc(mask));
    // broadcast result
    res = __shfl(res, leader);
    // each thread computes its own value
    return res + __popc(mask & ((1 << lane_id()) - 1));
} // atomicAggInc

__global__
void warp_agg_filter_k (unsigned int *dst, const short *src, int* nres, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        if (src[i] > 0)
            dst[atomicAggInc(nres)] = i;
    }
}
*/


