#include "../include/util.cuh"

int WARP_SIZE = 32;

void get_grid_config (dim3 &grid, dim3 &threads)
{
    //Get the device properties
    static bool flag = 0;
    static dim3 lgrid, lthreads;
    if (!flag) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);

		//Adjust the grid dimensions based on the device properties
		int num_blocks = 1024 * 2 * devProp.multiProcessorCount;
		lgrid = dim3(num_blocks);
		lthreads = dim3(devProp.maxThreadsPerBlock / 1);
		flag = 1;
    }
    grid = lgrid;
    threads = lthreads;
}

void get_grid_config_block (dim3 &grid, dim3 &threads, int n)
{
    //Get the device properties
    static bool flag = 0;
    static dim3 lgrid, lthreads;
    if (!flag) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);

        //Adjust the grid dimensions based on the device properties
        int num_blocks = n; //1024 * 2 * devProp.multiProcessorCount;
        lgrid = dim3(num_blocks);
        lthreads = dim3(devProp.maxThreadsPerBlock / 4);
        flag = 1;
    }
    grid = lgrid;
    threads = lthreads;
}

void __gpuAssert (cudaError_t stat, int line, std::string file) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "Error, %s at line %d in file %s\n",
            cudaGetErrorString(stat), line, file.data());
        exit(1);
    }
}



// https://forums.developer.nvidia.com/t/how-to-use-atomiccas-to-implement-atomicadd-short-trouble-adapting-programming-guide-example/22712
__device__ short atomicAddShort(short* address, short val)

{

    unsigned int *base_address = (unsigned int *)((size_t)address & ~2);

    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;

unsigned int long_old = atomicAdd(base_address, long_val);

    if((size_t)address & 2) {

        return (short)(long_old >> 16);

    } else {

        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

        if (overflow)

            atomicSub(base_address, overflow);

        return (short)(long_old & 0xffff);

    }

}



/*
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
//*/
