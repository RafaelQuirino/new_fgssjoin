#ifndef _BLOCK_CUH_
#define _BLOCK_CUH_

#include "string/data.cuh"
#include "index.cuh"
#include "util.cuh"



/*
 *  DOCUMENTATION
 */
__host__
void process_blocks (sets_t* sets, Index index, float threshold); //, int size);



#endif /* _BLOCK_CUH_ */