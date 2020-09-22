#ifndef _BLOCK_CUH_
#define _BLOCK_CUH_

#include "util.cuh"
#include "data.cuh"
#include "index.cuh"



/*
 *  DOCUMENTATION
 */
__host__
void process_blocks (sets_t* sets, Index index, float threshold, char verbose);



#endif /* _BLOCK_CUH_ */