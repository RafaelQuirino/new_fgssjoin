#include <math.h>

#include "../../include/fgssjoin/block.cuh"
#include "../../include/fgssjoin/similarity.cuh"
#include "util.hpp"
#include "filtering.cuh"
#include "checking.cuh"



void print_separator () 
{
	fprintf(stderr, "--------------------------------------------------------------------------------\n");
}



int get_block_size () 
{
	size_t uCurAvailMemoryInBytes;
	size_t uTotalMemoryInBytes;
	// int nNoOfGPUs;

	CUresult result;
	CUdevice device;
	CUcontext context;

	cuInit(0);                      // Initialize CUDA
	// cuDeviceGetCount( &nNoOfGPUs ); // Get number of devices supporting CUDA
	// for( int nID = 0; nID < nNoOfGPUs; nID++ )
	// {
	    // cuDeviceGet( &device, nID );        // Get handle for device
	    cuDeviceGet( &device, 0 );          // Get handle for device
	    cuCtxCreate( &context, 0, device ); // Create context
	    result = cuMemGetInfo( &uCurAvailMemoryInBytes, &uTotalMemoryInBytes );
	    if( result == CUDA_SUCCESS )
	    {
	        fprintf(stderr, "Device: %d\nTotal Memory: %zu MB, Free Memory: %zu MB\n",
	                0, //nID,
	                uTotalMemoryInBytes / ( 1024 * 1024 ),
	                uCurAvailMemoryInBytes / ( 1024 * 1024 ));
	    }
	    // cuCtxDetach( context ); // Destroy context
	// }

	float limit = 0.95; // Percentage of available memory to take
	int block_size = floor(floor(sqrt((float)uCurAvailMemoryInBytes/6)) * limit);

	return block_size;
}



/*
 *  DOCUMENTATION
 */
__host__
void process_blocks (sets_t* sets, Index inv_index, float threshold)//, int block_size)
{
	fprintf(stderr, "\nBlock processing.\n");

	unsigned long t0, t1;
	unsigned long sim_pairs_count = 0;

	// Getting the block size -----------------------------
	// float mem = (float) ((candsize*sizeof(short)) + (candsize*sizeof(unsigned int))) / (1024*1024*1024);
	// fprintf(stderr, "Memory requirements: %.2g GB\n", mem);
	int block_size = get_block_size();
	fprintf(stderr, "block_size: %d\n", block_size);
	// exit(0);
	//-----------------------------------------------------

	// Blocks logic ---------------------------------------
	int div        = sets->num_sets / block_size;
	int mod        = sets->num_sets % block_size;
	int n_blocks   = mod == 0 ? div : div + 1;
	int last_block = mod == 0 ? block_size : mod;
	//-----------------------------------------------------

	// CREATING SCORES AND BUCKETS ARRAYS ------------------------------------------
	unsigned int candsize = block_size * block_size;

	int* d_compidx;                // index for filter_k compaction
	short* d_partial_scores;       // partial scores in filtering step
	unsigned int* d_comp_buckets;  // filtered candidates (buckets)

	gpu(cudaMalloc(&d_compidx, sizeof(int)));
	gpu(cudaMalloc(&d_partial_scores, candsize * sizeof(short)));
	gpu(cudaMalloc(&d_comp_buckets, candsize * sizeof(unsigned int)));
    //------------------------------------------------------------------------------

	// print_separator();
	for (int i = 0; i < n_blocks; i++)
	{
		int i_offset = i * block_size;
		int bsize = i == n_blocks - 1 ? last_block : block_size;

		// fprintf(stderr, "\nINDEX BLOCK (%d)... (size: %d, i_offset: %d)\n", i, bsize, i_offset);

		for (int j = 0; j <= i; j++)
		{
			int q_offset = j * block_size;
			int bsize = j == n_blocks - 1 ? last_block : block_size;

			// fprintf(stderr, "\n\tQUERYING WITH BLOCK (%d)... (size: %d, q_offset: %d)\n\n", j, bsize, q_offset);

			//######################################################################
			// If the last set from the query block is bigger than the
			// maxsize of the first set in the index, we can skip this block
			//######################################################################
			int q_size = (int) sets->len[(q_offset+bsize) - 1];
			int s_maxsize = floor(jac_max_size(sets->len[i_offset], threshold));
			if (q_size > s_maxsize) {
				// printf("\t\t => The last set from this block is bigger than\n");
				// printf("\t\t    the maxsize of the first set from the index !\n");
				// printf("\t\t    Skipping...\n");
				continue;
			}
			//######################################################################

			// FILTERING -----------------------------------------------------------
			// fprintf(stderr, "\t\t => FILTERING... \n");
			// t0 = ut_get_time_in_microseconds();

			unsigned int *d_candidates, csize;

			filtering_block (
				sets, inv_index,
				d_partial_scores, d_comp_buckets, d_compidx,
				candsize, threshold, q_offset, i_offset, block_size, bsize,
				&d_candidates,  &csize
			);

			// t1 = ut_get_time_in_microseconds();
			// fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
			// fprintf(stderr, "\t\t    no of candidates: %d\n", csize);
			//----------------------------------------------------------------------

			// CHECKING ------------------------------------------------------------
			// fprintf(stderr,"\n\t\t => CHECKING... ");
			// t0 = ut_get_time_in_microseconds();

			unsigned int *similar_pairs;
			unsigned short *scores;

			checking_block (
				d_partial_scores,
				d_candidates, sets, threshold, csize,
				q_offset, i_offset, block_size,
				&similar_pairs, &scores
			);

			// t1 = ut_get_time_in_microseconds();
			// fprintf(stderr, " %g ms\n\n", ut_interval_in_miliseconds(t0,t1));
			//----------------------------------------------------------------------

			for (unsigned k = 0; k < csize; k++)
			{
				unsigned int   bucket = similar_pairs[k];
				unsigned int   query  = (bucket / block_size) + q_offset;
				unsigned int   source = (bucket % block_size) + i_offset;
				unsigned short score  = scores[k];

				if (score > 0)
				{
					float similarity = jac_similarity(sets->len[query], sets->len[source], score);
					if (similarity >= threshold) 
					{
						fprintf(stdout, "%d %d %f\n", sets->id[query], sets->id[source], similarity);
						sim_pairs_count++;
					}
				}
			}

			free(similar_pairs);
			free(scores);
		}

		// print_separator();
	}

	fprintf(stderr, "Number of similar pairs: %lu\n", sim_pairs_count);
}
