#include "filtering.cuh"



/*
 *  DOCUMENTATION
 */
__host__
void filtering_block (
	sets_t* sets, Index inv_index,
	short* d_partial_scores, unsigned int* d_comp_buckets, int* d_nres, 
	unsigned int bsize, float threshold, unsigned int q_offset, unsigned int i_offset, 
	unsigned int block_size, unsigned int n_queries,
	unsigned int** d_candidates, unsigned int* candidates_size,
	char verbose
)
{
	unsigned long t0, t1;

	dim3 grid, block;
	get_grid_config(grid, block);

	gpu(cudaMemset(d_partial_scores, 0, bsize * sizeof(short)));


	// CALLING THE FILTERING KERNEL ---------------------------------------------------------------
    if (verbose) {
    	fprintf(stderr, "\t\t\t* Calling filtering_kernel... ");
		t0 = ut_get_time_in_microseconds();
	}

	filtering_kernel_block <<<grid,block>>> (
		inv_index.d_lists, inv_index.d_count, inv_index.d_index,
		sets->d_pos, sets->d_len, sets->d_tokens, d_partial_scores, 
		threshold, q_offset, i_offset, block_size, n_queries
	);
	gpu(cudaDeviceSynchronize());

	if (verbose) {
		t1 = ut_get_time_in_microseconds();
		fprintf(stderr, "%g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }
    //---------------------------------------------------------------------------------------------


    // COMPACTING FILTERED CANDIDATES -------------------------------------------------------------
	if (verbose) {
        fprintf(stderr, "\t\t\t* Compacting filtered buckets... ");
        t0 = ut_get_time_in_microseconds();
    }

	gpu(cudaMemset(d_nres, 0, sizeof(int)));
	filter_k <<<grid,block>>> (d_comp_buckets, d_partial_scores, d_nres, bsize);
	// warp_agg_filter_k <<<grid,block>>> (d_comp_buckets, d_partial_scores, d_nres, bsize);
	gpu(cudaDeviceSynchronize());

	int nres;
	gpu(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));
	unsigned int comp_buckets_size = (unsigned int) nres;

    if (verbose) {
        t1 = ut_get_time_in_microseconds();
        fprintf(stderr, "%g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }
    //---------------------------------------------------------------------------------------------


	*d_candidates = d_comp_buckets;
    *candidates_size = comp_buckets_size;

    printf("candidates_size: %u\n", comp_buckets_size);
    unsigned int* tmp = (unsigned int*) malloc(comp_buckets_size * sizeof(unsigned int));
    gpu(cudaMemcpy(tmp, *d_candidates, comp_buckets_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("buckets:\n");
    for (int i = 0; i < comp_buckets_size; i++)
        printf("b(%u) ", tmp[i]);
    printf("\n\n");
    short* tmp2 = (short*) malloc(bsize * sizeof(short));
    gpu(cudaMemcpy(tmp2, d_partial_scores, bsize * sizeof(short), cudaMemcpyDeviceToHost));
    printf("partial_scores:\n");
    for (int i = 0; i < bsize; i++)
        printf("ps(%u) ", tmp2[i]);
    printf("\n\n");
    exit(0);
}



/*
 *  DOCUMENTATION
 */
__global__
void filtering_kernel_block (
	Entry* inv_index, int* count, int* index,
	unsigned int* pos, unsigned int* len, unsigned int* tokens, short* scores, 
    float threshold, unsigned int q_offset, unsigned int i_offset, 
    unsigned int block_size, unsigned int n
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
	{
        unsigned int query = i + q_offset;
		unsigned int query_pos = pos[ query ];
        unsigned int query_len = len[ query ];
        unsigned int query_jac_prefix_size = jac_max_prefix(query_len, threshold);
		unsigned int query_jac_min_size = (unsigned int) floor(jac_min_size(query_len, threshold));
        unsigned int query_jac_max_size = (unsigned int) floor(jac_max_size(query_len, threshold));

		for (unsigned j = 0; j < query_jac_prefix_size; j++)
		{
			unsigned int query_tpos = j;
			unsigned int token = tokens[ query_pos + j ];
			unsigned int list_size = count[ token ];
			unsigned int list_index = index[ token ] - list_size;

			for (unsigned k = 0; k < list_size; k++)
			{
				Entry e = inv_index[ list_index + k ];
				unsigned int source = (unsigned int) e.set_id;
				unsigned int source_tpos = (unsigned int) e.pos;

                /*
                 *  This was necessary since the inverted index is now complete
                 *  and not just a block index as before. So we have to check 
                 *  whether the set over there is in the bounds of the index block
                 */
                // if (query < source)
                if (query < source && source >= i_offset && source < i_offset + block_size)
                {
                	unsigned int source_len = len[ source ];

					unsigned int source_jac_min_size = (unsigned int) floor(jac_min_size(source_len, threshold));
                    unsigned int source_jac_max_size = (unsigned int) floor(jac_max_size(source_len, threshold));

					if (source_len < query_jac_min_size ||
                        source_len > query_jac_max_size ||
                        query_len < source_jac_min_size ||
                        query_len > source_jac_max_size)
                    {
						continue;
					}

                    unsigned long bucket = ((query - q_offset) * block_size) + (source - i_offset);
                    short score = scores[ bucket ];

                    if (score >= 0)
                    {
                        unsigned int query_rem = query_len - query_tpos;
                        unsigned int source_rem = source_len - source_tpos;
                        unsigned int min_rem = query_rem < source_rem ? query_rem : source_rem;
                        float jac_minoverlap = jac_min_overlap (query_len, source_len, threshold);

                        if ((score + 1 + min_rem) < jac_minoverlap)
                            scores[ bucket ] = -1;
                        else
                            scores[ bucket ] += 1;
                    }
                }
			}
		}
	}
}



/*
 *  DOCUMENTATION
 */
__host__
void filtering_block_new (
	sets_t* sets, Index inv_index,
	int* d_partial_scores, unsigned int* d_comp_buckets, int* d_nres, 
	unsigned int bsize, float threshold, unsigned int q_offset, unsigned int i_offset, 
	unsigned int block_size, unsigned int n_queries,
	unsigned int** d_candidates, unsigned int* candidates_size,
	char verbose
)
{
	unsigned long t0, t1;

	dim3 grid, block;
	get_grid_config_block(grid, block, block_size);

    // printf("<<<%u, %u>>>\n", grid.x, block.x);

	gpu(cudaMemset(d_partial_scores, 0, bsize * sizeof(int)));


	// CALLING THE FILTERING KERNEL ---------------------------------------------------------------
    if (verbose) {
    	fprintf(stderr, "\t\t\t* Calling filtering_kernel... ");
		t0 = ut_get_time_in_microseconds();
	}

	filtering_kernel_block_new <<<grid,block>>> (
		inv_index.d_lists, inv_index.d_count, inv_index.d_index,
		sets->d_pos, sets->d_len, sets->d_tokens, d_partial_scores, 
		threshold, q_offset, i_offset, block_size, n_queries
	);
	gpu(cudaDeviceSynchronize());

	if (verbose) {
		t1 = ut_get_time_in_microseconds();
		fprintf(stderr, "%g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }
    //---------------------------------------------------------------------------------------------


    // COMPACTING FILTERED CANDIDATES -------------------------------------------------------------
	if (verbose) {
        fprintf(stderr, "\t\t\t* Compacting filtered buckets... ");
        t0 = ut_get_time_in_microseconds();
    }

	gpu(cudaMemset(d_nres, 0, sizeof(int)));
	filter_k_int <<<grid,block>>> (d_comp_buckets, d_partial_scores, d_nres, bsize);
	gpu(cudaDeviceSynchronize());

	int nres;
	gpu(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));
	unsigned int comp_buckets_size = (unsigned int) nres;

    if (verbose) {
        t1 = ut_get_time_in_microseconds();
        fprintf(stderr, "%g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }
    //---------------------------------------------------------------------------------------------


	*d_candidates = d_comp_buckets;
    *candidates_size = comp_buckets_size;

    // printf("candidates_size: %u\n", comp_buckets_size);
    // unsigned int* tmp = (unsigned int*) malloc(comp_buckets_size * sizeof(unsigned int));
    // gpu(cudaMemcpy(tmp, *d_candidates, comp_buckets_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // printf("buckets:\n");
    // for (int i = 0; i < comp_buckets_size; i++)
    //     printf("b(%u) ", tmp[i]);
    // printf("\n\n");
    // int* tmp2 = (int*) malloc(bsize * sizeof(int));
    // gpu(cudaMemcpy(tmp2, d_partial_scores, bsize * sizeof(int), cudaMemcpyDeviceToHost));
    // printf("partial_scores:\n");
    // for (int i = 0; i < bsize; i++)
    //     printf("ps(%u) ", tmp2[i]);
    // printf("\n\n");
    // exit(0);
}



/*
 *  DOCUMENTATION
 */
__global__
void filtering_kernel_block_new (
	Entry* inv_index, int* count, int* index,
	unsigned int* pos, unsigned int* len, unsigned int* tokens, int* scores, 
    float threshold, unsigned int q_offset, unsigned int i_offset, 
    unsigned int block_size, unsigned int n
)
{
    unsigned int query = blockIdx.x + q_offset;
    unsigned int query_pos = pos[ query ];
    unsigned int query_len = len[ query ];
    unsigned int query_jac_prefix_size = jac_max_prefix(query_len, threshold);
    unsigned int query_jac_min_size = (unsigned int) floor(jac_min_size(query_len, threshold));

    for (unsigned int i = threadIdx.x; i < query_jac_prefix_size; i += blockDim.x)
    {

        unsigned int token = tokens[ query_pos + i ];
        unsigned int list_size = count[ token ];
        unsigned int list_index = index[ token ] - list_size;

        for (unsigned k = 0; k < list_size; k++)
        {
            Entry e = inv_index[ list_index + k ];
            unsigned int source = (unsigned int) e.set_id;

            if (query < source && source >= i_offset && source < i_offset + block_size)
            {
                unsigned int source_len = len[ source ];
                unsigned int source_jac_max_size = (unsigned int) floor(jac_max_size(source_len, threshold));

                if (!(source_len < query_jac_min_size || query_len > source_jac_max_size))
                {
                    unsigned long bucket = ((query - q_offset) * block_size) + (source - i_offset);
                    atomicAdd(&scores[bucket], 1);
                }
            }
        }
    }
}
