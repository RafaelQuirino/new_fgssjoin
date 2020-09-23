#include "checking.cuh"



/*
 *  DOCUMENTATION
 */
__host__
void checking_block (
	short* d_partial_scores,
	unsigned int* d_buckets, sets_t* sets, float threshold, unsigned int csize, 
	unsigned int q_offset, unsigned int i_offset, unsigned int block_size,
	unsigned int** similar_pairs, unsigned short** scores, unsigned int* size,
	int* d_nres, char verbose
)
{
    unsigned short* d_scores;
    gpu(cudaMalloc(&d_scores, csize * sizeof(unsigned short)));
    gpu(cudaMemset(d_scores, 0, csize * sizeof(unsigned short)));

	unsigned int* d_comp_buckets;
	gpu(cudaMalloc(&d_comp_buckets, csize * sizeof(unsigned int)));

	unsigned short* d_comp_scores;
	gpu(cudaMalloc(&d_comp_scores, csize * sizeof(unsigned short)));

	dim3 grid, block;
	get_grid_config(grid, block);
	checking_kernel_block <<<grid,block>>> (
		d_partial_scores,
		d_buckets, d_scores,
		sets->d_pos, sets->d_len, sets->d_tokens, threshold,
		q_offset, i_offset, block_size, csize
	);
	gpu(cudaDeviceSynchronize());

	// COMPACTING FILTERED CANDIDATES -------------------------------------------------------------
    unsigned long t0, t1;

    if (verbose) {
        fprintf(stderr, "\n\t\t\t* Compacting checked buckets... ");
        t0 = ut_get_time_in_microseconds();
    }

    gpu(cudaMemset(d_nres, 0, sizeof(int)));
    chk_compact_1 <<<grid,block>>> (d_comp_buckets, d_buckets, d_scores, d_nres, csize);
    gpu(cudaDeviceSynchronize());

    gpu(cudaMemset(d_nres, 0, sizeof(int)));
    chk_compact_2 <<<grid,block>>> (d_comp_scores, d_scores, d_nres, csize);
    gpu(cudaDeviceSynchronize());

    // gpu(cudaMemset(d_nres, 0, sizeof(int)));
    // chk_compact_short_1 <<<grid,block>>> (d_comp_buckets, d_buckets, d_partial_scores, d_nres, block_size*block_size);
    // gpu(cudaDeviceSynchronize());

    // gpu(cudaMemset(d_nres, 0, sizeof(int)));
    // chk_compact_short_2 <<<grid,block>>> (d_comp_scores, d_partial_scores, d_nres, block_size*block_size);
    // gpu(cudaDeviceSynchronize());

    int nres;
    gpu(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));
    unsigned int comp_buckets_size = (unsigned int) nres;

    if (verbose) {
        t1 = ut_get_time_in_microseconds();
        fprintf(stderr, "%g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }

    // unsigned int* h_buckets = (unsigned int*) malloc(nres * sizeof(unsigned int));
    unsigned int* h_buckets; cudaMallocHost((void**)&h_buckets, nres * sizeof(unsigned int));
    gpu(cudaMemcpy(h_buckets, d_comp_buckets, nres * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // unsigned short* h_scores = (unsigned short*) malloc(nres * sizeof(unsigned short));
    unsigned short* h_scores; cudaMallocHost((void**)&h_scores, nres * sizeof(unsigned short));
    gpu(cudaMemcpy(h_scores, d_comp_scores, nres * sizeof(unsigned short), cudaMemcpyDeviceToHost));
    //---------------------------------------------------------------------------------------------

    gpu(cudaFree(d_comp_scores));
    gpu(cudaFree(d_comp_buckets));
    gpu(cudaFree(d_scores));

	*similar_pairs = h_buckets;
	*scores = h_scores;
    *size = comp_buckets_size;
}



/*
 *  DOCUMENTATION
 */
__global__
void checking_kernel_block (
	short* partial_scores,
	unsigned int* buckets, unsigned short* scores,
	unsigned int* pos, unsigned int* len, unsigned int* tokens, float threshold,
	unsigned int q_offset, unsigned int i_offset, unsigned int block_size, unsigned int csize
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < csize; i += blockDim.x * gridDim.x)
	{
		unsigned int bucket = buckets[i];

		unsigned int query = (bucket / block_size) + q_offset;
		unsigned int source = (bucket % block_size) + i_offset;
		unsigned int query_len = len[ query ];
		unsigned int source_len = len[ source ];
		int minoverlap = floor(jac_min_overlap(query_len, source_len, threshold));


		// Simpler version, without reusing
		// partial scores information...
		//----------------------------------
		unsigned int p1 = 0, p2 = 0;
		unsigned short score = 0;


		// More refined solution, reusing
		// partial scores information...
		//--------------------------------
		score = (unsigned short) partial_scores[bucket];
		unsigned int query_prefix = jac_max_prefix(query_len, threshold);
		unsigned int source_prefix = jac_mid_prefix(source_len, threshold);
		unsigned int tkn1 = tokens[pos[query] + (query_prefix-1)];
		unsigned int tkn2 = tokens[pos[source] + (source_prefix-1)];

		if (tkn1 <= tkn2) p1 = query_prefix;
		else              p2 = source_prefix;


		while (p1 < query_len && p2 < source_len)
		{
			unsigned int tkn1 = tokens[ pos[query] + p1 ];
			unsigned int tkn2 = tokens[ pos[source] + p2 ];

			if ((p1 == query_len-1 && tkn1 < tkn2) ||
				(p2 == source_len-1 && tkn2 < tkn1))
				break;

			if (tkn1 == tkn2)
			{
				score++;
				p1++; p2++;

			}
			else
			{
				unsigned int whichset = tkn1 < tkn2 ? 1 : 2;
				unsigned int rem;

				if (whichset == 1) rem = query_len - p1;
				else rem = source_len - p2;

				if ((rem + score) < minoverlap) {
					scores[i] = 0;
                    partial_scores[bucket] = 0;
					break;
				}

				if (whichset == 1) p1++;
				else p2++;
			}
		}

        if (score < minoverlap) {
            scores[i] = 0;
            partial_scores[bucket] = 0;
        } else {
		  scores[i] = score;
          partial_scores[bucket] = (short) score;
        }
	}
}



//=====================================================================================================================
//=====================================================================================================================
//=====================================================================================================================



/*
 *  DOCUMENTATION
 */
__host__
void checking_block_int (
    int* d_partial_scores,
    unsigned int* d_buckets, sets_t* sets, float threshold, unsigned int csize, 
    unsigned int q_offset, unsigned int i_offset, unsigned int block_size,
    unsigned int** similar_pairs, unsigned short** scores, unsigned int* size,
    int* d_nres, char verbose
)
{
    unsigned short* d_scores;
    gpu(cudaMalloc(&d_scores, csize * sizeof(unsigned short)));
    gpu(cudaMemset(d_scores, 0, csize * sizeof(unsigned short)));

    unsigned int* d_comp_buckets;
    gpu(cudaMalloc(&d_comp_buckets, csize * sizeof(unsigned int)));

    unsigned short* d_comp_scores;
    gpu(cudaMalloc(&d_comp_scores, csize * sizeof(unsigned short)));

    dim3 grid, block;
    get_grid_config(grid, block);
    checking_kernel_block_int <<<grid,block>>> (
        d_partial_scores,
        d_buckets, d_scores,
        sets->d_pos, sets->d_len, sets->d_tokens, threshold,
        q_offset, i_offset, block_size, csize
    );
    gpu(cudaDeviceSynchronize());

    // COMPACTING FILTERED CANDIDATES -------------------------------------------------------------
    unsigned long t0, t1;

    if (verbose) {
        fprintf(stderr, "\n\t\t\t* Compacting checked buckets... ");
        t0 = ut_get_time_in_microseconds();
    }

    gpu(cudaMemset(d_nres, 0, sizeof(int)));
    chk_compact_1 <<<grid,block>>> (d_comp_buckets, d_buckets, d_scores, d_nres, csize);
    gpu(cudaDeviceSynchronize());

    gpu(cudaMemset(d_nres, 0, sizeof(int)));
    chk_compact_2 <<<grid,block>>> (d_comp_scores, d_scores, d_nres, csize);
    gpu(cudaDeviceSynchronize());

    int nres;
    gpu(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));
    unsigned int comp_buckets_size = (unsigned int) nres;

    if (verbose) {
        t1 = ut_get_time_in_microseconds();
        fprintf(stderr, "%g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }

    unsigned int* h_buckets = (unsigned int*) malloc(nres * sizeof(unsigned int));
    gpu(cudaMemcpy(h_buckets, d_comp_buckets, nres * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    unsigned short* h_scores = (unsigned short*) malloc(nres * sizeof(unsigned short));
    gpu(cudaMemcpy(h_scores, d_comp_scores, nres * sizeof(unsigned short), cudaMemcpyDeviceToHost));
    //---------------------------------------------------------------------------------------------

    gpu(cudaFree(d_comp_scores));
    gpu(cudaFree(d_comp_buckets));
    gpu(cudaFree(d_scores));

    *similar_pairs = h_buckets;
    *scores = h_scores;
    *size = comp_buckets_size;
}



/*
 *  DOCUMENTATION
 */
__global__
void checking_kernel_block_int (
    int* partial_scores,
    unsigned int* buckets, unsigned short* scores,
    unsigned int* pos, unsigned int* len, unsigned int* tokens, float threshold,
    unsigned int q_offset, unsigned int i_offset, unsigned int block_size, unsigned int csize
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < csize; i += blockDim.x * gridDim.x)
    {
        unsigned int bucket = buckets[i];

        unsigned int query = (bucket / block_size) + q_offset;
        unsigned int source = (bucket % block_size) + i_offset;
        unsigned int query_len = len[ query ];
        unsigned int source_len = len[ source ];
        int minoverlap = floor(jac_min_overlap(query_len, source_len, threshold));


        // Simpler version, without reusing
        // partial scores information...
        //----------------------------------
        unsigned int p1 = 0, p2 = 0;
        unsigned short score = 0;


        // More refined solution, reusing
        // partial scores information...
        //--------------------------------
        score = (unsigned short) partial_scores[bucket];
        unsigned int query_prefix = jac_max_prefix(query_len, threshold);
        unsigned int source_prefix = jac_mid_prefix(source_len, threshold);
        unsigned int tkn1 = tokens[pos[query] + (query_prefix-1)];
        unsigned int tkn2 = tokens[pos[source] + (source_prefix-1)];

        if (tkn1 <= tkn2) p1 = query_prefix;
        else              p2 = source_prefix;


        while (p1 < query_len && p2 < source_len)
        {
            unsigned int tkn1 = tokens[ pos[query] + p1 ];
            unsigned int tkn2 = tokens[ pos[source] + p2 ];

            if ((p1 == query_len-1 && tkn1 < tkn2) ||
                (p2 == source_len-1 && tkn2 < tkn1))
                break;

            if (tkn1 == tkn2)
            {
                score++;
                p1++; p2++;

            }
            else
            {
                unsigned int whichset = tkn1 < tkn2 ? 1 : 2;
                unsigned int rem;

                if (whichset == 1) rem = query_len - p1;
                else rem = source_len - p2;

                if ((rem + score) < minoverlap) {
                    scores[i] = 0;
                    break;
                }

                if (whichset == 1) p1++;
                else p2++;
            }
        }

        if (score < minoverlap)
            scores[i] = 0;
        else
          scores[i] = score;
    }
}



//=====================================================================================================================
//=====================================================================================================================
//=====================================================================================================================



/*
 *  DOCUMENTATION
 */
__host__
void checking_block_very_new (
    short* d_partial_scores, sets_t* sets, float threshold,
    unsigned int q_offset, unsigned int i_offset, unsigned int block_size,
    unsigned int** similar_pairs, unsigned short** scores, unsigned int* size,
    int* d_nres, char verbose
)
{
    dim3 grid, block;
    get_grid_config(grid, block);
    gpu(cudaMemset(d_nres, 0, sizeof(int)));
    checking_kernel_block_very_new <<<grid,block>>> (
        d_partial_scores,
        sets->d_pos, sets->d_len, sets->d_tokens, threshold,
        q_offset, i_offset, block_size,
        d_nres
    );
    gpu(cudaDeviceSynchronize());

    int nres;
    gpu(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));
    unsigned int simsize = (unsigned int) nres;

    unsigned int* d_comp_buckets;
    gpu(cudaMalloc(&d_comp_buckets, simsize * sizeof(unsigned int)));
    
    unsigned short* d_comp_scores;
    gpu(cudaMalloc(&d_comp_scores, simsize * sizeof(unsigned short)));

    // COMPACTING FILTERED CANDIDATES -------------------------------------------------------------
    unsigned long t0, t1;

    if (verbose) {
        fprintf(stderr, "\n\t\t\t* Compacting checked buckets... ");
        t0 = ut_get_time_in_microseconds();
    }

    gpu(cudaMemset(d_nres, 0, sizeof(int)));
    chk_compact_new_buckets <<<grid,block>>> (d_comp_buckets, d_partial_scores, d_nres, block_size*block_size);
    gpu(cudaDeviceSynchronize());
    
    gpu(cudaMemset(d_nres, 0, sizeof(int)));
    chk_compact_new_scores <<<grid,block>>> (d_comp_scores, d_partial_scores, d_nres, block_size*block_size);
    gpu(cudaDeviceSynchronize());

    if (verbose) {
        t1 = ut_get_time_in_microseconds();
        fprintf(stderr, "%g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }

    // unsigned int* h_buckets = (unsigned int*) malloc(simsize * sizeof(unsigned int));
    unsigned int* h_buckets; cudaMallocHost((void**)&h_buckets, simsize * sizeof(unsigned int));
    gpu(cudaMemcpy(h_buckets, d_comp_buckets, simsize * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // unsigned short* h_scores = (unsigned short*) malloc(simsize * sizeof(unsigned short));
    unsigned short* h_scores; cudaMallocHost((void**)&h_scores, simsize * sizeof(unsigned short));
    gpu(cudaMemcpy(h_scores, d_comp_scores, simsize * sizeof(unsigned short), cudaMemcpyDeviceToHost));
    //---------------------------------------------------------------------------------------------

    gpu(cudaFree(d_comp_buckets));
    gpu(cudaFree(d_comp_scores));

    *similar_pairs = h_buckets;
    *scores = h_scores;
    *size = simsize;
}



/*
 *  DOCUMENTATION
 */
__global__
void checking_kernel_block_very_new (
    short* partial_scores,
    unsigned int* pos, unsigned int* len, unsigned int* tokens, 
    float threshold, unsigned int q_offset, unsigned int i_offset, unsigned int block_size, 
    int *simcount
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < block_size * block_size; i += blockDim.x * gridDim.x)
    {
        unsigned int bucket = i;
        short score = partial_scores[bucket];

        if (score > 0) {

            unsigned int query = (bucket / block_size) + q_offset;
            unsigned int source = (bucket % block_size) + i_offset;
            unsigned int query_len = len[ query ];
            unsigned int source_len = len[ source ];
            int minoverlap = floor(jac_min_overlap(query_len, source_len, threshold));

            unsigned int p1 = 0, p2 = 0;
            unsigned int query_prefix = jac_max_prefix(query_len, threshold);
            unsigned int source_prefix = jac_mid_prefix(source_len, threshold);
            unsigned int tkn1 = tokens[pos[query] + (query_prefix-1)];
            unsigned int tkn2 = tokens[pos[source] + (source_prefix-1)];

            if (tkn1 <= tkn2) p1 = query_prefix;
            else              p2 = source_prefix;


            while (p1 < query_len && p2 < source_len)
            {
                unsigned int tkn1 = tokens[ pos[query] + p1 ];
                unsigned int tkn2 = tokens[ pos[source] + p2 ];

                if ((p1 == query_len-1 && tkn1 < tkn2) ||
                    (p2 == source_len-1 && tkn2 < tkn1))
                    break;

                if (tkn1 == tkn2)
                {
                    score++;
                    p1++; p2++;

                }
                else
                {
                    unsigned int whichset = tkn1 < tkn2 ? 1 : 2;
                    unsigned int rem;

                    if (whichset == 1) rem = query_len - p1;
                    else rem = source_len - p2;

                    if ((rem + score) < minoverlap) {
                        partial_scores[bucket] = -1;
                        break;
                    }

                    if (whichset == 1) p1++;
                    else p2++;
                }
            }

            if (score < minoverlap) {
                partial_scores[bucket] = -1;
            } else {
                partial_scores[bucket] = (short) score;
                atomicAdd(simcount, 1);
            }
        }
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_new_buckets (
    unsigned int *buckets, 
    const short *scores_block, 
    int *nres,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        if (scores_block[i] > 0) {
            buckets[atomicAdd(nres, 1)] = i;
        }
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_new_scores (
    unsigned short *scores, 
    const short *scores_block, 
    int *nres,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        if (scores_block[i] > 0) {
            scores[atomicAdd(nres, 1)] = (unsigned short) scores_block[i];
        }
    }
}



//=====================================================================================================================
//=====================================================================================================================
//=====================================================================================================================



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_1 (
    unsigned int *dst, 
    unsigned int *buckets, 
    const unsigned short *scores, 
    int *nres, 
    int n
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        if (scores[i] > 0) {
            dst[atomicAdd(nres, 1)] = buckets[i];
        }
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_2 (
    unsigned short *dst, 
    const unsigned short *scores, 
    int *nres, 
    int n
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        unsigned short score = scores[i];
        if (score > 0) {
            dst[atomicAdd(nres, 1)] = score;
        }
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_short_1 (
    unsigned int *dst, 
    unsigned int *buckets, 
    const short *scores, 
    int *nres, 
    int n
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        if (scores[i] > 0) {
            dst[atomicAdd(nres, 1)] = buckets[i];
        }
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void chk_compact_short_2 (
    unsigned short *dst, 
    const short *scores, 
    int *nres, 
    int n
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) 
    {
        short score = scores[i];
        if (score > 0) {
            dst[atomicAdd(nres, 1)] = score;
        }
    }
}