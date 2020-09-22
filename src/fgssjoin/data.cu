#include "../../include/fgssjoin/string/data.cuh"
#include "io.hpp"
#include "util.hpp"
#include "util.cuh"
#include "scan.cuh"
#include "similarity.cuh"



/*
 *  DOCUMENTATION
 */
__host__
sets_t* sets_new ()
{
	sets_t* sets = (sets_t*) malloc(sizeof(sets_t));
	
	sets->id     = (unsigned int*) malloc(sizeof(unsigned int));
	sets->pos    = (unsigned int*) malloc(sizeof(unsigned int));
	sets->len    = (unsigned int*) malloc(sizeof(unsigned int));
	sets->tokens = (unsigned int*) malloc(sizeof(unsigned int));

	return sets;
}



/*
 *  DOCUMENTATION
 */
__host__
sets_t* ppjoin_format (const char* filepath)
{
	unsigned long t0, t1;
	fprintf(stderr, "* Reading %s...\n", filepath);
	t0 = ut_get_time_in_microseconds();

	int     sets_count    = 0;
	int     terms_count   = 0;
	int     tokens_count  = 0;
	int*    intbuff       = (int*) malloc(sizeof(int));
	sets_t* sets          = sets_new();

	FILE* f = fopen(filepath, "rb");
	if (f == NULL) { fputs("File error", stderr); exit(1); }

	vector<unsigned int> vecids;
	vector<unsigned int> veclens;
	vector< vector<unsigned int> > vecsets;

	while (1) {
		int res, size = 0, *tokens;

		res = fread(intbuff, sizeof(int), 1, f);
		if (res == 1)
			vecids.push_back((unsigned int) *intbuff);
		else
			break;
		
		res = fread(intbuff, sizeof(int), 1, f);
		if (res == 1) {
			size = *intbuff;
			veclens.push_back((unsigned int) size);
		} else {
			fprintf(stdout, "Reading error.\n");
			exit(1);
		}

		tokens = (int*) malloc(size * sizeof(int));
		vector<unsigned int> vecset;
		res = fread(tokens, sizeof(int), size, f);
		if (res == size) {
			for (int i = 0; i < size; i++) {
				vecset.push_back((unsigned int) tokens[i]);
				if (tokens[i] >= terms_count)
					terms_count = tokens[i] + 1;
			}
			vecsets.push_back(vecset);
			free(tokens);
		} else {
			fprintf(stdout, "Reading error.\n");
			exit(1);
		}

		sets_count   += 1;
		tokens_count += size;
	}

	fclose(f);

	sets->num_sets   = sets_count;
	sets->num_terms  = terms_count;
	sets->num_tokens = tokens_count;

	sets->id  = (unsigned int*) realloc(sets->id, sets_count * sizeof(unsigned int));
	sets->len = (unsigned int*) realloc(sets->len, sets_count * sizeof(unsigned int));
	for (int i = 0; i < sets_count; i++) {
		sets->id[i]  = vecids[sets_count-1-i];
		sets->len[i] = veclens[sets_count-1-i];
	}
	vecids.clear();
	veclens.clear();

	size_t tmp = 0;
	sets->tokens = (unsigned int*) realloc(sets->tokens, tokens_count * sizeof(unsigned int));
	for (int i = sets_count-1; i >= 0; i--)
		for (int j = 0; j < vecsets[i].size(); j++)
			sets->tokens[tmp++] = vecsets[i][j];
	vecsets.clear();

	sets->pos = (unsigned int*) malloc(sets_count * sizeof(unsigned int));
	sets->pos[0] = 0;
	for (int i = 1; i < sets_count; i++)
		sets->pos[i] = sets->pos[i-1] + sets->len[i-1];

	t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "  - Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));

    fprintf(stderr, "Number of sets: %zu\n", sets->num_sets);
	fprintf(stderr, "Number of terms: %zu\n", sets->num_terms);
	fprintf(stderr, "Number of tokens: %zu\n", sets->num_tokens);

	return sets;
}



/*
 *  DOCUMENTATION
 */
__host__
void prepare_data (sets_t* sets, float threshold)
{
	unsigned long t0, t1;

	dim3 grid, block;
    get_grid_config(grid, block);

	// --------------------------------------------------------------------------------------------

    fprintf(stderr, "* Sending data to device...\n");
    t0 = ut_get_time_in_microseconds();

    gpu(cudaMalloc(&sets->d_pos, sets->num_sets * sizeof(unsigned int)));
    gpu(cudaMalloc(&sets->d_len, sets->num_sets * sizeof(unsigned int)));
    gpu(cudaMalloc(&sets->d_tokens, sets->num_tokens * sizeof(unsigned int)));

    gpu(cudaMemcpy(sets->d_pos, sets->pos, sets->num_sets * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpu(cudaMemcpy(sets->d_len, sets->len, sets->num_sets * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpu(cudaMemcpy(sets->d_tokens, sets->tokens, sets->num_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));

    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "  - Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));

    // --------------------------------------------------------------------------------------------

    fprintf(stderr, "* Creating mid_prefix tokens...\n");
    t0 = ut_get_time_in_microseconds();

    unsigned int* d_bin;
    unsigned int* d_scan;
    unsigned int  midpref_tokens_size;

    gpuAssert(cudaMalloc(&sets->d_midpref_pos, sets->num_sets * sizeof(unsigned int)));
    gpuAssert(cudaMalloc(&sets->d_midpref_len, sets->num_sets * sizeof(unsigned int)));

    // Selecting midprefix tokens for compaction
    select_and_index_midpref_tokens (
    	sets->d_tokens, sets->d_pos, sets->d_len, sets->d_midpref_len,
        sets->num_sets, sets->num_tokens, threshold, 
		&d_bin, &d_scan, &midpref_tokens_size
	);
	sets->num_midpref_tokens = midpref_tokens_size;

    // Calculating pos array for midprefix_tokens (through a scan)
    exclusive_scan <unsigned int> (sets->d_midpref_len, sets->d_midpref_pos, sets->num_sets);

    // Getting midprefix_tokens by compacting tokens array
    gpu(cudaMalloc(&sets->d_midpref_tokens, midpref_tokens_size * sizeof(unsigned int)));
    compact_midpref_tokens_kernel <<<grid,block>>> (
    	sets->d_midpref_tokens, sets->d_tokens, d_bin, d_scan, sets->num_tokens
    );
    gpu(cudaDeviceSynchronize());

    // Freeing compaction needed arrays
    gpu(cudaFree(d_bin));
    gpu(cudaFree(d_scan));

    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "  - Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));

    //---------------------------------------------------------------------------------------------
}



__host__
void select_and_index_midpref_tokens (
    unsigned int* d_tokens, unsigned int* d_pos, unsigned int* d_len, 
    unsigned int* d_midpref_len,
    unsigned int n_sets, unsigned int n_tokens, float threshold,
    unsigned int** d_select, unsigned int** d_index, unsigned int* select_size_out
)
{
	dim3 grid, block;
    get_grid_config(grid, block);

    unsigned int* d_bin;
    gpu(cudaMalloc(&d_bin, n_tokens * sizeof(unsigned int)));
    gpu(cudaMemset(d_bin, 0, n_tokens * sizeof(unsigned int)));

    select_midpref_tokens_kernel <<<grid,block>>> (d_bin, d_pos, d_len, d_midpref_len, threshold, n_sets);
    gpu(cudaDeviceSynchronize());

    unsigned int* d_scan;
    gpu(cudaMalloc(&d_scan, n_tokens * sizeof(unsigned int)));
    exclusive_scan <unsigned int> (d_bin, d_scan, n_tokens);

    unsigned int last_bin;
    unsigned int size;
    gpu(cudaMemcpy(&last_bin, d_bin + (n_tokens - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    gpu(cudaMemcpy(&size, d_scan + (n_tokens - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (last_bin == 1) size += 1;

    *d_select = d_bin;
    *d_index = d_scan;
    *select_size_out = size;
}



/*
 *  DOCUMENTATION
 */
__global__
void select_midpref_tokens_kernel (
	unsigned int* bin, unsigned int* pos, unsigned int* len, unsigned int* midpref_len,
    float threshold, unsigned int n_sets
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n_sets; i += blockDim.x * gridDim.x)
    {
        unsigned int prefix_size = jac_mid_prefix(len[i], threshold);
        midpref_len[i] = prefix_size;
        for (unsigned j = 0; j < prefix_size; j++)
        {
            bin[pos[i] + j] = 1;
        }
    }
}



/*
 *  DOCUMENTATION
 */
__global__
void compact_midpref_tokens_kernel (
	unsigned int* midpref_tokens, unsigned int* tokens,
    unsigned int* select, unsigned int* index, unsigned int n_tokens
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n_tokens; i += blockDim.x * gridDim.x)
    {
        if (select[i] == 1)
        {
            unsigned int pos = index[i];
            midpref_tokens[pos] = tokens[i];
        }
    }
}