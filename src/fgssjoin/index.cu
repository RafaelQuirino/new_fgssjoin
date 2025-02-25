#include "scan.cuh"
#include "index.cuh"
#include "similarity.cuh"



/*
 *  DOCUMENTATION
 */
Entry* create_midpref_entries (
    sets_t* sets, unsigned int* d_entry_pos, 
    unsigned int n_entries, unsigned int offset, unsigned int block_size
)
{
    dim3 grid, block;
    get_grid_config(grid, block);
    Entry* d_entries;
    gpuAssert(cudaMalloc(&d_entries, n_entries * sizeof(Entry)));
    midpref_kernel <<<grid, block>>> (
        d_entries, sets->d_midpref_pos, sets->d_midpref_len,
        sets->d_midpref_tokens, d_entry_pos, offset, block_size
    );
    gpu(cudaDeviceSynchronize());

    return d_entries;
}



/*
 *  DOCUMENTATION
 */
__global__
void midpref_kernel (
    Entry* entries, unsigned int* midpref_pos, unsigned int* midpref_len,
    unsigned int* midpref_tokens, unsigned int* entry_pos, 
    unsigned int offset, unsigned int n_sets
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n_sets; i += blockDim.x * gridDim.x)
    {
        int x = i + offset;
        for (unsigned j = 0; j < midpref_len[x]; j++) 
        {
            int set = x;
            int idx = midpref_pos[x] + j;
            int token = midpref_tokens[idx];
            entries[entry_pos[i] + j].set_id = set;
            entries[entry_pos[i] + j].term_id = token;
            entries[entry_pos[i] + j].pos = (int) j;
        }
    }
}



/*
 *  DOCUMENTATION
 */
__host__ 
Index inverted_index_block (
    sets_t* sets, float threshold, 
    unsigned int offset, unsigned int block_size
) 
{
    unsigned int *d_entry_pos, n_entries, last_midpref;
    gpu(cudaMalloc(&d_entry_pos, block_size * sizeof(unsigned int)));

    exclusive_scan <unsigned int> (sets->d_midpref_len + offset, d_entry_pos, block_size);
    
    gpu(cudaMemcpy(&n_entries, d_entry_pos+(block_size-1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    gpu(cudaMemcpy(&last_midpref, sets->d_midpref_len+(offset+block_size-1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    n_entries += last_midpref;
    
    Entry* d_entries = create_midpref_entries(sets, d_entry_pos, n_entries, offset, block_size);
    gpu(cudaFree(d_entry_pos));



    int num_docs = (int) sets->num_sets;
    int num_terms = (int) sets->num_terms;
    int num_entries = (int) n_entries;
    Entry *d_lists;
    int *d_count, *d_index;


    gpu(cudaMalloc(&d_lists, num_entries * sizeof(Entry)));
    gpu(cudaMalloc(&d_index, num_terms * sizeof(int)));
    gpu(cudaMalloc(&d_count, num_terms * sizeof(int)));   
    gpu(cudaMemset(d_count, 0, num_terms * sizeof(int)));


    dim3 grid, block;
    get_grid_config(grid, block);
    df_count_kernel <<<grid, block>>> (d_entries, d_count, num_entries);     
    exclusive_scan <int> (d_count, d_index, num_terms);
    inverted_index_kernel <<<grid, block>>> (d_entries, d_lists, d_index, num_entries);
    gpuAssert(cudaDeviceSynchronize());


    Index inv = Index(d_lists, d_index, d_count, num_docs, num_terms, num_entries);
    gpuAssert(cudaFree(d_entries));


    return inv;
}



/*
 *  DOCUMENTATIONS
 */
__host__ 
Index new_inverted_index_block (
    sets_t* sets, float threshold, 
    unsigned int offset, unsigned int block_size
)
{
    vector<Entry> entries;
    for (unsigned i = offset; i < offset+block_size; i++) {
        unsigned int pos = sets->pos[i], len = sets->len[i];
        unsigned int prefsize = jac_mid_prefix(len, threshold);

        for (unsigned j = 0; j < prefsize; j++) {
            unsigned int term = sets->tokens[pos + j];
            entries.push_back(Entry((int)i, (int)term, (int)j));
        }
    }

    Entry* d_entries;
    gpu(cudaMalloc(&d_entries, entries.size() * sizeof(Entry)));
    gpu(cudaMemcpy(d_entries, &entries[0], entries.size() * sizeof(Entry), cudaMemcpyHostToDevice));


    int num_docs = (int) sets->num_sets;
    int num_terms = (int) sets->num_terms;
    int num_entries = (int) entries.size();
    Entry *d_lists;
    int *d_count, *d_index;


    gpu(cudaMalloc(&d_lists, num_entries * sizeof(Entry)));
    gpu(cudaMalloc(&d_index, num_terms * sizeof(int)));
    gpu(cudaMalloc(&d_count, num_terms * sizeof(int)));   
    gpu(cudaMemset(d_count, 0, num_terms * sizeof(int)));


    dim3 grid, block;
    get_grid_config(grid, block);
    df_count_kernel <<<grid, block>>> (d_entries, d_count, num_entries);     
    exclusive_scan <int> (d_count, d_index, num_terms);
    inverted_index_kernel <<<grid, block>>> (d_entries, d_lists, d_index, num_entries);
    gpuAssert(cudaDeviceSynchronize());


    Index inv = Index(d_lists, d_index, d_count, num_docs, num_terms, num_entries);
    gpu(cudaFree(d_entries));


    return inv;   
}



/*
 *  DOCUMENTATION
 */
__host__ 
Index inverted_index (sets_t* sets, float threshold, char verbose) 
{
	unsigned long t0, t1;

    if (verbose) {
        fprintf(stderr, "* Building inverted index...\n");
        t0 = ut_get_time_in_microseconds();
    }

    Entry* d_entries = create_midpref_entries(sets, sets->num_midpref_tokens, 0, sets->num_sets);

    int num_sets    = (int) sets->num_sets;
    int num_terms   = (int) sets->num_terms;
    int num_entries = (int) sets->num_midpref_tokens;
    Entry *d_lists;
    int *d_count, *d_index;


    gpu(cudaMalloc(&d_lists, num_entries * sizeof(Entry)));
    gpu(cudaMalloc(&d_index, num_terms * sizeof(int)));
    gpu(cudaMalloc(&d_count, num_terms * sizeof(int)));   
    gpu(cudaMemset(d_count, 0, num_terms * sizeof(int)));


    dim3 grid, block;
    get_grid_config(grid, block);
    df_count_kernel <<<grid, block>>> (d_entries, d_count, num_entries);     
    exclusive_scan <int> (d_count, d_index, num_terms);
    inverted_index_kernel <<<grid, block>>> (d_entries, d_lists, d_index, num_entries);
    gpu(cudaDeviceSynchronize());


    Index index = Index(d_lists, d_index, d_count, num_sets, num_terms, num_entries);
    gpu(cudaFree(d_entries));

    if (verbose) {
        t1 = ut_get_time_in_microseconds();
        fprintf(stderr, "  - Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));
    }
    
    return index;
}



/*
 *  DOCUMENTATION
 */
__host__ 
void print_inverted_index (Index invindex, sets_t* sets)
{
	int num_terms   = invindex.num_terms;
	int num_entries = invindex.num_entries;
	int *index      = (int*) malloc(num_terms * sizeof(int));
	int *count      = (int*) malloc(num_terms * sizeof(int));
	Entry *entries  = (Entry*) malloc(num_entries * sizeof(Entry));

	gpu(cudaMemcpy(index, invindex.d_index, num_terms * sizeof(int), cudaMemcpyDeviceToHost));
	gpu(cudaMemcpy(count, invindex.d_count, num_terms * sizeof(int), cudaMemcpyDeviceToHost));
	gpu(cudaMemcpy(entries, invindex.d_lists, num_entries * sizeof(Entry), cudaMemcpyDeviceToHost));

	for (int i = 0; i < num_terms; i++) {
		int pos = i == 0 ? 0 : index[i-1];
		int len = count[i];
		fprintf(stderr, "term: [%d], size: %d\n", entries[pos].term_id, len);
		for (int j = 0; j < len; j++)
			fprintf(stderr, "(%u) ", sets->id[entries[pos + j].set_id]);
		fprintf(stderr, "\n\n");
	}

	free(index);
	free(count);
	free(entries);
}



/*
 *  DOCUMENTATION
 */
__host__
Entry* create_midpref_entries (
    sets_t* sets, unsigned int n_entries, 
    unsigned int offset, unsigned int block_size
)
{
    dim3 grid, block;
    get_grid_config(grid, block);
    
    Entry* d_entries;
    gpu(cudaMalloc(&d_entries, n_entries * sizeof(Entry)));
    
    midpref_kernel <<<grid, block>>> (d_entries, 
    	sets->d_midpref_pos, sets->d_midpref_len, sets->d_midpref_tokens, 
        offset, block_size
	);
    gpu(cudaDeviceSynchronize());

    return d_entries;
}



/*
 *  DOCUMENTATION
 */
__global__
void midpref_kernel (
    Entry* entries, unsigned int* midpref_pos, unsigned int* midpref_len,
    unsigned int* midpref_tokens, unsigned int offset, unsigned int n_sets
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n_sets; i += blockDim.x * gridDim.x)
    {
        int x = i + offset;
        for (unsigned j = 0; j < midpref_len[x]; j++) 
        {
            int set = x;
            int idx = midpref_pos[x] + j;
            int token = midpref_tokens[idx];
            entries[idx].set_id = set;
            entries[idx].term_id = token;
            entries[idx].pos = (int) j;
        }
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void df_count_kernel (Entry *entries, int *count, int n) 
{
    int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);
    int offset = block_size * (blockIdx.x);
    int lim = offset + block_size;
    if (lim >= n) lim = n;
    int size = lim - offset;

    entries += offset;

    for (int i = threadIdx.x; i < size; i+= blockDim.x) 
    {
        int term_id = entries[i].term_id;
        atomicAdd(count + term_id, 1);
    }
}



/*
 *  DOCUMENTATION
 */
__global__ 
void inverted_index_kernel(Entry *entries, Entry *lists, int *index, int n) 
{  
    int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);
    int offset = block_size * (blockIdx.x);
    int lim = offset + block_size;
    if (lim >= n) lim = n;
    int size = lim - offset;

    entries += offset;

    for (int i = threadIdx.x; i < size; i+= blockDim.x) 
    {
        Entry entry = entries[i];
        int pos = atomicAdd(index + entry.term_id, 1);
        lists[pos] = entry;
    }
}