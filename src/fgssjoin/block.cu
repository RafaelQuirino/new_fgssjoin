#include <math.h>

#include "util.hpp"
#include "block.cuh"
#include "checking.cuh"
#include "filtering.cuh"
#include "similarity.cuh"



/*
 *  DOCUMENTATION
 */
int get_block_size (sets_t* sets, float threshold, char verbose) 
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
            if (verbose) {
                fprintf(stderr, "Device: %d\nTotal Memory: %zu MB, Free Memory: %zu MB\n",
                    0, //nID,
                    uTotalMemoryInBytes / ( 1024 * 1024 ),
                    uCurAvailMemoryInBytes / ( 1024 * 1024 )
                );
            }
        }
        // cuCtxDetach( context ); // Destroy context
    // }

    float limit = 0.9; // Percentage of available memory to take
    int block_size = floor(floor(sqrt((float)uCurAvailMemoryInBytes/6)) * limit);
    if (limit == 0.0)
        block_size = 1;

    if (block_size >= sets->num_sets)
        block_size = sets->num_sets;

    return block_size;
}



/*
 *  DOCUMENTATION
 */
__host__
void process_blocks (sets_t* sets, Index inv_index, float threshold, char verbose)
{
    if (verbose)
        fprintf(stderr, "\nBlock processing.\n");

    unsigned long t0, t1;
    unsigned long sim_pairs_count = 0;
    unsigned long candidates_count = 0;
    unsigned long candidates_checked_count = 0;
    double t;

    // Getting the block size -----------------------------
    int block_size = get_block_size(sets, threshold, verbose);
    if (verbose)
        fprintf(stderr, "block_size: %d\n", block_size);
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

    if (verbose)
        ut_print_separator("-", 99);

    for (int i = 0; i < n_blocks; i++)
    {
        int i_offset = i * block_size;
        int bsize = i == n_blocks - 1 ? last_block : block_size;

        if (verbose)
            fprintf(stderr, "\nINDEX BLOCK (%d)... (size: %d, i_offset: %d)\n", i, bsize, i_offset);

        for (int j = 0; j <= i; j++)
        {
            int q_offset = j * block_size;
            int bsize = j == n_blocks - 1 ? last_block : block_size;

            if (verbose)
                fprintf(stderr, "\n\tQUERYING WITH BLOCK (%d)... (size: %d, q_offset: %d)\n\n", j, bsize, q_offset);

            //######################################################################
            // If the last set from the query block is bigger than the
            // maxsize of the first set in the index, we can skip this block
            //######################################################################
            int q_size = (int) sets->len[(q_offset+bsize) - 1];
            int s_maxsize = floor(jac_max_size(sets->len[i_offset], threshold));
            if (q_size > s_maxsize) {
                if (verbose) {
                    fprintf(stderr, "\t\t => The last set from this block is bigger than\n");
                    fprintf(stderr, "\t\t    the maxsize of the first set from the index !\n");
                    fprintf(stderr, "\t\t    Skipping...\n");
                }
                continue;
            }
            //######################################################################

            // FILTERING -----------------------------------------------------------
            if (verbose) {
                fprintf(stderr, "\t\t => FILTERING... \n");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int *d_candidates, csize;

            filtering_block (
                sets, inv_index,
                d_partial_scores, d_comp_buckets, d_compidx,
                candsize, threshold, q_offset, i_offset, block_size, bsize,
                &d_candidates,  &csize,
                verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
                fprintf(stderr, "\t\t    no of candidates: %d\n", csize);
            }
            //----------------------------------------------------------------------

            // CHECKING ------------------------------------------------------------
            if (verbose) {
                fprintf(stderr,"\n\t\t => CHECKING... ");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int *similar_pairs;
            unsigned short *scores;
            unsigned int size;

            checking_block (
                d_partial_scores,
                d_candidates, sets, threshold, csize,
                q_offset, i_offset, block_size,
                &similar_pairs, &scores, &size,
                d_compidx, verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
            }
            //----------------------------------------------------------------------

            candidates_count += csize;
            candidates_checked_count += size;

            t0 = ut_get_time_in_microseconds();
            for (unsigned k = 0; k < size; k++) {
                unsigned int   bucket = similar_pairs[k];
                unsigned int   query  = (bucket / block_size) + q_offset;
                unsigned int   source = (bucket % block_size) + i_offset;
                unsigned short score  = scores[k];

                if (score > 0) {
                    float similarity = jac_similarity(sets->len[query], sets->len[source], score);
                    if (similarity >= threshold)  {
                        fprintf(stdout, "%d %d %f\n", sets->id[query], sets->id[source], similarity);
                        sim_pairs_count++;
                    }
                }
            }
            t1 = ut_get_time_in_microseconds();
            t += ut_interval_in_miliseconds(t0,t1);

            cudaFreeHost(similar_pairs);
            cudaFreeHost(scores);
        }

        if (verbose)
            ut_print_separator("-", 99);
    }

    fprintf(stderr, "# Candidates: %lu\n", candidates_count);
    fprintf(stderr, "# Intersections: %lu\n", candidates_checked_count);
    fprintf(stderr, "# Results: %lu\n", sim_pairs_count);
    fprintf(stderr, "  - Time rendering: %g ms\n", t);
}



//=====================================================================================================================
//=====================================================================================================================
//=====================================================================================================================



/*
 *  DOCUMENTATION
 */
int get_block_size_index (sets_t* sets, float threshold, char verbose) 
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
            if (verbose) {
                fprintf(stderr, "Device: %d\nTotal Memory: %zu MB, Free Memory: %zu MB\n",
                    0, //nID,
                    uTotalMemoryInBytes / ( 1024 * 1024 ),
                    uCurAvailMemoryInBytes / ( 1024 * 1024 )
                );
            }
        }
        // cuCtxDetach( context ); // Destroy context
    // }

    float limit = 0.9; // Percentage of available memory to take
    int block_size = floor(floor(sqrt((float)uCurAvailMemoryInBytes/6)) * limit);
    if (limit == 0.0)
        block_size = 1;

    if (block_size >= sets->num_sets)
        block_size = sets->num_sets;

    return block_size;
}



/*
 *  DOCUMENTATION
 */
__host__
void process_blocks_index (sets_t* sets, float threshold, char verbose)
{
    if (verbose)
        fprintf(stderr, "\nBlock processing with index construction.\n");

    unsigned long t0, t1;
    unsigned long sim_pairs_count = 0;
    unsigned long candidates_count = 0;
    unsigned long candidates_checked_count = 0;
    double t;

    // Getting the block size -----------------------------
    int block_size = get_block_size_index(sets, threshold, verbose);
    if (verbose)
        fprintf(stderr, "block_size: %d\n", block_size);
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

    if (verbose)
        ut_print_separator("-", 99);

    for (int i = 0; i < n_blocks; i++)
    {
        int i_offset = i * block_size;
        int bsize = i == n_blocks - 1 ? last_block : block_size;

        if (verbose) {
            fprintf(stderr, "\nCREATING INDEX BLOCK (%d)... (size: %d, i_offset: %d)\n", i, bsize, i_offset);
            t0 = ut_get_time_in_microseconds();
        }

        Index inv_index = new_inverted_index_block(sets, threshold, i_offset, bsize);

        if (verbose) {
            t1 = ut_get_time_in_microseconds();
            fprintf(stderr, "DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
        }

        for (int j = 0; j <= i; j++)
        {
            int q_offset = j * block_size;
            int bsize = j == n_blocks - 1 ? last_block : block_size;

            if (verbose)
                fprintf(stderr, "\n\tQUERYING WITH BLOCK (%d)... (size: %d, q_offset: %d)\n\n", j, bsize, q_offset);

            //######################################################################
            // If the last set from the query block is bigger than the
            // maxsize of the first set in the index, we can skip this block
            //######################################################################
            int q_size = (int) sets->len[(q_offset+bsize) - 1];
            int s_maxsize = floor(jac_max_size(sets->len[i_offset], threshold));
            if (q_size > s_maxsize) {
                if (verbose) {
                    fprintf(stderr, "\t\t => The last set from this block is bigger than\n");
                    fprintf(stderr, "\t\t    the maxsize of the first set from the index !\n");
                    fprintf(stderr, "\t\t    Skipping...\n");
                }
                continue;
            }
            //######################################################################

            // FILTERING -----------------------------------------------------------
            if (verbose) {
                fprintf(stderr, "\t\t => FILTERING... \n");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int *d_candidates, csize;

            filtering_block_index (
                sets, inv_index,
                d_partial_scores, d_comp_buckets, d_compidx,
                candsize, threshold, q_offset, i_offset, block_size, bsize,
                &d_candidates,  &csize,
                verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
                fprintf(stderr, "\t\t    no of candidates: %d\n", csize);
            }
            //----------------------------------------------------------------------

            // CHECKING ------------------------------------------------------------
            if (verbose) {
                fprintf(stderr,"\n\t\t => CHECKING... ");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int *similar_pairs;
            unsigned short *scores;
            unsigned int size;

            checking_block (
                d_partial_scores,
                d_candidates, sets, threshold, csize,
                q_offset, i_offset, block_size,
                &similar_pairs, &scores, &size,
                d_compidx, verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
            }
            //----------------------------------------------------------------------

            candidates_count += csize;
            candidates_checked_count += size;

            t0 = ut_get_time_in_microseconds();
            for (unsigned k = 0; k < size; k++) {
                unsigned int   bucket = similar_pairs[k];
                unsigned int   query  = (bucket / block_size) + q_offset;
                unsigned int   source = (bucket % block_size) + i_offset;
                unsigned short score  = scores[k];

                if (score > 0) {
                    float similarity = jac_similarity(sets->len[query], sets->len[source], score);
                    if (similarity >= threshold)  {
                        fprintf(stdout, "%d %d %f\n", sets->id[query], sets->id[source], similarity);
                        sim_pairs_count++;
                    }
                }
            }
            t1 = ut_get_time_in_microseconds();
            t += ut_interval_in_miliseconds(t0,t1);

            cudaFreeHost(similar_pairs);
            cudaFreeHost(scores);
        }

        if (verbose)
            ut_print_separator("-", 99);
    }

    fprintf(stderr, "# Candidates: %lu\n", candidates_count);
    fprintf(stderr, "# Intersections: %lu\n", candidates_checked_count);
    fprintf(stderr, "# Results: %lu\n", sim_pairs_count);
    fprintf(stderr, "  - Time rendering: %g ms\n", t);
}



//=====================================================================================================================
//=====================================================================================================================
//=====================================================================================================================



/*
 *  DOCUMENTATION
 */
int get_block_size_int (sets_t* sets, float threshold, char verbose) 
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
            if (verbose) {
                fprintf(stderr, "Device: %d\nTotal Memory: %zu MB, Free Memory: %zu MB\n",
                    0, //nID,
                    uTotalMemoryInBytes / ( 1024 * 1024 ),
                    uCurAvailMemoryInBytes / ( 1024 * 1024 )
                );
            }
        }
        // cuCtxDetach( context ); // Destroy context
    // }

    float limit = 0.9; // Percentage of available memory to take
    int block_size = floor(floor(sqrt((float)uCurAvailMemoryInBytes/8)) * limit);
    if (limit == 0.0)
        block_size = 1;

    if (block_size >= sets->num_sets)
        block_size = sets->num_sets;

    return block_size;
}



/*
 *  DOCUMENTATION
 */
__host__
void process_blocks_int (sets_t* sets, Index inv_index, float threshold, char verbose)
{
    if (verbose)
        fprintf(stderr, "\nBlock processing.\n");

    unsigned long t0, t1;
    unsigned long sim_pairs_count = 0;
    unsigned long candidates_count = 0;
    unsigned long candidates_checked_count = 0;
    double t;

    // Getting the block size -----------------------------
    int block_size = get_block_size_int(sets, threshold, verbose);
    if (verbose)
        fprintf(stderr, "block_size: %d\n", block_size);
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
    int* d_partial_scores;         // partial scores in filtering step
    unsigned int* d_comp_buckets;  // filtered candidates (buckets)

    gpu(cudaMalloc(&d_compidx, sizeof(int)));
    gpu(cudaMalloc(&d_partial_scores, candsize * sizeof(int)));
    gpu(cudaMalloc(&d_comp_buckets, candsize * sizeof(unsigned int)));
    //------------------------------------------------------------------------------

    if (verbose)
        ut_print_separator("-", 99);

    for (int i = 0; i < n_blocks; i++)
    {
        int i_offset = i * block_size;
        int bsize = i == n_blocks - 1 ? last_block : block_size;

        if (verbose)
            fprintf(stderr, "\nINDEX BLOCK (%d)... (size: %d, i_offset: %d)\n", i, bsize, i_offset);

        for (int j = 0; j <= i; j++)
        {
            int q_offset = j * block_size;
            int bsize = j == n_blocks - 1 ? last_block : block_size;

            if (verbose)
                fprintf(stderr, "\n\tQUERYING WITH BLOCK (%d)... (size: %d, q_offset: %d)\n\n", j, bsize, q_offset);

            //######################################################################
            // If the last set from the query block is bigger than the
            // maxsize of the first set in the index, we can skip this block
            //######################################################################
            int q_size = (int) sets->len[(q_offset+bsize) - 1];
            int s_maxsize = floor(jac_max_size(sets->len[i_offset], threshold));
            if (q_size > s_maxsize) {
                if (verbose) {
                    fprintf(stderr, "\t\t => The last set from this block is bigger than\n");
                    fprintf(stderr, "\t\t    the maxsize of the first set from the index !\n");
                    fprintf(stderr, "\t\t    Skipping...\n");
                }
                continue;
            }
            //######################################################################

            // FILTERING -----------------------------------------------------------
            if (verbose) {
                fprintf(stderr, "\t\t => FILTERING... \n");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int *d_candidates, csize;

            filtering_block_new (
                sets, inv_index,
                d_partial_scores, d_comp_buckets, d_compidx,
                candsize, threshold, q_offset, i_offset, block_size, bsize,
                &d_candidates,  &csize,
                verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
                fprintf(stderr, "\t\t    no of candidates: %d\n", csize);
            }
            //----------------------------------------------------------------------

            // CHECKING ------------------------------------------------------------
            if (verbose) {
                fprintf(stderr,"\n\t\t => CHECKING... ");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int *similar_pairs;
            unsigned short *scores;
            unsigned int size;

            checking_block_int (
                d_partial_scores,
                d_candidates, sets, threshold, csize,
                q_offset, i_offset, block_size,
                &similar_pairs, &scores, &size,
                d_compidx, verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, " %g ms\n\n", ut_interval_in_miliseconds(t0,t1));
            }
            //----------------------------------------------------------------------

            candidates_count += csize;
            candidates_checked_count += size;

            t0 = ut_get_time_in_microseconds();
            for (unsigned k = 0; k < size; k++) {
                unsigned int   bucket = similar_pairs[k];
                unsigned int   query  = (bucket / block_size) + q_offset;
                unsigned int   source = (bucket % block_size) + i_offset;
                unsigned short score  = scores[k];

                if (score > 0) {
                    float similarity = jac_similarity(sets->len[query], sets->len[source], score);
                    if (similarity >= threshold)  {
                        fprintf(stdout, "%d %d %f\n", sets->id[query], sets->id[source], similarity);
                        sim_pairs_count++;
                    }
                }
            }
            t1 = ut_get_time_in_microseconds();
            t += ut_interval_in_miliseconds(t0,t1);

            free(similar_pairs);
            free(scores);
        }

        if (verbose)
            ut_print_separator("-", 99);
    }

    fprintf(stderr, "# Candidates: %lu\n", candidates_count);
    fprintf(stderr, "# Intersections: %lu\n", candidates_checked_count);
    fprintf(stderr, "# Results: %lu\n", sim_pairs_count);
    fprintf(stderr, "  - Time rendering: %g ms\n", t);
}



//=====================================================================================================================
//=====================================================================================================================
//=====================================================================================================================



/*
 *  DOCUMENTATION
 */
int get_block_size_very_new (sets_t* sets, float threshold, char verbose) 
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
            if (verbose) {
                fprintf(stderr, "Device: %d\nTotal Memory: %zu MB, Free Memory: %zu MB\n",
                    0, //nID,
                    uTotalMemoryInBytes / ( 1024 * 1024 ),
                    uCurAvailMemoryInBytes / ( 1024 * 1024 )
                );
            }
        }
        // cuCtxDetach( context ); // Destroy context
    // }

    float limit = 0.2; // Percentage of available memory to take
    int block_size = floor(floor(sqrt((float)uCurAvailMemoryInBytes/2)) * limit);
    if (limit == 0.0)
        block_size = 1;

    if (block_size >= sets->num_sets)
        block_size = sets->num_sets;

    return block_size;
}



/*
 *  DOCUMENTATION
 */
__host__
void process_blocks_very_new (sets_t* sets, Index inv_index, float threshold, char verbose)
{
    if (verbose)
        fprintf(stderr, "\nBlock processing.\n");

    unsigned long t0, t1;
    unsigned long sim_pairs_count = 0;
    unsigned long candidates_count = 0;
    unsigned long candidates_checked_count = 0;
    double t;

    // Getting the block size -----------------------------
    int block_size = get_block_size_very_new(sets, threshold, verbose);
    if (verbose)
        fprintf(stderr, "block_size: %d\n", block_size);
    //-----------------------------------------------------

    // Blocks logic ---------------------------------------
    int div        = sets->num_sets / block_size;
    int mod        = sets->num_sets % block_size;
    int n_blocks   = mod == 0 ? div : div + 1;
    int last_block = mod == 0 ? block_size : mod;
    //-----------------------------------------------------

    // CREATING SCORES AND BUCKETS ARRAYS ------------------------------------------
    int* d_nres;                   // for counts and compactions
    short* d_partial_scores;       // partial scores in filtering step

    gpu(cudaMalloc(&d_nres, sizeof(int)));
    gpu(cudaMalloc(&d_partial_scores, block_size * block_size * sizeof(short)));
    //------------------------------------------------------------------------------

    if (verbose)
        ut_print_separator("-", 99);

    for (int i = 0; i < n_blocks; i++)
    {
        int i_offset = i * block_size;
        int bsize = i == n_blocks - 1 ? last_block : block_size;

        if (verbose)
            fprintf(stderr, "\nINDEX BLOCK (%d)... (size: %d, i_offset: %d)\n", i, bsize, i_offset);

        for (int j = 0; j <= i; j++)
        {
            int q_offset = j * block_size;
            int bsize = j == n_blocks - 1 ? last_block : block_size;

            if (verbose)
                fprintf(stderr, "\n\tQUERYING WITH BLOCK (%d)... (size: %d, q_offset: %d)\n\n", j, bsize, q_offset);

            //######################################################################
            // If the last set from the query block is bigger than the
            // maxsize of the first set in the index, we can skip this block
            //######################################################################
            int q_size = (int) sets->len[(q_offset+bsize) - 1];
            int s_maxsize = floor(jac_max_size(sets->len[i_offset], threshold));
            if (q_size > s_maxsize) {
                if (verbose) {
                    fprintf(stderr, "\t\t => The last set from this block is bigger than\n");
                    fprintf(stderr, "\t\t    the maxsize of the first set from the index !\n");
                    fprintf(stderr, "\t\t    Skipping...\n");
                }
                continue;
            }
            //######################################################################

            // FILTERING -----------------------------------------------------------
            if (verbose) {
                fprintf(stderr, "\t\t => FILTERING... \n");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int csize;

            filtering_block_very_new (
                sets, inv_index, d_partial_scores,
                threshold, q_offset, i_offset, block_size, bsize,
                &csize, d_nres,
                verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
                fprintf(stderr, "\t\t    no of candidates: %d\n", csize);
            }
            //----------------------------------------------------------------------

            // CHECKING ------------------------------------------------------------
            if (verbose) {
                fprintf(stderr,"\n\t\t => CHECKING... ");
                t0 = ut_get_time_in_microseconds();
            }

            unsigned int *similar_pairs;
            unsigned short *scores;
            unsigned int size;

            checking_block_very_new (
                d_partial_scores, sets, threshold,
                q_offset, i_offset, block_size,
                &similar_pairs, &scores, &size,
                d_nres, verbose
            );

            if (verbose) {
                t1 = ut_get_time_in_microseconds();
                fprintf(stderr, "\t\t    DONE IN %g ms\n", ut_interval_in_miliseconds(t0,t1));
            }
            //----------------------------------------------------------------------

            candidates_count += csize;
            candidates_checked_count += size;

            t0 = ut_get_time_in_microseconds();
            for (unsigned k = 0; k < size; k++) {
                unsigned int   bucket = similar_pairs[k];
                unsigned int   query  = (bucket / block_size) + q_offset;
                unsigned int   source = (bucket % block_size) + i_offset;
                unsigned short score  = scores[k];

                if (score > 0) {
                    float similarity = jac_similarity(sets->len[query], sets->len[source], score);
                    if (similarity >= threshold)  {
                        fprintf(stdout, "%d %d %f\n", sets->id[query], sets->id[source], similarity);
                        sim_pairs_count++;
                    }
                }
            }
            t1 = ut_get_time_in_microseconds();
            t += ut_interval_in_miliseconds(t0,t1);

            cudaFreeHost(similar_pairs);
            cudaFreeHost(scores);
        }

        if (verbose)
            ut_print_separator("-", 99);
    }

    fprintf(stderr, "# Candidates: %lu\n", candidates_count);
    fprintf(stderr, "# Intersections: %lu\n", candidates_checked_count);
    fprintf(stderr, "# Results: %lu\n", sim_pairs_count);
    fprintf(stderr, "  - Time rendering: %g ms\n", t);
}
