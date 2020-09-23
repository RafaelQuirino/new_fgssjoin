#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/fgssjoin/data.cuh"
#include "../include/fgssjoin/block.cuh"
#include "../include/fgssjoin/index.cuh"
#include "../include/util.hpp"



int main (int argc, char** argv)
{
    const char* filepath;
    float       threshold;
    char        verbose = 0;
    char        debug = 0;

    if (argc < 3) {
        fprintf(stdout, "Usage:\n./%s <ppjoin bin file> <threshold> [verbose]\n", argv[0]);
        exit(1);
    }

    filepath = argv[1];
    if (access(filepath, F_OK) == -1) {
        fprintf(stderr, "File \"%s\" does not exist.\n", filepath);
        exit(1);
    }

    threshold = atof(argv[2]);
    if (threshold <= 0.0f || threshold >= 1.0) {
        fprintf(stderr, "Threshold must be a float in the range (0.0..1.0).\n");
        exit(1);		
    }

    if (argc == 4){
        if (strcmp(argv[3], "verbose") == 0)
            verbose = 1;
        if (strcmp(argv[3], "debug") == 0)
            debug = 1;
    }

    if (argc == 5){
        if (strcmp(argv[4], "verbose") == 0)
            verbose = 1;
        if (strcmp(argv[4], "debug") == 0)
            debug = 1;
    }


    unsigned long t0, t1, t00, t01;


    fprintf(stderr, "Document: %s\n", filepath);
    fprintf(stderr, "Algorithm: %s\n", "fgssjoin");
    fprintf(stderr, "Threshold: Jaccard %g\n", threshold);


    fprintf(stderr, "... LOADING DATASET ...\n");
    sets_t* sets = ppjoin_format(filepath, verbose);
    fprintf(stderr, "# Records: %zu\n", sets->num_sets);
    fprintf(stderr, "# Terms: %zu\n", sets->num_terms);
    fprintf(stderr, "# Tokens: %zu\n", sets->num_tokens);
    fprintf(stderr, "# Average size: %g\n", sets->average_size);
    if (debug) {
        for (int i = 0; i < sets->num_sets; i++) {
            fprintf(stdout, "id: [%u]\n", sets->id[i]);
            fprintf(stdout, "size: [%u]\n", sets->len[i]);
            for (int j = 0; j < sets->len[i]; j++)
                fprintf(stdout, "(%u) ", sets->tokens[sets->pos[i] + j]);
            fprintf(stdout, "\n\n");
        }
    }


    // Run fgssjoin
    fprintf(stderr, "=== BEGIN JOIN (TIMER STARTED) ===\n");
    t0 = ut_get_time_in_microseconds();

    cudaSetDevice(0);

    // Prepare data and transfer to gpu memory
    fprintf(stderr, "# Preparing and transfering...\n");
    t00 = ut_get_time_in_microseconds();
    prepare_data(sets, threshold, verbose);
    t01 = ut_get_time_in_microseconds();
    fprintf(stderr, "  - Time spent: %g ms\n", ut_interval_in_miliseconds(t00,t01));

    // Build inverted index
    Index index = inverted_index(sets, threshold, verbose);
    fprintf(stderr, "# Inverted Index Entries: %d\n", index.num_entries);
    if (debug)
        print_inverted_index(index, sets);

    // Process filtering/checking block pairs
    // process_blocks(sets, index, threshold, verbose);
    // process_blocks_int(sets, index, threshold, verbose);
    process_blocks_very_new(sets, index, threshold, verbose);
	
    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "===  END JOIN (TIMER STOPPED)  ===\n");
    fprintf(stderr, "Total Running Time: %g s\n", ut_interval_in_miliseconds(t0,t1)/1000.0);


    return 0;
}
