#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/fgssjoin/string/data.cuh"
#include "../include/fgssjoin/block.cuh"
#include "../include/fgssjoin/index.cuh"
#include "util.hpp"



int main (int argc, char** argv)
{
	const char* filepath;
	float       threshold;

	if (argc < 3) {
		fprintf(stdout, "Usage:\n./%s <ppjoin bin file> <threshold>\n", argv[0]);
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


	unsigned long t0, t1;


	sets_t* sets = ppjoin_format(filepath);
	// for (int i = 0; i < sets->num_sets; i++) {
	// 	fprintf(stdout, "id: [%u]\n", sets->id[i]);
	// 	fprintf(stdout, "size: [%u]\n", sets->len[i]);
	// 	for (int j = 0; j < sets->len[i]; j++)
	// 		fprintf(stdout, "(%u) ", sets->tokens[sets->pos[i] + j]);
	// 	fprintf(stdout, "\n\n");
	// }


	// Prepare data and transfer to gpu memory
	prepare_data(sets, threshold);


	// Build inverted index
	Index index = inverted_index(sets, threshold);
	// print_inverted_index(index, sets);


	// Run fgssjoin
	t0 = ut_get_time_in_microseconds();
	process_blocks(sets, index, threshold); //, block_size);
	t1 = ut_get_time_in_microseconds();
	fprintf(stderr, "\nAlgorithm time: %g ms\n", ut_interval_in_miliseconds(t0,t1));


	return 0;
}
