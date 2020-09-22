#ifndef _FGSSJOIN_DATA_H_
#define _FGSSJOIN_DATA_H_

#include <stdio.h>
#include <stdlib.h>



/*
 *  DOCUMENTATION
 */
typedef struct 
{
	size_t num_sets;            /* Number of sets */
	size_t num_terms;           /* Number of terms (unique tokens) */
	size_t num_tokens;          /* Number of tokens */
	size_t num_midpref_tokens;  /* Number of midprefix tokens */

	// +---------------+
	// | Complete sets |
	// +---------------+

	// Cpu pointers
	unsigned int *id;           /* Record id */
	unsigned int *pos;          /* Starting position of each set */
	unsigned int *len;          /* Size of each set */
	unsigned int *tokens;       /* All sets concatenated */
	// Gpu pointers
	unsigned int *d_id;
	unsigned int *d_pos;
	unsigned int *d_len;
	unsigned int *d_tokens;

	// +------------------+
	// | Midprefix tokens |
	// +------------------+

	// Cpu pointers
	unsigned int* midpref_pos;
	unsigned int* midpref_len;
	unsigned int* midpref_tokens;
	// Gpu pointers
	unsigned int* d_midpref_pos;
	unsigned int* d_midpref_len;
	unsigned int* d_midpref_tokens;

} sets_t;



/*
 *  DOCUMENTATION
 */
__host__
sets_t* sets_new ();



/*
 *  DOCUMENTATION
 */
__host__
sets_t* ppjoin_format (const char* filepath);



/*
 *  DOCUMENTATION
 */
__host__
void prepare_data (sets_t* sets, float threshold);



/*
 *  DOCUMENTATION
 */
__host__
void select_and_index_midpref_tokens (
    unsigned int* d_tokens, unsigned int* d_pos, unsigned int* d_len, 
    unsigned int* d_midpref_len,
    unsigned int n_sets, unsigned int n_tokens, float threshold,
    unsigned int** d_select, unsigned int** d_index, unsigned int* select_size_out
);



/*
 *  DOCUMENTATION
 */
__global__
void select_midpref_tokens_kernel (
	unsigned int* bin, unsigned int* pos, unsigned int* len, unsigned int* midpref_len,
    float threshold, unsigned int n_sets
);



/*
 *  DOCUMENTATION
 */
__global__
void compact_midpref_tokens_kernel (
	unsigned int* midpref_tokens, unsigned int* tokens,
    unsigned int* select, unsigned int* index, unsigned int n_tokens
);



#endif /* _FGSSJOIN_DATA_H_ */
