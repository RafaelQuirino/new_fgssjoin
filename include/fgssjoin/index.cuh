#ifndef _INDEX_CUH_
#define _INDEX_CUH_

#include "util.cuh"
#include "data.cuh"



/*
 *  DOCUMENTATION
 */
struct Entry
{
	int set_id;
	int term_id;
	int pos;
	Entry (int set_id, int term_id, int pos) : 
    	set_id(set_id), term_id(term_id), pos(pos) 
    {}
};



/*
 *  DOCUMENTATION
 */
struct Index
{
    int *d_index;		//Indicates where each list ends in the inverted index (position after the end)
    int *d_count;		//Number of entries for a given term in the inverted index
    Entry *d_lists;		//Inverted index in gpu memory

    int num_sets;		//Number of sets
    int num_terms;		//Number of terms
    int num_entries;	//Number of entries

    Index (Entry *d_lists = NULL, 
        int *d_index = NULL, int *d_count = NULL, 
        int num_sets = 0, int num_terms = 0, int num_entries = 0) :
        
        d_lists(d_lists),
        d_index(d_index),
        d_count(d_count),
        num_sets(num_sets),
        num_terms(num_terms),
        num_entries(num_entries)
    {}
};



/*
 *  DOCUMENTATIONS
 */
__host__ 
Index new_inverted_index_block (
    sets_t* sets, float threshold, 
    unsigned int offset, unsigned int block_size
);



/*
 *  DOCUMENTATION
 */
__host__ 
Index inverted_index (sets_t* sets, float threshold, char verbose);



/*
 *  DOCUMENTATION
 */
__host__ 
void print_inverted_index (Index index, sets_t* sets);



/*
 *  DOCUMENTATION
 */
__host__
Entry* create_midpref_entries (
    sets_t* sets, unsigned int n_entries, 
    unsigned int offset, unsigned int block_size
);



/*
 *  DOCUMENTATION
 */
__global__
void midpref_kernel (
    Entry* entries, 
    unsigned int* midpref_pos, unsigned int* midpref_len, unsigned int* midpref_tokens, 
    unsigned int offset, unsigned int n_sets
);



/*
 *  DOCUMENTATION
 */
__global__ 
void df_count_kernel (Entry *entries, int *count, int n);



/*
 *  DOCUMENTATION
 */
__global__ 
void inverted_index_kernel(Entry *entries, Entry *lists, int *index, int n);



#endif /* _INDEX_CUH_ */
