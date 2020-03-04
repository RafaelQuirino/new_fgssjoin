/* DOCUMENTATION

*/



#ifndef _SORT_H_
#define _SORT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#include "token.hpp"



using namespace std;



/* DOCUMENTATION
 *
 */
int partition_1 (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    int l, int r
);

void quicksort_aux_1 (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    int l, int r
);

void quicksort_1 (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    unsigned int n
);

// This one is meant to be called...
void sort_vec_idx_by_size (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    unsigned int n
);



/* DOCUMENTATION
 *
 */
int partition_2 (token_t* arr,int l, int r);

void quicksort_aux_2 (token_t* arr, int l, int r);

void quicksort_2 (token_t* arr, unsigned int n);

void sort_tokens_by_freq (token_t* arr, unsigned int n);



#endif // _SORT_H_
