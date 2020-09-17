/* DOCUMENTATION

*/



#ifndef _TOKEN_H_
#define _TOKEN_H_



#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>



#define QGRAM     0
#define HASH      1
#define FREQ      2
#define DOC_ID    3
#define ORDER_ID  4
#define POS       5



using namespace std;



/* DOCUMENTATION
 * 
 */
typedef struct token_t
{
	string        qgram; // Token qgram.
	unsigned long hash;  // Qgram hash code.
	unsigned int  freq;  // Frequency among records.

	int doc_id;   // Except when in dictionary (came from no specific record).
	              // In this case, set to -1.

	int order_id; // Id in frequency order.
	int orig_pos; // Original position in the set before sorting.
	int pos;      // Position in the set after sorting in frequency order.

} token_t;



/* DOCUMENTATION
 * TODO
 */
unordered_map<unsigned long,token_t>
tk_get_dict (vector< vector<string> > records);



/* DOCUMENTATION
 * TODO
 */ 
vector< vector<token_t> > // Token sets
tk_get_tokensets (
	unordered_map<unsigned long,token_t> dict,
	vector< vector<string> > recs,
	vector<size_t>& index
);



/*
 * Sort the set of sets by set sizes.
 * Returns an index array to the sets original positions before sorting
 * (index to original sets positions)
 */
// unsigned int*
vector<size_t>
tk_sort_sets (vector< vector<token_t> >& tsets);



/* DOCUMENTATION
 * TODO
 */
void tk_sort_each_set (vector< vector<token_t> >& tsets);



/*
 * Converts the tokensets data into sets of
 * one-dimensional arrays for use in the GPU
 */
unsigned int* tk_convert_tokensets (
	vector< vector<token_t> > tsets, int num_tokens,
	unsigned int** pos_out, unsigned int** len_out
);



/*
 * Auxiliar functions for common procedures and debugging
 */
void tk_print_token (token_t tkn);
void tk_print_tset  (vector<token_t> tset, int field);
void tk_print_tsets (vector< vector<token_t> > tsets, int field);
void tk_print_dict  (unordered_map<unsigned long,token_t> dict);



#endif // _TOKEN_H_
