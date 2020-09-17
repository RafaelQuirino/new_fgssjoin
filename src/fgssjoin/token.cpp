#include <string.h>  // strlen
#include <numeric>   // iota
#include <algorithm> // sort

#include "util.hpp"
#include "sort.hpp"
#include "qgram.hpp"
#include "token.hpp"



// Comparing functions (for sorting) ------------------------------------------
bool tk_compare_freq (token_t t1, token_t t2)
{
	return t1.freq < t2.freq;
}

bool tk_compare_qgram (token_t t1, token_t t2)
{
	return strcmp (t1.qgram.c_str(), t2.qgram.c_str()) < 0;
}

bool compare_order (token_t a, token_t b)
{
	return a.order_id < b.order_id;
}

bool compare_size (vector<token_t> s1, vector<token_t> s2)
{
	return s1.size() > s2.size();
}
//-----------------------------------------------------------------------------



/* DOCUMENTATION
 *
 */
unordered_map<unsigned long,token_t>
tk_get_dict (vector< vector<string> > records)
{
	unordered_map<unsigned long,token_t> dict;

	for (int i = 0; i < records.size(); i++)
	{
		for (int j = 0; j < records[i].size(); j++)
		{
			string qgram = records[i][j];
			unsigned long hcode = qg_hash(qgram);

			// First check if token is already in dictionary.
            // If not, create it. If it is, increment frequency.
            unordered_map<unsigned long,token_t>::const_iterator result;
			result = dict.find(hcode);

			// If entry not found in dictionary
            if (result == dict.end()) {
                token_t newtkn;
				newtkn.qgram    = qgram;
				newtkn.hash     = hcode;
                newtkn.freq     = 1;
                newtkn.doc_id   = -1;
                newtkn.order_id = -1;
                dict[hcode]     = newtkn;
            }
            else {
                dict[hcode].freq += 1;
            }
		}
	}

	vector<token_t> tkns;
	for (auto& it: dict) {
    	tkns.push_back(it.second);
	}

	sort(tkns.begin(), tkns.end(), tk_compare_freq);
	
	for (int i = 0; i < tkns.size(); i++) {
		dict[tkns[i].hash].order_id = i;
	}

	return dict;
}



/*
 * Build the token sets using the records and the dictionary. 
 * Returns the token sets.
 */
vector< vector<token_t> >
tk_get_tokensets (
	unordered_map<unsigned long,token_t> dict,
	vector< vector<string> >             records,
	vector<size_t>&                      index
)
{
    vector< vector<token_t> > tsets;
	unsigned long t0, t1;



	fprintf (stderr, "\tBuilding token sets...\n");
    t0 = ut_get_time_in_microseconds();

    for (unsigned int i = 0; i < records.size(); i++)
    {
		vector<string> record = records[i];
        vector<token_t> set;

        for (unsigned int j = 0; j < record.size(); j++)
        {
            string qgram = record[j];
			unsigned long hcode = qg_hash(qgram);

            unordered_map<unsigned long,token_t>::const_iterator result;
			result = dict.find(hcode);

            if (result == dict.end()) {
                fprintf(stderr, "Error in tk_get_tokensets, line %d. Token not in dictionary.\n", __LINE__);
            }
            else {
                token_t newtkn;
				newtkn.qgram    = qgram;
				newtkn.hash     = dict[hcode].hash;
                newtkn.freq     = dict[hcode].freq;
                newtkn.order_id = dict[hcode].order_id;

                set.push_back(newtkn);
            }

        } // for (int j = 0; j < recs[i].size(); j++)

        tsets.push_back(set);

    } // for (int i = 0; i < recs.size(); i++)

	t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "\t> Done. It took %gms.\n", ut_interval_in_miliseconds(t0, t1));

    fprintf (stderr, "\tSorting sets by size...\n");
    t0 = ut_get_time_in_microseconds();
    
    index = tk_sort_sets(tsets);
    
    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "\t> Done. It took %gms.\n", ut_interval_in_miliseconds(t0, t1));

    fprintf (stderr, "\tSorting each set by token frequency...\n");
    t0 = ut_get_time_in_microseconds();

    tk_sort_each_set(tsets);

    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "\t> Done. It took %gms.\n", ut_interval_in_miliseconds(t0, t1));

    return tsets;
}



/*
 * Auxiliar function to sort an array and its index
 */
template <typename T>
vector<size_t> tk_sort_indexes(const vector<T> &v) {

  // Initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // Sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1].size() > v[i2].size();});

  return idx;
}

/*
 * Returns an index array to find original positions
 * of records before sorting
 */
// unsigned int*
vector<size_t>
tk_sort_sets (vector< vector<token_t> >& tsets)
{
    unsigned int num_sets = tsets.size();

    // unsigned int* index;
    // index = (unsigned int*) malloc (num_sets * sizeof(unsigned int));
    // for (unsigned int i = 0; i < num_sets; i++)
    //     index[i] = i;

	// Sorting algorithm
	// sort_vec_idx_by_size(tsets, index, tsets.size());

	// cout << endl;
	// for (int i = 0; i < tsets.size(); i++) cout << "[ " << i << "]"; cout << endl;
	// for (int i = 0; i < tsets.size(); i++) cout << "[" << tsets[i].size() << "]"; cout << endl << endl;

	vector<size_t> idx = tk_sort_indexes(tsets);
	sort(tsets.begin(), tsets.end(), compare_size);

	// for (int i = 0; i < tsets.size(); i++) cout << "[ " << idx[i] << "]"; cout << endl;
	// for (int i = 0; i < tsets.size(); i++) cout << "[" << tsets[i].size() << "]"; cout << endl << endl;

    // Fix doc_id for all tokens now...
    for (unsigned i = 0; i < tsets.size(); i++) {
        for (unsigned j = 0; j < tsets[i].size(); j++) {
            tsets[i][j].doc_id = i;
        }
    }

    return idx;
}



/* DOCUMENTATION
 *
 */
void tk_sort_each_set (vector< vector<token_t> >& tsets)
{
    for (unsigned int i = 0; i < tsets.size(); i++)
        sort (tsets[i].begin(), tsets[i].end(), compare_order);
}



/* DOCUMENTATION
 * 
 */
unsigned int* tk_convert_tokensets (
	vector< vector<token_t> > tsets, int num_tokens,
	unsigned int** pos_out, unsigned int** len_out
)
{
	unsigned int n     = tsets.size();
	unsigned int* pos  = (unsigned int*) malloc(n * sizeof(unsigned int));
	unsigned int* len  = (unsigned int*) malloc(n * sizeof(unsigned int));
	unsigned int* sets = (unsigned int*) malloc(num_tokens * sizeof(unsigned int));

	for (unsigned int i = 0; i < n; i++)
		len[i] = tsets[i].size();

	pos[0] = 0;
	for (unsigned int i = 1; i < n; i++)
		pos[i] = pos[i-1] + len[i-1];

	int k = 0;
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < tsets[i].size(); j++) {
			sets[k] = tsets[i][j].order_id;
			k += 1;
		}
	}

	*pos_out = pos;
	*len_out = len;

	return sets;
}



/* DOCUMENTATION
 * 
 */
void tk_print_token (token_t tkn)
{
	printf("hash     : %lu\n", tkn.hash);
	printf("freq     : %u\n", tkn.freq);
	printf("doc_id   : %d\n", tkn.doc_id);
	printf("order_id : %d\n", tkn.order_id);
	printf("pos      : %d\n", tkn.pos);
	printf("------------------------------------------\n");
}



/* DOCUMENTATION
 * 
 */
void tk_print_tset (vector<token_t> tset, int field)
{
	printf("[");
	for (unsigned int i = 0; i < tset.size(); i++) {
		char c = i == tset.size()-1 ? ']' : ',';
		if (field == QGRAM)
			printf("%s%c", tset[i].qgram.c_str(), c);
		else if (field == HASH)
			printf("%lu%c", tset[i].hash, c);
		else if (field == FREQ)
			printf("%u%c", tset[i].freq, c);
		else if (field == DOC_ID)
			printf("%d%c", tset[i].doc_id, c);
		else if (field == ORDER_ID)
			printf("%d%c", tset[i].order_id, c);
		else if (field == POS)
			printf("%d%c", tset[i].pos, c);
	}
	printf("\n");
}



/* DOCUMENTATION
 * 
 */
void tk_print_tsets (vector< vector<token_t> > tsets, int field)
{
	for (unsigned int i = 0; i < tsets.size(); i++)
		tk_print_tset(tsets[i], field);
}



/* DOCUMENTATION
 *
 */
void tk_print_dict (unordered_map<unsigned long,token_t> dict)
{
	for (pair<unsigned long,token_t> element : dict)
    {
		cout << "(" << element.first << ") :: " << endl;
		cout << " \t qgram: ("   << element.second.qgram    << ")" << endl;
		cout << " \t hash: "     << element.second.hash     << endl;
		cout << " \t freq: "     << element.second.freq     << endl;
		cout << " \t doc_id: "   << element.second.doc_id   << endl;
		cout << " \t order_id: " << element.second.order_id << endl;
		cout << endl;
    }
}