/* 
 * DOCUMENTATION
 */



#ifndef _PREPROCESS_H_
#define _PREPROCESS_H_

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;



/*
 *  Documentation
 */
vector<string> 
pp_get_data (string filepath);



/*
 *  Documentation
 */
void 
pp_proc_data (vector<string>& data);



/*
 *  Documentation
 */
vector< vector<string> > 
pp_get_qgrams (vector<string>& data, int qgramsize);



/*
 *  Documentation
 */
vector< vector<unsigned long> > 
pp_calc_hashes (vector< vector<string> >& qgrams);



/*
 *  Documentation
 */
unordered_map<unsigned long, unsigned int>
pp_get_dict (vector< vector<unsigned long> >& hashes);



/*
 *  Documentation
 */
vector< vector<unsigned int> >
pp_get_sets (
	vector< vector<unsigned long> >& hashes,
	unordered_map<unsigned long, unsigned int>& dict
);



#endif // _PREPROCESS_H_
