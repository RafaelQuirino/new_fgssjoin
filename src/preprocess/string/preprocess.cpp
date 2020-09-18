#include "preprocess.hpp"
#include "data.hpp"
#include "qgram.hpp"

using namespace std;



/*
 *  Documentation
 */
vector<string> 
pp_get_data (string filepath)
{
    return dat_get_input_data(filepath, 0);
}



/*
 *  Documentation
 */
void 
pp_proc_data (vector<string>& data)
{
    dat_proc_data(data);
}



/*
 *  Documentation
 */
vector< vector<string> > 
pp_get_qgrams (vector<string>& data, int qgramsize)
{
    return qg_get_records(data, qgramsize);
}



/*
 *  Documentation
 */
vector< vector<unsigned long> > 
pp_calc_hashes (vector< vector<string> >& qgrams)
{
    return qg_get_sets(qgrams);
}



// /*
//  *  Documentation
//  */
// unordered_map<unsigned long, unsigned int>
// pp_get_dict (vector< vector<unsigned long> >& hashes)
// {
//     return NULL;
// }



// /*
//  *  Documentation
//  */
// vector< vector<unsigned int> >
// pp_get_sets (
//     vector< vector<unsigned long> >& hashes,
//     unordered_map<unsigned long, unsigned int>& dict
// )
// {
//     return NULL;
// }
