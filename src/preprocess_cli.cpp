#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

#include "io.hpp"
#include "util.hpp"
#include "qgram.hpp"
#include "preprocess.hpp"

using namespace std;



/*
 *  DOCUMENTATION
 */
const char* PREPROCESS_USAGE = "\
\e[1mfgssjoin\033[0m's string preprocess engine.\n\
Usage:\n\
    preprocess selfjoin <only_file> <q_gram>\n\
    preprocess dualjoin <1st_file> <2nd_file> <q_gram>\n\
Arguments:\n\
    <only_file>    Single file with string data, one for each line (for selfjoin)\n\
    <1st_file>     First file with string data (for dualjoin)\n\
    <2nd_file>     Second file with string data (for dualjoin)\n\
    <q_gram>       Qgram size (positive integer)\n\
";



/*
 *  DOCUMENTATION
 */
const char* PREPROCESS_SELFJOIN    = "selfjoin"; 
const char* PREPROCESS_DUALJOIN    = "dualjoin";
const int   PREPROCESS_QGRAM_LIMIT = 1024;



/*
 *  DOCUMENTATION
 */
typedef struct {
    const char*  cmd;
    const char*  only_file;
    const char*  first_file;
    const char*  second_file;
    unsigned int qgram;
} cli_args_t;



/*
 *  DOCUMENTATION
 */
cli_args_t* get_cli_args (int argc, char** argv)
{
    cli_args_t* args = (cli_args_t*) malloc(sizeof(cli_args_t));

    char* fname;

    if (argc < 2) {
        fprintf(stdout, "%s", PREPROCESS_USAGE);
        exit(0);
    }

    if (strcmp(argv[1], PREPROCESS_SELFJOIN) == 0) {
        args->cmd = PREPROCESS_SELFJOIN;
        if (argc < 4) {
            fprintf(stderr, "Missing arguments.\n\n%s", PREPROCESS_USAGE);
            exit(0);
        } else {
            fname = argv[2];
            if (access(fname, F_OK) != -1) {
                args->only_file = fname;
            } else {
                fprintf(stderr, "File \"%s\" does not exist.\n", fname);
                exit(0);
            }
            args->qgram = atoi(argv[3]);
            if (args->qgram < 1 || args->qgram > PREPROCESS_QGRAM_LIMIT) {
                fprintf(stderr, "Bad \"qgram\" value. Must be between 1 and %d.\n", PREPROCESS_QGRAM_LIMIT);
                exit(0);
            }
        }
    }

    else if (strcmp(argv[1], PREPROCESS_DUALJOIN) == 0) {
        args->cmd = PREPROCESS_DUALJOIN;
        if (argc < 5) {
            fprintf(stderr, "Missing arguments.\n\n%s", PREPROCESS_USAGE);
            exit(0);
        } else {
            fname = argv[2];
            if (access(fname, F_OK) != -1) {
                args->first_file = fname;
            } else {
                fprintf(stderr, "File \"%s\" does not exist.\n", fname);
                exit(0);
            }
            fname = argv[3];
            if (access(fname, F_OK) != -1) {
                args->second_file = fname;
            } else {
                fprintf(stderr, "File \"%s\" does not exist.\n", fname);
                exit(0);
            }
            args->qgram = atoi(argv[4]);
            if (args->qgram < 1 || args->qgram > PREPROCESS_QGRAM_LIMIT) {
                fprintf(stderr, "Bad \"qgram\" value. Must be between 1 and %d.\n", PREPROCESS_QGRAM_LIMIT);
                exit(0);
            }
        }
    }

    else {
        fprintf(stderr, "%s", PREPROCESS_USAGE);
        exit(0);
    }

    return args;
}



/*
 *  DOCUMENTATION
 */
void print_cli_args (cli_args_t* args)
{
    fprintf(stdout, "command: %s\n", args->cmd);
    if (strcmp(args->cmd, PREPROCESS_SELFJOIN) == 0)
        fprintf(stdout, "only_file: %s\n", args->only_file);
    else {
        fprintf(stdout, "first_file: %s\n", args->first_file);
        fprintf(stdout, "second_file: %s\n", args->second_file);
    }
    fprintf(stdout, "qgram: %d\n", args->qgram);
}



/*
 *  DOCUMENTATION
 */
template <typename T>
vector<T> flatten(const vector<vector<T>>& v) 
    {
    size_t total_size = 0;
    for (const auto& sub : v)
        total_size += sub.size(); // I wish there was a transform_accumulate
    vector<T> result;
    result.reserve(total_size);
    for (const auto& sub : v)
        result.insert(result.end(), sub.begin(), sub.end());
    return result;
}



/*
 *  DOCUMENTATION
 */
template <typename T>
vector<size_t> sort_index (vector<T> &v) 
{
    // initialize original index locations
    vector<size_t> idx(v.size());
    int x = 0;
    iota(idx.begin(), idx.end(), x++);

    stable_sort(idx.begin(), idx.end(), [&](int i,int j) { return v[i] < v[j]; });

    // return the sorted index
    // (an index to the original vector)
    return idx;
}



/*
 *  DOCUMENTATION
 */
template <typename T>
vector<size_t> sort_with_index (vector<T> &v) 
{
    // initialize original index locations
    vector<size_t> idx(v.size());
    int x = 0;
    iota(idx.begin(), idx.end(), x++);

    stable_sort(idx.begin(), idx.end(), [&](int i,int j) { return v[i] < v[j]; });
    stable_sort(v.begin(), v.end());

    // return the sorted index
    // (an index to the original vector)
    return idx;
}



/*
 *  DOCUMENTATION
 */
template <typename T>
void sort_based_on_first (vector<T> &v_1st, vector<T> &v_2nd) 
{
    stable_sort(v_2nd.begin(), v_2nd.end(), [&](int i,int j) { return v_1st[i] < v_1st[j]; });
    stable_sort(v_1st.begin(), v_1st.end());
}



/*
 *  DOCUMENTATION
 */
int main (int argc, char** argv)
{
    cli_args_t* args = get_cli_args(argc, argv);
    // print_cli_args(args);

    string filepath1, filepath2;
    if (strcmp(args->cmd, PREPROCESS_SELFJOIN) == 0) {
        filepath1 = args->only_file;
    } else {
        filepath1 = args->first_file;
        filepath2 = args->second_file;        
    }
    // cout << filepath1 << ", " << filepath2 << endl;

    // Doc
    vector<string> data = pp_get_data(filepath1);
    // ut_print_str_vec(data);

    // Doc
    pp_proc_data(data);
    // ut_print_str_vec(data);

    // Doc
    vector< vector<string> > qgrams = pp_get_qgrams(data, 3);
    data.clear();
    // qg_print_records(qgrams);

    // Doc
    vector< vector<unsigned long> > hashes = pp_calc_hashes(qgrams);
    qgrams.clear();
    // qg_print_sets(hashes);

    // Doc
    vector<unsigned int> len(hashes.size());
    for (int i = 0; i < hashes.size(); i++)
        len[i] = hashes[i].size();
    vector<unsigned int> pos(hashes.size(), 0);
    for (int i = 1; i < hashes.size(); i++)
        pos[i] = pos[i-1] + len[i-1];

    //-------------------------------------------------------------------------
    // The real fun
    //-------------------------------------------------------------------------
    // Create T by flattening hashes
    vector<unsigned long> T = flatten(hashes);
    hashes.clear();
    // // Testing
    // cout << "T:\n";
    // for (int i = 0; i < T.size(); i++)
    //     cout << "(" << T[i] << ") ";
    // cout << endl << endl;

    // Creating T' and Ti'
    vector<unsigned long> T_prime = T;
    vector<size_t> Ti_prime = sort_with_index(T_prime);
    size_t N = T_prime.size();
    // // Testing
    // cout << "T_prime:\n";
    // for (int i = 0; i < N; i++)
    //     cout << "(" << T_prime[i] << ") ";
    // cout << endl << endl << "Ti_prime:\n";
    // for (int i = 0; i < N; i++)
    //     cout << "(" << Ti_prime[i] << ") ";
    // cout << endl << endl;

    // Now Dt and Df 
    // (actually i don't need Dt...)
    vector<unsigned long> Dt; // * Comment this in production
    vector<unsigned long> Df;
    unsigned long curritem  = T_prime[0];
    unsigned long currcount = -1;
    for (int i = 0; i < N; i++) {
        unsigned long item = T_prime[i];

        currcount += 1;

        if (item != curritem || i == N-1) {
            Dt.push_back(curritem); // * Comment this in production
            Df.push_back(currcount);
            curritem = item;
            currcount = 0;
        }
    }
    // // Testing
    // for (int i = 0; i < Dt.size(); i++) {
    //     cout << Dt[i] << ": " << Df[i] << " [" << i << "]" << endl;
    // }
    cout << endl;
    T_prime.clear();
    Dt.clear(); // * Comment this in production

    // Now Dt', Df', Di' and Idx
    // (well, actually only Di' and Idx)
    vector<size_t> Di_prime = sort_index(Df);
    int N_dict = Df.size();
    vector<size_t> Idx(N_dict);
    Idx[0] = 0;
    for (int i = 1; i < N_dict; i++)
        Idx[i] = Idx[i-1] + Df[i-1];
    // // Testing
    // for (int i = 0; i < Di_prime.size(); i++)
    //     cout << "[" << Di_prime[i] << "]";
    // cout << endl << endl;

    // And then, Tid
    vector<unsigned int> Tid(N_dict);
    for (int i = 0; i < N_dict; i++)
        Tid[Di_prime[i]] = i;

    // // Testing
    // for (int i = 0; i < N; i++)
    //     cout << "(" << T[i] << ") ";
    // cout << endl << endl;
    // Finally, assign Tid's for every token in T
    for (int i = 0; i < N_dict; i++) {
        for (int j = Idx[i]; j <= Idx[i]+Df[i]; j++) {
            T[Ti_prime[j]] = Tid[i];
        }
    }
    // // Testing
    // for (int i = 0; i < N; i++)
    //     cout << "(" << T[i] << ") ";
    // cout << endl << endl;
    //-------------------------------------------------------------------------

	return 0;
}
