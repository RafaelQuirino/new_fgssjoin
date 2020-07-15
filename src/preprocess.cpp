#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "io.hpp"
#include "util.hpp"
#include "data.hpp"
#include "qgram.hpp"
#include "token.hpp"

int main (int argc, char** argv)
{
	if (argc != 2) {
        printf("Missing parameter.\n");
        exit(0);
    }

    // CONSTANTS
    string file_path(argv[1]);
    int input_size = 0;
    int qgram_size = 2;

    // VARIABLES
    unsigned long                        n_sets;
    unsigned long                        n_terms;
    unsigned long                        n_tokens;
    vector<string>                       raw_data;
    vector<string>                       proc_data;
    vector< vector<string> >             docs;
    unordered_map<unsigned long,token_t> dict; 

    // MEASURING PERFORMANCE (in multiple leves)
    unsigned long t0, t1, t00, t01;
    double total_time;

    fprintf (stderr, "+----------------+\n");
    fprintf (stderr, "| PRE-PROCESSING |\n");
    fprintf (stderr, "+----------------+\n\n");

    t0 = ut_get_time_in_microseconds();

    // READING DATA AND CREATING RECORDS ---------------------------------------
    fprintf (stderr, "Reading data and creating records...\n");
    t00 = ut_get_time_in_microseconds();

    // READING INPUT DATA
    raw_data = dat_get_input_data (file_path, input_size);

    printf("\traw_data size: %zu\n", raw_data.size());

    // PROCESSING DATA
    proc_data = dat_get_proc_data (raw_data);

    // CREATING QGRAM RECORDS
    docs = qg_get_records (proc_data, qgram_size);

    t01 = ut_get_time_in_microseconds();
    fprintf(stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds(t00, t01));
    //--------------------------------------------------------------------------

    // qg_print_records(docs);

    // CREATING TOKEN DICTIONARY -----------------------------------------------
    fprintf(stderr, "Creating token dictionary...\n");
    t00 = ut_get_time_in_microseconds();

    dict = tk_get_dict (docs);

    n_sets = docs.size();
    n_terms = dict.size();
    n_tokens = 0;
    for (int i = 0; i < docs.size(); i++)
    {
        n_tokens += docs[i].size();
    }

    t01 = ut_get_time_in_microseconds();
    fprintf(stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds(t00, t01));

    fprintf(stderr, "n_sets  : %lu\n", n_sets);
    fprintf(stderr, "n_terms : %lu\n", n_terms);
    fprintf(stderr, "n_tokens: %lu\n\n", n_tokens);
    //--------------------------------------------------------------------------

    // tk_print_dict(dict);

    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, 
        "Pre-processing %s took %gms.\n\n", 
        file_path.c_str(), 
        ut_interval_in_miliseconds(t0, t1)
    );

	return 0;
}
