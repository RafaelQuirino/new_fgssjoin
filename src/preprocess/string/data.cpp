#include "io.hpp"
#include "util.hpp"
#include "data.hpp"

#include <iostream>
#include <algorithm>
#include <unordered_map>



/* DOCUMENTATION
 *
 */
vector<string>
dat_get_input_data (string file_path, int input_size)
{
    // if (input_size == 0) fprintf (stderr, "\tReading %s...\n", file_path.c_str());
	// else fprintf (stderr, "\tReading %d lines from %s...\n", input_size, file_path.c_str());
	// unsigned long t0 = ut_get_time_in_microseconds();

	vector<string> input;
	if (input_size == 0) input = io_getlines (file_path);
	else input = io_getnlines (file_path, input_size);

	// unsigned long t1 = ut_get_time_in_microseconds();
	// fprintf (stderr, "\t> Done in %gms.\n", ut_interval_in_miliseconds (t0,t1));

    return input;
}



/* DOCUMENTATION
 * Auxiliar function
 */
char dat_easytolower(char in){
    if (in <= 'Z' && in >= 'A')
        return in - ('Z'-'z');
    return in;
}



/* DOCUMENTATION
 *
 */
vector<string>
dat_get_proc_data (vector<string>& input_data)
{
    // I hope this makes a copy...
    vector<string> proc_data = input_data;

    fprintf (stderr, "\tProcessing data...\n");
    unsigned long t0 = ut_get_time_in_microseconds();

    for (unsigned i = 0; i < input_data.size(); i++) {
        transform (
            proc_data[i].begin(), proc_data[i].end(), proc_data[i].begin(),
            dat_easytolower
        );
    }

    unsigned long t1 = ut_get_time_in_microseconds();
    fprintf (stderr, "\t> Done in %gms.\n", ut_interval_in_miliseconds (t0,t1));

    return proc_data;
}



/* DOCUMENTATION
 *
 */
void
dat_proc_data (vector<string>& input_data)
{

    // fprintf (stderr, "\tProcessing data...\n");
    // unsigned long t0 = ut_get_time_in_microseconds();

    for (unsigned i = 0; i < input_data.size(); i++) {
        transform (
            input_data[i].begin(), input_data[i].end(), input_data[i].begin(),
            dat_easytolower
        );
    }

    // unsigned long t1 = ut_get_time_in_microseconds();
    // fprintf (stderr, "\t> Done in %gms.\n", ut_interval_in_miliseconds (t0,t1));
}



/* DOCUMENTATION
 *
 */
// void 
// writeOutputFile (
//     unsigned int* pos, unsigned int* len, unsigned int* tokens, 
//     unsigned int n_sets, unsigned int n_tokens, unsigned int n_terms, 
//     int input_size, int qgram, string path, const char* outdir
// ) 
// {
//     char fname[128];
//     vector<string> v = split(path, '/');
//     const char* oldname = split(v[v.size()-1], '.')[0].c_str();
//     if (input_size == 0 || input_size >= n_sets)
//         input_size = n_sets;

//     char* slash = outdir[strlen(outdir)-1] == '/' ? (char*)"" : (char*)"/";
//     sprintf(fname, "%s%s%s_%d_%d.sets", outdir, slash, oldname, input_size, qgram);
//     //sprintf(fname, "%s_%d_%d.sets", oldname, input_size, qgram);

//     printf("Writing file: %s\n", fname);

//     FILE *fp;
//     fp = fopen(fname, "wb");

//     unsigned int sizes[3] = {n_sets, n_tokens, n_terms};
//     fwrite(sizes, sizeof(unsigned int), 3, fp);
//     fwrite(pos, sizeof(unsigned int), n_sets, fp);
//     fwrite(len, sizeof(unsigned int), n_sets, fp);
//     fwrite(tokens, sizeof(unsigned int), n_tokens, fp);

//     fclose(fp);
// }