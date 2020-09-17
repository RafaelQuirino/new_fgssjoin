#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <unordered_map>

#include "preprocess/string/io.hpp"
#include "preprocess/string/util.hpp"
#include "preprocess/string/data.hpp"
#include "preprocess/string/qgram.hpp"



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



const char* PREPROCESS_SELFJOIN    = "selfjoin"; 
const char* PREPROCESS_DUALJOIN    = "dualjoin";
const int   PREPROCESS_QGRAM_LIMIT = 1024;



typedef struct {
    const char*  cmd;
    const char*  only_file;
    const char*  first_file;
    const char*  second_file;
    unsigned int qgram;
} cli_args_t;



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



int main (int argc, char** argv)
{
    cli_args_t* args = get_cli_args(argc, argv);
    print_cli_args(args);

	return 0;
}
