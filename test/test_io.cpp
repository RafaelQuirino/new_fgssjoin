#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/test_io.hpp"
#include "../include/io.hpp"

/*
 *
 */
test_suite_t* test_io_getsuite ()
{
    test_suite_t* suite = test_suite_new("IO");
    test_suite_push(suite, test_case_new("io_getlines",  test_io_getlines));
    test_suite_push(suite, test_case_new("io_getnlines", test_io_getnlines));

    return suite;
}



/*
 *
 */
int test_io_getlines ()
{
    int result = TEST_PASS;

    // SETTING FIXTURES
    int NUMLINES = 32;
    FILE* file_1 = fopen("/tmp/test_io_file_1", "w");
    for (int i = 0; i < NUMLINES; i++)
    {
        fprintf(file_1,"test_io_file_1\n");
    }
    fclose(file_1);

    // TESTING FUNCTION
    string filename("/tmp/test_io_file_1");
    vector<string> lines = io_getlines(filename);
    if (
        lines.size() != NUMLINES ||
        strcmp(lines[0].c_str(),"test_io_file_1") != 0 ||
        strcmp(lines[0].c_str(),"test_io_file_1\n") == 0
    )
    {
        result = TEST_FAIL; 
    }

    // CLEANING FIXTURES
    system("rm /tmp/test_io_file_1");
    
    return result;
}



/*
 *
 */
int test_io_getnlines ()
{
    int result = TEST_PASS;

    // SETTING FIXTURES
    int NLINES   = 16;
    int NUMLINES = 32;
    FILE* file_1 = fopen("/tmp/test_io_file_1", "w");
    for (int i = 0; i < NUMLINES; i++)
    {
        fprintf(file_1,"test_io_file_1\n");
    }
    fclose(file_1);

    // TESTING FUNCTION
    string filename("/tmp/test_io_file_1");
    vector<string> lines = io_getnlines(filename, NLINES);
    if (
        lines.size() != NLINES ||
        strcmp(lines[0].c_str(),"test_io_file_1") != 0 ||
        strcmp(lines[0].c_str(),"test_io_file_1\n") == 0
    )
    {
        result = TEST_FAIL; 
    }

    // CLEANING FIXTURES
    system("rm /tmp/test_io_file_1");
    
    return result;
}
