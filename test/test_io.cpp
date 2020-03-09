#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/test_io.hpp"
#include "../include/io.hpp"

/*
 *
 */
TestSuite test_io_getsuite ()
{
    TestSuite suite = test_suite_new("IO", test_io_getfixtures());
    test_suite_push(suite, test_case_new("io_getlines",  test_io_getlines));
    test_suite_push(suite, test_case_new("io_getnlines", test_io_getnlines));

    return suite;
}

/*
 *
 */
TestFixtures test_io_getfixtures ()
{
    TestFixtures fixtures = test_fixtures_new(
        test_io_createfile, 
        test_io_destroyfile
    );

    int* num_lines = (int*) malloc(sizeof(int));
    (*num_lines) = 32;
    test_fixtures_push(fixtures, 
        test_fixture_new(
            "num_lines",
            (void*)num_lines
        )
    );
    
    test_fixtures_push(fixtures, 
        test_fixture_new(
            "file_name",
            (void*)"/tmp/test_io_file_1"
        )
    );
    
    test_fixtures_push(fixtures,
        test_fixture_new(
            "line_example_1", 
            (void*)"test_io_file_1"
        )
    );
    
    test_fixtures_push(fixtures,
        test_fixture_new(
            "line_example_2", 
            (void*)"test_io_file_1\n"
        )
    );

    return fixtures;
}

/*
 *
 */
void test_io_createfile (TestFixtures fixtures)
{
    int* num_lines = (int*) test_fixtures_findobj(fixtures, "num_lines");
    const char* file_name = (const char*) test_fixtures_findobj(fixtures, "file_name");
    const char* line_text = (const char*) test_fixtures_findobj(fixtures, "line_example_1");
    
    FILE* file = fopen(file_name, "w");
    for (int i = 0; i < *num_lines; i++)
    {
        fprintf(file, "%s\n", line_text);
    }
    fclose(file);
}

/*
 *
 */
void test_io_destroyfile (TestFixtures fixtures)
{
    const char* file_name = (const char*) test_fixtures_findobj(fixtures, "file_name");

    char cmd[128];
    sprintf(cmd, "rm %s", file_name);
    system(cmd);
}

/*
 *
 */
int test_io_getlines (TestFixtures fixtures)
{
    int result = TEST_PASS;

    int* num_lines = (int*) test_fixtures_findobj(fixtures, "num_lines");
    const char* file_name = (const char*) test_fixtures_findobj(fixtures, "file_name");
    const char* line_1    = (const char*) test_fixtures_findobj(fixtures, "line_example_1");
    const char* line_2    = (const char*) test_fixtures_findobj(fixtures, "line_example_2");

    string filename(file_name);
    vector<string> lines = io_getlines(filename);
    if (
        lines.size() != *num_lines ||
        strcmp(lines[0].c_str(), line_1) != 0 ||
        strcmp(lines[0].c_str(), line_2) == 0
    )
    {
        result = TEST_FAIL; 
    }
    
    return result;
}

/*
 *
 */
int test_io_getnlines (TestFixtures fixtures)
{
    int result = TEST_PASS;

    int* num_lines = (int*) test_fixtures_findobj(fixtures, "num_lines");
    const char* file_name = (const char*) test_fixtures_findobj(fixtures, "file_name");
    const char* line_1    = (const char*) test_fixtures_findobj(fixtures, "line_example_1");
    const char* line_2    = (const char*) test_fixtures_findobj(fixtures, "line_example_2");

    int NLINES   = 16;
    string filename(file_name);
    vector<string> lines = io_getnlines(filename, NLINES);
    if (
        lines.size() != NLINES ||
        strcmp(lines[0].c_str(), line_1) != 0 ||
        strcmp(lines[0].c_str(), line_2) == 0
    )
    {
        result = TEST_FAIL; 
    }

    return result;
}
