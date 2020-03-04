/* +---------------------------------------------------------------------------+
 * | TEST FRAMEWORK
 * | --------------
 * |
 * | Test framework source file.
 * |
 * +---------------------------------------------------------------------------+
 */



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "include/test_framework.hpp"



//=============================================================================
// TEST FIXTURE FUNCTIONS
//=============================================================================

//=============================================================================





//=============================================================================
// TEST UTIL FUNCTIONS
//=============================================================================

/*
 *
 */
char* test_get_result (int result)
{
    char* str = (char*) malloc(32 * sizeof(char));

    if (result == TEST_FAIL) 
    {
        sprintf(str, "%s[%sFAIL%s%s]%s", 
            TEST_BOLD, TEST_RED, TEST_DEFAULT, TEST_BOLD, TEST_DEFAULT
        );
    }
    else if (result == TEST_PASS) 
    {
        sprintf(str, "%s[%sPASS%s%s]%s", 
            TEST_BOLD, TEST_GREEN, TEST_DEFAULT, TEST_BOLD, TEST_DEFAULT
        );
    }

    return str;
}

/*
 *
 */
void test_print_result (int result)
{
    char* str = test_get_result(result);
    fprintf(stdout, "%s", str);
    free(str);
}

//=============================================================================





//=============================================================================
// TEST CASE FUNCTIONS
//=============================================================================

/*
 *
 */
test_case_t* test_case_new (const char* name, test_func_t function)
{
    test_case_t* tcase = (test_case_t*) malloc(sizeof(test_case_t));
    tcase->name     = name;
    tcase->function = function;
    return tcase;
}

/*
 *
 */
void test_case_free (test_case_t* tcase)
{
    free(tcase);
}

/*
 *
 */
int test_case_run (test_case_t* tcase)
{
    return tcase->function();
}

/*
 *
 */
int test_case (test_case_t* tcase)
{
    int result;

    fprintf(stdout, "  - %s%s%s \t", 
        TEST_BOLD, tcase->name, TEST_DEFAULT
    );
    
    result = test_case_run(tcase);
    test_print_result(result);
    fprintf(stdout, "\n");

    return result;
}

//=============================================================================





//=============================================================================
// TEST SUITE FUNCTIONS
//=============================================================================

/*
 *
 */
test_suite_t* test_suite_new (const char* name)
{
    test_case_t** list = (test_case_t**) malloc(TEST_STDSIZE*sizeof(test_case_t*));
    for (int i = 0; i < TEST_STDSIZE; i++) 
    {
        list[i] = (test_case_t*) malloc(sizeof(test_case_t));
    }

    test_suite_t* tsuite = (test_suite_t*) malloc(sizeof(test_suite_t));
    tsuite->name         = name;
    tsuite->cases        = list;
    tsuite->size         = 0;
    tsuite->total_size   = TEST_STDSIZE;

    return tsuite;
}

/*
 *
 */
void test_suite_free (test_suite_t* tsuite)
{
    for (int i = 0; i < tsuite->size; i++) {
        test_case_free(tsuite->cases[i]);
    }
    free(tsuite);
}

/*
 *
 */
void test_suite_push (test_suite_t* tsuite, test_case_t* tcase)
{
    if (tsuite->size == tsuite->total_size)
    {
        tsuite->total_size = tsuite->size * 2;
        tsuite->cases =
            (test_case_t**) realloc(tsuite->cases,tsuite->total_size*sizeof(test_case_t*));
        for (int i = 0; i < tsuite->total_size-tsuite->size; i++)
        {
            tsuite->cases[tsuite->size+i] = 
                (test_case_t*) malloc(sizeof(test_case_t));
        }
    }
    else
    {
        tsuite->cases[tsuite->size++] = tcase;
    }
}

/*
 *
 */
void test_suite_pop (test_suite_t* tsuite)
{
    // TODO
}

/*
 *
 */
void test_suite_add (test_suite_t* tsuite, test_case_t* tcase)
{
    // TODO
}

/*
 *
 */
void test_suite_remove (test_suite_t* tsuite)
{
    // TODO
}

/*
 *
 */
int test_suite_exec (test_suite_t* tsuite)
{
    int result = TEST_PASS;

    for (int i = 0; i < tsuite->size; i++) 
    {
        result = test_case(tsuite->cases[i]) && result;
    }

    return result;
}

/*
 *
 */
int test_suite_run (test_suite_t* suite)
{
    int result;

    fprintf(stdout, "> %s%s%s%s\n", 
        TEST_BOLD, TEST_BLUE, suite->name, TEST_DEFAULT
    );
    
    result = test_suite_exec(suite);
    
    fprintf(stdout, "\n");

    return result;
}

//=============================================================================





//=============================================================================
// TEST FUNCTIONS
//=============================================================================

/*
 *
 */
test_t* test_new (const char* name)
{
    test_suite_t** list = (test_suite_t**) malloc(TEST_STDSIZE * sizeof(test_suite_t*));
    for (int i = 0; i < TEST_STDSIZE; i++) 
    {
        list[i] = (test_suite_t*) malloc(sizeof(test_suite_t));
    }

    test_t* t = (test_t*) malloc(sizeof(test_t));
    t->name         = name;
    t->suites       = list;
    t->size         = 0;
    t->total_size   = TEST_STDSIZE;

    return t;
}

/*
 *
 */
void test_free (test_t* t)
{
    for (int i = 0; i < t->size; i++) {
        test_suite_free(t->suites[i]);
    }
    free(t);
}

/*
 *
 */
void test_push (test_t* t, test_suite_t* tsuite)
{
    if (t->size == t->total_size)
    {
        t->total_size = t->size * 2;
        t->suites =
            (test_suite_t**) realloc(t->suites,t->total_size*sizeof(test_suite_t*));
        for (int i = 0; i < t->total_size-t->size; i++)
        {
            t->suites[t->size+i] = 
                (test_suite_t*) malloc(sizeof(test_suite_t));
        }
    }
    else
    {
        t->suites[t->size++] = tsuite;
    }
}

/*
 *
 */
void test_pop (test_t* t)
{
    // TODO
}

/*
 *
 */
void test_add (test_t* t, test_suite_t* tsuite, int position)
{
    // TODO
}

/*
 *
 */
void test_remove (test_t* t, int position)
{
    // TODO
}

/*
 *
 */
int test_exec (test_t* t)
{
    int result = TEST_PASS;
    for (int i = 0; i < t->size; i++) 
    {
        int res = test_suite_run(t->suites[i]);
        result = res && result;
    }

    return result;
}

/*
 *
 */
int test_run (test_t* t)
{
    fprintf(stdout, "%s%s%s%s\n\n", 
        TEST_BOLD, TEST_BLUE, t->name, TEST_DEFAULT
    );
    int result = test_exec(t);

    return result;
}

//=============================================================================
