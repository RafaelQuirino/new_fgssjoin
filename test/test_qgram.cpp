#include <stdio.h>
#include <stdlib.h>

#include "include/test_qgram.hpp"
#include "../include/qgram.hpp"

/*
 *
 */
test_suite_t* test_qgram_getsuite ()
{
    test_suite_t* suite = test_suite_new("QGRAM");
    test_suite_push(suite, test_case_new("qgram_function", test_qgram_function));

    return suite;
}

/*
 *
 */
int test_qgram_function ()
{
    int result = TEST_PASS;

    // SETTING FIXTURES

    // TESTING FUNCTION

    // CLEANING FIXTURES
    
    return result;
}