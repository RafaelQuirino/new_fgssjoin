#include <stdio.h>
#include <stdlib.h>

#include "include/test_util.hpp"
#include "../include/util.hpp"

/*
 *
 */
test_suite_t* test_util_getsuite ()
{
    test_suite_t* suite = test_suite_new("UTIL");
    test_suite_push(suite, test_case_new("util_function", test_util_function));

    return suite;
}

/*
 *
 */
int test_util_function ()
{
    int result = TEST_PASS;

    // SETTING FIXTURES

    // TESTING FUNCTION

    // CLEANING FIXTURES
    
    return result;
}