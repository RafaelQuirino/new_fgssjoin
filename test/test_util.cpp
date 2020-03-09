#include <stdio.h>
#include <stdlib.h>

#include "include/test_util.hpp"
#include "../include/util.hpp"

/*
 *
 */
TestSuite test_util_getsuite ()
{
    TestSuite suite = test_suite_new("UTIL", test_util_getfixtures());
    test_suite_push(suite, test_case_new("util_function", test_util_function));

    return suite;
}

/*
 *
 */
TestFixtures test_util_getfixtures ()
{
    return NULL;
}

/*
 *
 */
int test_util_function (TestFixtures fixtures)
{
    int result = TEST_PASS;

    // SETTING FIXTURES

    // TESTING FUNCTION

    // CLEANING FIXTURES
    
    return result;
}