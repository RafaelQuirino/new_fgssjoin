#include <stdio.h>
#include <stdlib.h>

#include "include/test_token.hpp"
#include "../include/token/token.hpp"

/*
 *
 */
test_suite_t* test_token_getsuite ()
{
    test_suite_t* suite = test_suite_new("TOKEN");
    test_suite_push(suite, test_case_new("token_function", test_token_function));

    return suite;
}

/*
 *
 */
int test_token_function ()
{
    int result = TEST_PASS;

    // SETTING FIXTURES

    // TESTING FUNCTION

    // CLEANING FIXTURES
    
    return result;
}