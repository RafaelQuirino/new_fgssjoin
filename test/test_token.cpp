#include <stdio.h>
#include <stdlib.h>

#include "include/test_token.hpp"
#include "../include/token/token.hpp"

/*
 *
 */
TestSuite test_token_getsuite ()
{
    TestSuite suite = test_suite_new("TOKEN", test_token_getfixtures());
    test_suite_push(suite, test_case_new("token_function", test_token_function));

    return suite;
}

/*
 *
 */
TestFixtures test_token_getfixtures ()
{
    return NULL;
}

/*
 *
 */
int test_token_function (TestFixtures fixtures)
{
    int result = TEST_PASS;
    
    return result;
}