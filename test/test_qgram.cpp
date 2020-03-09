#include <stdio.h>
#include <stdlib.h>

#include "include/test_qgram.hpp"
#include "../include/qgram.hpp"

/*
 *
 */
TestSuite test_qgram_getsuite ()
{
    TestSuite suite = test_suite_new("QGRAM", test_qgram_getfixtures());
    test_suite_push(suite, test_case_new("qgram_function", test_qgram_function));

    return suite;
}

/*
 *
 */
TestFixtures test_qgram_getfixtures ()
{
    return NULL;
}

/*
 *
 */
int test_qgram_function (TestFixtures fixtures)
{
    int result = TEST_PASS;
    
    return result;
}