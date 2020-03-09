#ifndef __TEST_UTIL_HPP__
#define __TEST_UTIL_HPP__

// Test framework
#include "ctest.hpp"

/*
 *  Get UTIL test suite
 */
TestSuite test_util_getsuite ();

/*
 *
 */
TestFixtures test_util_getfixtures ();

/*
 *  Test suite functions
 */
int test_util_function (TestFixtures fixtures);

#endif /* __TEST_UTIL_HPP__ */
