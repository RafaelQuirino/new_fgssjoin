#ifndef __TEST_QGRAM_HPP__
#define __TEST_QGRAM_HPP__

// Test framework
#include "ctest.hpp"

/*
 *  Get QGRAM test suite
 */
TestSuite test_qgram_getsuite ();

/*
 *
 */
TestFixtures test_qgram_getfixtures ();

/*
 *  Test suite functions
 */
int test_qgram_function (TestFixtures fixtures);

#endif /* __TEST_QGRAM_HPP__ */
