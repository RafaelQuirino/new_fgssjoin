#ifndef __TEST_IO_HPP__
#define __TEST_IO_HPP__

// Test framework
#include "ctest.hpp"

/*
 *  Get IO test suite
 */
TestSuite test_io_getsuite ();

/*
 *  Get IO fixtures
 */
TestFixtures test_io_getfixtures ();
void         test_io_createfile  (TestFixtures fixtures);
void         test_io_destroyfile (TestFixtures fixtures);

/*
 *  IO test suite functions
 */
int test_io_readlines (TestFixtures fixtures);
int test_io_getlines  (TestFixtures fixtures);
int test_io_getnlines (TestFixtures fixtures);

#endif /* __TEST_IO_HPP__ */
