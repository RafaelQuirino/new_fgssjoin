#ifndef __TEST_IO_HPP__
#define __TEST_IO_HPP__

#include "test_framework.hpp"

/*
 *  Get IO test suite
 */
test_suite_t* test_io_getsuite ();

/*
 *  IO test suite functions
 */
int test_io_readlines ();
int test_io_getlines  ();
int test_io_getnlines ();

#endif /* __TEST_IO_HPP__ */
