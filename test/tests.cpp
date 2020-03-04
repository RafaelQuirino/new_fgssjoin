#include <stdio.h>
#include <stdlib.h>

#include "include/test_io.hpp"
#include "include/test_util.hpp"
#include "include/test_qgram.hpp"
#include "include/test_token.hpp"

/*
 *
 */
int main (int argc, char** argv)
{
    test_t* test = test_new("FGSSJOIN TEST");
    test_push(test, test_io_getsuite());
    test_push(test, test_util_getsuite());
    test_push(test, test_qgram_getsuite());
    test_push(test, test_token_getsuite());

    int result = test_run(test);

    fprintf(stdout, "Test result: %s\n", test_get_result(result));
    test_free(test);

    return 0;
}