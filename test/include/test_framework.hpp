/* +---------------------------------------------------------------------------+
 * | TEST FRAMEWORK
 * | --------------
 * |
 * | Test framework header file.
 * |
 * +---------------------------------------------------------------------------+
 */



#ifndef __TEST_FRAMEWORK_HPP__
#define __TEST_FRAMEWORK_HPP__



/*
 *  TEST RESULT CONSTANTS
 */
#define TEST_FAIL 0
#define TEST_PASS 1



/*
 *  TEST COLOR CONSTANTS
 */
#define TEST_DEFAULT "\033[0m"
#define TEST_BOLD    "\e[1m"
#define TEST_RED     "\e[31m"
#define TEST_GREEN   "\e[92m"
#define TEST_BLUE    "\e[34m"
#define TEST_MAGENTA "\e[95m"
#define TEST_CYAN    "\e[96m"



/*
 *  INITIAL SIZE OF LISTS
 */
#define TEST_STDSIZE 32



/*
 *  TEST FUNCTION TYPE
 */
typedef int(*test_func_t)(void);



/*
 *  TEST SUITE FIXTURE TYPE
 */
typedef struct {
    const char* name;
    void* object;
} test_fixture_t;



/*
 *  LIST OF FIXTURES
 */
typedef struct {
    test_fixture_t** list;
    int size;
} test_fixtures_t;



/*
 *  TEST CASE TYPE
 */
typedef struct {
    test_func_t function;
    const char* name;
} test_case_t;



/*
 *  TEST SUITE TYPE
 */
typedef struct {
    const char*      name;
    test_case_t**    cases;
    test_fixtures_t* fixtures;
    int size;
    int total_size;
} test_suite_t;



/*
 *  TEST STATS TYPE
 *  TODO
 */
typedef struct {
    // About cases
    int failed_cases;
    int passed_cases;
    // About suites
    int failed_suites;
    int passed_suites;
} test_stats_t;



/*
 *  TEST TYPE
 */
typedef struct {
    const char* name;
    test_suite_t** suites;
    int size;
    int total_size;
} test_t;





//=============================================================================
// TEST UTIL FUNCTIONS
//=============================================================================
/*
 *
 */
char* test_get_result (int result);

/*
 *
 */
void test_print_result (int result);

//=============================================================================





//=============================================================================
// TEST CASE FUNCTIONS
//=============================================================================
/*
 *
 */
test_case_t* test_case_new (const char* name, test_func_t function);

/*
 *
 */
void test_case_free (test_case_t* tcase);

/*
 *
 */
int test_case_run (test_case_t* tcase);

/*
 *
 */
int test_case (test_case_t* tcase);

//=============================================================================





//=============================================================================
// TEST SUITE FUNCTIONS
//=============================================================================

/*
 *
 */
test_suite_t* test_suite_new (const char* name);

/*
 *
 */
void test_suite_free (test_suite_t* tsuite);

/*
 *
 */
void test_suite_push (test_suite_t* tsuite, test_case_t* tcase);

/*
 *
 */
void test_suite_pop (test_suite_t* tsuite);

/*
 *
 */
void test_suite_add (test_suite_t* tsuite, test_case_t* tcase);

/*
 *
 */
void test_suite_remove (test_suite_t* tsuite);

/*
 *
 */
int test_suite_exec (test_suite_t* tsuite);

/*
 *
 */
int test_suite_run (test_suite_t* suite);

//=============================================================================





//=============================================================================
// TEST FUNCTIONS
//=============================================================================

/*
 *
 */
test_t* test_new (const char* name);

/*
 *
 */
void test_free (test_t* tsuite);

/*
 *
 */
void test_push (test_t* t, test_suite_t* tsuite);

/*
 *
 */
void test_pop (test_t* t);

/*
 *
 */
void test_add (test_t* t, test_suite_t* tsuite, int position);

/*
 *
 */
void test_remove (test_t* t, int position);

/*
 *
 */
int test_exec (test_t* t);

/*
 *
 */
int test_run (test_t* t);

//=============================================================================



#endif /*__TEST_FRAMEWORK_HPP__*/
