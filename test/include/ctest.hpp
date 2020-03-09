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
#define TEST_YELLOW  "\e[33m"



/*
 *  INITIAL SIZE OF LISTS
 */
#define TEST_STDSIZE 32



/*
 *  TEST SUITE FIXTURE TYPE
 */
typedef struct 
{
    const char* name;
    void*       object;

} test_fixture_t;



/*
 *  LIST OF FIXTURES
 */
typedef struct _test_fixtures_t
{
    const char*      name;
    test_fixture_t** list;
    int              size;
    int              total_size;

    void(*create) (_test_fixtures_t*);
    void(*destroy)(_test_fixtures_t*);

} test_fixtures_t;



/*
 *  FUNCTION TYPES
 */
typedef void(*test_fixtures_func_t)(test_fixtures_t*);
typedef int(*test_func_t)(test_fixtures_t*);



/*
 *  TEST CASE TYPE
 */
typedef struct _test_case_t
{
    const char* name;
    test_func_t function;

} test_case_t;



/*
 *  TEST SUITE TYPE
 */
typedef struct 
{
    const char*      name;
    test_case_t**    cases;
    test_fixtures_t* fixtures;
    int              size;
    int              total_size;

} test_suite_t;



/*
 *  TEST STATISTICS TYPE
 *  TODO
 */
typedef struct 
{
    // About cases
    int failed_cases;
    int passed_cases;
    // About suites
    int failed_suites;
    int passed_suites;
    
    // TODO

} test_stats_t;



/*
 *  TEST TYPE
 */
typedef struct 
{
    const char*       name;
    test_suite_t**    suites;
    test_fixtures_t** fixtures;
    int               size;
    int               total_size;

} test_t;



/*
 *  SPECIAL CLASS LIKE ALIASES
 */
typedef test_fixtures_func_t TestFixturesFunction;
typedef test_func_t          TestFunction;
typedef test_fixture_t*      TestFixture;
typedef test_fixtures_t*     TestFixtures;
typedef test_case_t*         TestCase;
typedef test_suite_t*        TestSuite;
typedef test_t*              Test;





//=============================================================================
// TEST UTIL FUNCTIONS
//=============================================================================
/*
 *
 */
char* test_get_resultstr (int result);

/*
 *
 */
void test_print_resultstr (int result);

//=============================================================================





//=============================================================================
// TEST FIXTURE FUNCTIONS
//=============================================================================

/*
 *
 */
TestFixture test_fixture_new (const char* name, void* object);

/*
 *
 */
void test_fixture_free (TestFixture fixture);

/*
 *
 */
TestFixtures test_fixtures_new (
    TestFixturesFunction create_fixtures,
    TestFixturesFunction destroy_fixtures
);

/*
 *
 */
void test_fixtures_free (TestFixtures fixtures);

/*
 *
 */
void test_fixtures_push (TestFixtures fixtures, TestFixture fixture);

/*
 *
 */
void test_fixtures_pop ();

/*
 *
 */
void test_fixtures_add (TestFixtures fixtures, TestFixture fixture, int position);

/*
 *
 */
void test_fixtures_remove (TestFixtures fixtures, int position);

/*
 *
 */
TestFixture test_fixtures_find (TestFixtures fixtures, const char* name);

/*
 *
 */
void* test_fixtures_findobj (TestFixtures fixtures, const char* name);

//=============================================================================





//=============================================================================
// TEST CASE FUNCTIONS
//=============================================================================
/*
 *
 */
TestCase test_case_new (const char* name, TestFunction function);

/*
 *
 */
void test_case_free (TestCase tcase);

/*
 *
 */
int test_case_exec (TestCase tcase, TestFixtures fixtures);

/*
 *
 */
int test_case_run (TestCase tcase, TestFixtures fixtures);

//=============================================================================





//=============================================================================
// TEST SUITE FUNCTIONS
//=============================================================================

/*
 *
 */
TestSuite test_suite_new (const char* name, TestFixtures fixtures);

/*
 *
 */
void test_suite_free (TestSuite tsuite);

/*
 *
 */
void test_suite_push (TestSuite tsuite, TestCase tcase);

/*
 *
 */
void test_suite_pop (TestSuite tsuite);

/*
 *
 */
void test_suite_add (TestSuite tsuite, TestCase tcase);

/*
 *
 */
void test_suite_remove (TestSuite tsuite);

/*
 *
 */
// UPDATE:
int test_suite_exec (TestSuite tsuite);

/*
 *
 */
int test_suite_run (TestSuite suite);

//=============================================================================





//=============================================================================
// TEST FUNCTIONS
//=============================================================================

/*
 *
 */
Test test_new (const char* name);

/*
 *
 */
void test_free (Test test);

/*
 *
 */
void test_push (Test test, TestSuite suite);

/*
 *
 */
void test_pop (Test test);

/*
 *
 */
void test_add (Test test, TestSuite suite, int position);

/*
 *
 */
void test_remove (Test test, int position);

/*
 *
 */
int test_exec (Test test);

/*
 *
 */
int test_run (Test test);

//=============================================================================



#endif /*__TEST_FRAMEWORK_HPP__*/
