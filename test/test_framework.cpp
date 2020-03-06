/* +---------------------------------------------------------------------------+
 * | TEST FRAMEWORK
 * | --------------
 * |
 * | Test framework source file.
 * |
 * +---------------------------------------------------------------------------+
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "include/test_framework.hpp"



//=============================================================================
// TEST UTIL FUNCTIONS
//=============================================================================

/*
 *
 */
char* test_get_result (int result)
{
    char* str = (char*) malloc(32 * sizeof(char));

    if (result == TEST_FAIL) 
    {
        sprintf(str, "%s[%sFAIL%s%s]%s", 
            TEST_BOLD, TEST_RED, TEST_DEFAULT, TEST_BOLD, TEST_DEFAULT
        );
    }
    else if (result == TEST_PASS) 
    {
        sprintf(str, "%s[%sPASS%s%s]%s", 
            TEST_BOLD, TEST_GREEN, TEST_DEFAULT, TEST_BOLD, TEST_DEFAULT
        );
    }

    return str;
}

/*
 *
 */
void test_print_result (int result)
{
    char* str = test_get_result(result);
    fprintf(stdout, "%s", str);
    free(str);
}

//=============================================================================





//=============================================================================
// TEST FIXTURE FUNCTIONS
//=============================================================================

/*
 *
 */
TestFixture test_fixture_new (const char* name, void* object)
{
    TestFixture fixture = (TestFixture) malloc(sizeof(test_fixture_t));
    fixture->name   = name;
    fixture->object = object;

    return fixture;
}

/*
 *
 */
void test_fixture_free (TestFixture fixture)
{
    free(fixture->object);
    free(fixture);
}

/*
 *
 */
TestFixtures test_fixtures_new (const char* name)
{
    int i;

    TestFixture* list = (TestFixture*) malloc(TEST_STDSIZE*sizeof(TestFixture));
    for (i = 0; i < TEST_STDSIZE; i++)
    {
        list[i] = (TestFixture) malloc(sizeof(test_fixture_t));
    }
    
    TestFixtures fixtures = (TestFixtures) malloc(sizeof(test_fixtures_t));
    fixtures->name        = name;
    fixtures->list        = list;
    fixtures->size        = 0;
    fixtures->total_size  = TEST_STDSIZE;
    
    return fixtures;
}

/*
 *
 */
void test_fixtures_free (TestFixtures fixtures)
{
    int i;
    for (i = 0; i < fixtures->total_size; i++) 
    {
        test_fixture_free(fixtures->list[i]);
    }
    free(fixtures);
}

/*
 *
 */
void test_fixtures_push (TestFixtures fixtures, TestFixture fixture)
{
    int i;

    if (fixtures->size == fixtures->total_size)
    {
        fixtures->total_size = fixtures->size * 2;
        fixtures->list =
            (TestFixture*) realloc(fixtures->list,fixtures->total_size*sizeof(TestFixture));
        for (i = 0; i < fixtures->total_size-fixtures->size; i++)
        {
            fixtures->list[fixtures->size+i] = 
                (TestFixture) malloc(sizeof(test_fixture_t));
        }
    }
    else
    {
        fixtures->list[fixtures->size++] = fixture;
    }
}

/*
 *
 */
void test_fixtures_pop (TestFixtures fixtures)
{

}

/*
 *
 */
void test_fixtures_add (TestFixtures fixtures, TestFixture fixture, int position)
{

}

/*
 *
 */
void test_fixtures_remove (TestFixtures fixtures, int position)
{

}

/*
 *
 */
TestFixture test_fixtures_find (TestFixtures fixtures, const char* name)
{
    int i;
    for (i = 0; i < fixtures->size; i++)
    {
        if (strcmp(fixtures->list[i]->name, name) == 0)
        {
            return fixtures->list[i];
        }
    }

    return NULL;
}

//=============================================================================





//=============================================================================
// TEST CASE FUNCTIONS
//=============================================================================

/*
 *
 */
TestCase test_case_new (const char* name, TestFunction function)
{
    TestCase tcase  = (TestCase) malloc(sizeof(test_case_t));
    tcase->name     = name;
    tcase->function = function;

    return tcase;
}

/*
 *
 */
void test_case_free (TestCase tcase)
{
    free(tcase);
}

/*
 *
 */
int test_case_exec (TestCase tcase)
{
    return tcase->function();
}

/*
 *
 */
int test_case_run (TestCase tcase)
{
    int result;

    fprintf(stdout, "  - %s%s%s \t", 
        TEST_BOLD, tcase->name, TEST_DEFAULT
    );
    
    result = test_case_exec(tcase);
    test_print_result(result);
    fprintf(stdout, "\n");

    return result;
}

//=============================================================================





//=============================================================================
// TEST SUITE FUNCTIONS
//=============================================================================

/*
 *
 */
TestSuite test_suite_new (const char* name)
{
    int i;

    TestCase* list = (TestCase*) malloc(TEST_STDSIZE*sizeof(TestCase));
    for (i = 0; i < TEST_STDSIZE; i++) 
    {
        list[i] = (TestCase) malloc(sizeof(test_case_t));
    }

    TestSuite suite = (TestSuite) malloc(sizeof(test_suite_t));
    suite->name         = name;
    suite->cases        = list;
    suite->fixtures     = NULL;
    suite->size         = 0;
    suite->total_size   = TEST_STDSIZE;

    return suite;
}

/*
 *
 */
void test_suite_free (TestSuite tsuite)
{
    int i;
    for (i = 0; i < tsuite->size; i++) 
    {
        test_case_free(tsuite->cases[i]);
    }
    free(tsuite);
}

/*
 *
 */
void test_suite_push (TestSuite tsuite, TestCase tcase)
{
    int i;

    if (tsuite->size == tsuite->total_size)
    {
        tsuite->total_size = tsuite->size * 2;
        tsuite->cases =
            (TestCase*) realloc(tsuite->cases,tsuite->total_size*sizeof(TestCase));
        for (i = 0; i < tsuite->total_size-tsuite->size; i++)
        {
            tsuite->cases[tsuite->size+i] = 
                (TestCase) malloc(sizeof(test_case_t));
        }
    }
    else
    {
        tsuite->cases[tsuite->size++] = tcase;
    }
}

/*
 *
 */
void test_suite_pop (TestSuite tsuite)
{
    // TODO
}

/*
 *
 */
void test_suite_add (TestSuite tsuite, TestCase tcase)
{
    // TODO
}

/*
 *
 */
void test_suite_remove (TestSuite tsuite)
{
    // TODO
}

/*
 *
 */
int test_suite_exec (TestSuite tsuite)
{
    int i, result = TEST_PASS;

    for (i = 0; i < tsuite->size; i++) 
    {
        result = test_case_run(tsuite->cases[i]) && result;
    }

    return result;
}

/*
 *
 */
int test_suite_run (TestSuite suite)
{
    int result;

    fprintf(stdout, "> %s%s%s%s\n", 
        TEST_BOLD, TEST_BLUE, suite->name, TEST_DEFAULT
    );
    
    result = test_suite_exec(suite);
    
    fprintf(stdout, "\n");

    return result;
}

//=============================================================================





//=============================================================================
// TEST FUNCTIONS
//=============================================================================

/*
 *
 */
Test test_new (const char* name)
{
    int i;

    TestSuite* list = (TestSuite*) malloc(TEST_STDSIZE * sizeof(TestSuite));
    for (i = 0; i < TEST_STDSIZE; i++) 
    {
        list[i] = (TestSuite) malloc(sizeof(test_suite_t));
    }

    Test t = (Test) malloc(sizeof(test_t));
    t->name         = name;
    t->suites       = list;
    t->size         = 0;
    t->total_size   = TEST_STDSIZE;

    return t;
}

/*
 *
 */
void test_free (Test t)
{
    int i;
    for (i = 0; i < t->size; i++) 
    {
        test_suite_free(t->suites[i]);
    }
    free(t);
}

/*
 *
 */
void test_push (Test t, TestSuite tsuite)
{
    int i;

    if (t->size == t->total_size)
    {
        t->total_size = t->size * 2;
        t->suites =
            (TestSuite*) realloc(t->suites,t->total_size*sizeof(TestSuite));
        for (i = 0; i < t->total_size-t->size; i++)
        {
            t->suites[t->size+i] = 
                (TestSuite) malloc(sizeof(test_suite_t));
        }
    }
    else
    {
        t->suites[t->size++] = tsuite;
    }
}

/*
 *
 */
void test_pop (Test t)
{
    // TODO
}

/*
 *
 */
void test_add (Test t, TestSuite tsuite, int position)
{
    // TODO
}

/*
 *
 */
void test_remove (Test t, int position)
{
    // TODO
}

/*
 *
 */
int test_exec (Test t)
{
    int i, result = TEST_PASS;
    for (i = 0; i < t->size; i++) 
    {
        int res = test_suite_run(t->suites[i]);
        result = res && result;
    }

    return result;
}

/*
 *
 */
int test_run (Test t)
{
    fprintf(stdout, "%s%s%s%s\n\n", 
        TEST_BOLD, TEST_BLUE, t->name, TEST_DEFAULT
    );
    int result = test_exec(t);

    return result;
}

//=============================================================================
