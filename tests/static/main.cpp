#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CE::StaticFunctionTest"

#define CE_DEBUG_CACHE_USAGE
#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>

#include <ce/execute.hpp>
#include <math.h>
#include <cmath>
void foo0(){

}

int foo1(int value){
    return value * 2;
}

double foo2(int value1, double value2){
    return std::pow(value2, value1);
}

double foo3(double a, double x, double b){
    return a * x + b;
}

void foo4(double a, double x, double b, double& y){
    y = a * x + b;
}

BOOST_AUTO_TEST_CASE(initialize){
    ce::ICacheEngine::setEngine(ce::ICacheEngine::create());
}

BOOST_AUTO_TEST_CASE(test_foo1){
    BOOST_REQUIRE_EQUAL(ce::exec(foo1, 10), foo1(10));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    BOOST_REQUIRE_EQUAL(ce::exec(foo1, 11), foo1(11));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    // Get cached results
    BOOST_REQUIRE_EQUAL(ce::exec(foo1, 10),foo1(10));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
    BOOST_REQUIRE_EQUAL(ce::exec(foo1, 11), foo1(11));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
}

BOOST_AUTO_TEST_CASE(test_foo2) {
    BOOST_REQUIRE_EQUAL(ce::exec(foo2, 10, 5.0), foo2(10,5.0));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    BOOST_REQUIRE_EQUAL(ce::exec(foo2, 11, 6.0), foo2(11,6.0));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    // Get cached results
    BOOST_REQUIRE_EQUAL(ce::exec(foo2, 10, 5.0), foo2(10, 5.0));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
    BOOST_REQUIRE_EQUAL(ce::exec(foo2, 11, 6.0), foo2(11, 6.0));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
}

BOOST_AUTO_TEST_CASE(test_foo3) {
    BOOST_REQUIRE_EQUAL(ce::exec(foo3, 2, 3, 4), foo3(2,3,4));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    BOOST_REQUIRE_EQUAL(ce::exec(foo3, 5, 6, 7), foo3(5,6,7));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    // Get cached results
    BOOST_REQUIRE_EQUAL(ce::exec(foo3, 2,3,4), foo3(2,3,4));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
    BOOST_REQUIRE_EQUAL(ce::exec(foo3, 5,6,7), foo3(5,6,7));
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
}

BOOST_AUTO_TEST_CASE(test_foo4) {
    double result1, result2 = 0.0;
    ce::exec(foo4, 2,3,4, ce::make_output(result1));
    foo4(2,3,4, result2);
    BOOST_REQUIRE_EQUAL(result1, result2);
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    ce::exec(foo4, 5, 6, 7, ce::make_output(result1));
    foo4(5, 6, 7, result2);
    BOOST_REQUIRE_EQUAL(result1, result2); 
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    // Get cached results
    ce::exec(foo4, 2, 3, 4, ce::make_output(result1));
    foo4(2, 3, 4, result2);
    BOOST_REQUIRE_EQUAL(result1, result2);
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
    ce::exec(foo4, 5, 6, 7, ce::make_output(result1));
    foo4(5, 6, 7, result2);
    BOOST_REQUIRE_EQUAL(result1, result2);
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
}

BOOST_AUTO_TEST_CASE(test_chain){
    auto result1 = ce::exec(foo2, 5, 10);
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);
    auto result2 = ce::exec(foo3, ce::make_input(result1), 4,5);
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), false);

    auto result3 = ce::exec(foo2, 5, 10);
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
    BOOST_REQUIRE_EQUAL(result1.m_hash, result3.m_hash);
    auto result4 = ce::exec(foo3, ce::make_input(result3), 4, 5);
    BOOST_REQUIRE_EQUAL(ce::wasCacheUsedLast(), true);
    BOOST_REQUIRE_EQUAL(result2.m_hash, result4.m_hash);
}


BOOST_AUTO_TEST_CASE(cleanup) {
    ce::ICacheEngine::releaseEngine();
}