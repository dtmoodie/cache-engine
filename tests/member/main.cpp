#define CE_DEBUG_CACHE_USAGE
#include "object.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CE::StaticFunctionTest"

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>

#include <ce/execute.hpp>
#include <ct/StringView.hpp>
#include <ct/reflect/MemberObjectPointer.hpp>

#include <cmath>
#include <iostream>
#include <math.h>

BOOST_AUTO_TEST_CASE(class_hash_uniqueness)
{
    /*static_assert(ce::classHash<TestObject>() != ce::classHash<TestHashedObject>(),
                  "Hash must be unique for different objects");
    static_assert(ct::ctcrc32_ignore_whitespace("TestObject::get") == ct::ctcrc32_ignore_whitespace("TestObject:: get"),
                  "Hash must ignore whitespace");*/
}

BOOST_AUTO_TEST_CASE(initialize)
{
    ce::ICacheEngine::setEngine(ce::ICacheEngine::create());
}

BOOST_AUTO_TEST_CASE(member_access_with_executor_owner)
{
    auto executor1 = ce::makeExecutor<TestObject>();
    auto executor2 = ce::makeExecutor<TestObject>();
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    auto old_hash = executor1.m_hash;
    auto ret1 = executor1.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    auto ret2 = executor2.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::wasCacheUsedLast() == true);
    BOOST_REQUIRE_EQUAL(ret1, ret2);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, old_hash);
}

// Since we are wrapping an object without knowing the prior state of the object, we cannot assume the object is in the
// same state
// TODO write a function that generates a hash based on the state of an object and use that in the wrapping case
BOOST_AUTO_TEST_CASE(member_access_with_executor_wrapper)
{
    auto val = ce::memberFunctionPointerValue(&TestObject::get);
    TestObject obj1;
    TestObject obj2;
    auto executor1 = ce::makeExecutor(obj1);
    auto executor2 = ce::makeExecutor(obj2);
    BOOST_REQUIRE_NE(executor1.m_hash, executor2.m_hash);
    auto old_hash = executor1.m_hash;
    auto ret1 = executor1.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    auto ret2 = executor2.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    BOOST_REQUIRE_EQUAL(ret1, ret2);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, old_hash);
}

BOOST_AUTO_TEST_CASE(member_set_with_executor_owner)
{
    auto executor1 = ce::makeExecutor<TestObject>();
    auto executor2 = ce::makeExecutor<TestObject>();
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    auto old_hash = executor1.m_hash;
    executor1.exec (&TestObject::set)(4);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    executor2.exec (&TestObject::set)(4);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    BOOST_REQUIRE_NE(executor1.m_hash, old_hash);

    old_hash = executor1.m_hash;
    auto ret1 = executor1.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    auto ret2 = executor2.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::wasCacheUsedLast() == true);
    BOOST_REQUIRE_EQUAL(old_hash, executor1.m_hash);
    BOOST_REQUIRE_EQUAL(ret1, 4);
    BOOST_REQUIRE_EQUAL(ret1, ret2);
}

BOOST_AUTO_TEST_CASE(member_apply_return_with_owner)
{
    auto executor1 = ce::makeExecutor<TestObject>();
    auto executor2 = ce::makeExecutor<TestObject>();
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    auto old_hash = executor1.m_hash;

    auto ret1 = executor1.exec(static_cast<int (TestObject::*)(int, int) const>(&TestObject::apply))(4, 5);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    // Intentially have different whitespace between the two function calls to verify a whitespace ignoreant hash is
    // used
    auto ret2 = executor2.exec(static_cast<int (TestObject::*)(int, int) const>(&TestObject::apply))(4, 5);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == true);

    BOOST_REQUIRE_EQUAL(ret1, ret2);
    BOOST_REQUIRE_EQUAL(old_hash, executor1.m_hash);
}

BOOST_AUTO_TEST_CASE(member_apply_with_owner)
{
    auto executor1 = ce::makeExecutor<TestObject>();
    auto executor2 = ce::makeExecutor<TestObject>();
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    auto old_hash = executor1.m_hash;
    int val1;
    int val2;
    auto out1 = ce::makeOutput(val1);
    auto out2 = ce::makeOutput(val2);
    executor1.exec(static_cast<void (TestObject::*)(int, int&) const>(&TestObject::apply))(4, out1);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    // Intentially have different whitespace between the two function calls to verify a whitespace ignoreant hash is
    // used
    executor2.exec(static_cast<void (TestObject::*)(int, int&) const>(&TestObject::apply))(4, out2);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == true);

    BOOST_REQUIRE_EQUAL(val1, val2);
    BOOST_REQUIRE_EQUAL(old_hash, executor1.m_hash);
    BOOST_REQUIRE_EQUAL(out1.m_hash, out2.m_hash);
    auto old_out_hash = out1.m_hash;
    executor1.exec (&TestObject::set)(4);
    BOOST_REQUIRE_NE(executor1.m_hash, executor2.m_hash);
    executor2.exec (&TestObject::set)(4);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);

    executor1.exec(static_cast<void (TestObject::*)(int, int&) const>(&TestObject::apply))(4, out1);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);

    executor2.exec(static_cast<void (TestObject::*)(int, int&) const>(&TestObject::apply))(4, out2);
    BOOST_REQUIRE(ce::wasCacheUsedLast() == true);

    BOOST_REQUIRE_EQUAL(val1, val2);
    BOOST_REQUIRE_NE(old_out_hash, out1.m_hash);
    BOOST_REQUIRE_EQUAL(out1.m_hash, out2.m_hash);
}

BOOST_AUTO_TEST_CASE(mutate_hashed_object)
{
    MutateOutputObject obj1;
    MutateOutputObject obj2;

    TestHashedOutputObject out1;
    TestHashedOutputObject out2;
    BOOST_REQUIRE_EQUAL(ce::getObjectHash(obj1), ce::getObjectHash(obj2));
    ce::exec (&MutateOutputObject::mutate)(obj1, ce::makeOutput(out1));
    BOOST_REQUIRE(ce::wasCacheUsedLast() == false);
    BOOST_REQUIRE_EQUAL(ce::getObjectHash(obj1), ce::getObjectHash(obj2));
    ce::exec (&MutateOutputObject::mutate)(obj2, ce::makeOutput(out2));
    BOOST_REQUIRE(ce::wasCacheUsedLast() == true);
    BOOST_REQUIRE_EQUAL(out1.hash, out2.hash);
    BOOST_REQUIRE_EQUAL(out1.data, out2.data);
    BOOST_REQUIRE_EQUAL(ce::getObjectHash(obj1), ce::getObjectHash(obj2));
}

BOOST_AUTO_TEST_CASE(cleanup)
{
    ce::ICacheEngine::releaseEngine();
}
