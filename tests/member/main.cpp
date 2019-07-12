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

BOOST_AUTO_TEST_SUITE(const_member_functions);

BOOST_AUTO_TEST_SUITE_END();

BOOST_AUTO_TEST_SUITE(mutable_member_functions);

BOOST_AUTO_TEST_CASE(const_member_accessor)
{
    auto eng = ce::ICacheEngine::instance();
    auto hashed1 = ce::wrapHash<TestObject>();
    auto hashed2 = ce::wrapHash<TestObject>();
    BOOST_REQUIRE_EQUAL(hashed1.hash(), hashed2.hash());
    auto ret1 = ce::exec(&TestObject::get, hashed1);
    BOOST_REQUIRE_EQUAL(ret1, 0);
    ret1 = ce::exec(&TestObject::get, hashed2);
    BOOST_REQUIRE(eng->wasCacheUsedLast() == true);
    BOOST_REQUIRE_EQUAL(ret1, 0);
    auto hash = hashed1.hash();
    eng->exec(&TestObject::set, hashed1, 5);
    BOOST_REQUIRE_NE(hash, hashed1.hash());
    BOOST_REQUIRE_EQUAL(hashed1.obj.member1, 5);
    ret1 = eng->exec(&TestObject::get, hashed1);
    BOOST_REQUIRE_EQUAL(ret1, 5);
    BOOST_REQUIRE_NE(hashed1.hash(), hashed2.hash());
    BOOST_REQUIRE(eng->wasCacheUsedLast() == false);
    ret1 = eng->exec(&TestObject::get, hashed1);
    BOOST_REQUIRE(eng->wasCacheUsedLast() == true);
}

BOOST_AUTO_TEST_SUITE_END();

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

    /*BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    auto ret2 = executor2.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == true);
    BOOST_REQUIRE_EQUAL(ret1, ret2);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, old_hash);*/
}

// Since we are wrapping an object without knowing the prior state of the object, we cannot assume the object is in the
// same state
// TODO write a function that generates a hash based on the state of an object and use that in the wrapping case
/*BOOST_AUTO_TEST_CASE(member_access_with_executor_wrapper)
{
    auto val = ce::memberFunctionPointerValue(&TestObject::get);
    TestObject obj1;
    TestObject obj2;
    auto executor1 = ce::makeExecutor(obj1);
    auto executor2 = ce::makeExecutor(obj2);
    BOOST_REQUIRE_NE(executor1.m_hash, executor2.m_hash);
    auto old_hash = executor1.m_hash;
    auto ret1 = executor1.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    auto ret2 = executor2.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
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
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    executor2.exec (&TestObject::set)(4);
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);
    BOOST_REQUIRE_NE(executor1.m_hash, old_hash);

    old_hash = executor1.m_hash;
    auto ret1 = executor1.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    auto ret2 = executor2.exec(&TestObject::get)();
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == true);
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
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    // Intentially have different whitespace between the two function calls to verify a whitespace ignoreant hash is
    // used
    auto ret2 = executor2.exec(static_cast<int (TestObject::*)(int, int) const>(&TestObject::apply))(4, 5);
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == true);

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
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    // Intentially have different whitespace between the two function calls to verify a whitespace ignoreant hash is
    // used
    executor2.exec(static_cast<void (TestObject::*)(int, int&) const>(&TestObject::apply))(4, out2);
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == true);

    BOOST_REQUIRE_EQUAL(val1, val2);
    BOOST_REQUIRE_EQUAL(old_hash, executor1.m_hash);
    BOOST_REQUIRE_EQUAL(out1.m_hash, out2.m_hash);
    auto old_out_hash = out1.m_hash;
    executor1.exec (&TestObject::set)(4);
    BOOST_REQUIRE_NE(executor1.m_hash, executor2.m_hash);
    executor2.exec (&TestObject::set)(4);
    BOOST_REQUIRE_EQUAL(executor1.m_hash, executor2.m_hash);

    executor1.exec(static_cast<void (TestObject::*)(int, int&) const>(&TestObject::apply))(4, out1);
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);

    executor2.exec(static_cast<void (TestObject::*)(int, int&) const>(&TestObject::apply))(4, out2);
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == true);

    BOOST_REQUIRE_EQUAL(val1, val2);
    BOOST_REQUIRE_NE(old_out_hash, out1.m_hash);
    BOOST_REQUIRE_EQUAL(out1.m_hash, out2.m_hash);
}
*/
/*BOOST_AUTO_TEST_CASE(mutate_hashed_object)
{
    MutateOutputObject obj1;
    MutateOutputObject obj2;

    TestHashedOutputObject out1;
    TestHashedOutputObject out2;
    BOOST_REQUIRE_EQUAL(ce::getObjectHash(obj1), ce::getObjectHash(obj2));
    ce::exec (&MutateOutputObject::mutate)(obj1, ce::makeOutput(out1));
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == false);
    BOOST_REQUIRE_EQUAL(ce::getObjectHash(obj1), ce::getObjectHash(obj2));
    ce::exec (&MutateOutputObject::mutate)(obj2, ce::makeOutput(out2));
    BOOST_REQUIRE(ce::ICacheEngine::instance()->wasCacheUsedLast() == true);
    BOOST_REQUIRE_EQUAL(out1.hash, out2.hash);
    BOOST_REQUIRE_EQUAL(out1.data, out2.data);
    BOOST_REQUIRE_EQUAL(ce::getObjectHash(obj1), ce::getObjectHash(obj2));
}*/
