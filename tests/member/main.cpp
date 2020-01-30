#include "gtest/gtest.h"

#define CE_DEBUG_CACHE_USAGE
#include "object.hpp"

#include <ce/execute.hpp>
#include <ct/StringView.hpp>
#include <ct/reflect/MemberObjectPointer.hpp>

#include <cmath>
#include <iostream>
#include <math.h>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(initialize, create)
{
    ce::ICacheEngine::setEngine(ce::ICacheEngine::create());
}

TEST(member_functions, const_member_accessor)
{
    auto eng = ce::ICacheEngine::instance();
    auto hashed1 = ce::wrapHash<TestObject>();
    auto hashed2 = ce::wrapHash<TestObject>();
    EXPECT_EQ(hashed1.hash(), hashed2.hash());
    auto ret1 = ce::exec(&TestObject::get, hashed1);
    EXPECT_EQ(ret1, 0);
    ret1 = ce::exec(&TestObject::get, hashed2);
    EXPECT_EQ(eng->wasCacheUsedLast(), true);
    EXPECT_EQ(ret1, 0);
    auto hash = hashed1.hash();
    eng->exec(&TestObject::set, hashed1, 5);
    EXPECT_NE(hash, hashed1.hash());
    EXPECT_EQ(hashed1.obj.member1, 5);
    ret1 = eng->exec(&TestObject::get, hashed1);
    EXPECT_EQ(ret1, 5);
    EXPECT_NE(hashed1.hash(), hashed2.hash());
    EXPECT_EQ(eng->wasCacheUsedLast(), false);
    ret1 = eng->exec(&TestObject::get, hashed1);
    EXPECT_EQ(eng->wasCacheUsedLast(), true);
}
