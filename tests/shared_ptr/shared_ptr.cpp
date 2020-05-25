#include "gtest/gtest.h"

#include <ce/shared_ptr.hpp>

using namespace ce;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(shared_ptr, copy_on_write)
{
    shared_ptr<float> ptr(std::make_shared<float>(5.0));
    ASSERT_TRUE(ptr.getCopier());
    shared_ptr<const float> cptr(ptr);
    ASSERT_TRUE(cptr.getCopier());
    EXPECT_EQ(ptr.get(), cptr.get());
    *ptr = 6;
    EXPECT_EQ(*ptr, *cptr);
    shared_ptr<float> cow(cptr);
    ASSERT_TRUE(cow.getCopier());
    EXPECT_EQ(static_cast<const shared_ptr<float>&>(cow).get(), cptr.get());
    auto cow_ptr = cow.get();
    EXPECT_NE(cow_ptr, cptr.get());
    *cow = 10;
    EXPECT_NE(*ptr, 10);
    EXPECT_EQ(*cow, 10);
    EXPECT_EQ(cow.get(), cow_ptr);
}

TEST(shared_ptr, const_handle_no_copy)
{
    // This tests a special case where a mutable handle is created from a const handle, but the mutable handle is the
    // only
    // shared ptr with a reference to the data.
    // Since no one else cares for the data, we can be certain that
    float* raw = nullptr;
    shared_ptr<float> test;
    {
        shared_ptr<const float> cptr;
        {
            shared_ptr<float> original(std::make_shared<float>(5.0));
            raw = original.get();
            cptr = original;
        }
        test = cptr;
    }

    EXPECT_EQ(test.get(), raw);
}

TEST(shared_ptr, const_handle_copy)
{
    float* raw = nullptr;
    shared_ptr<float> test;
    shared_ptr<const float> cptr;
    {
        {
            shared_ptr<float> original(std::make_shared<float>(5.0));
            raw = original.get();
            cptr = original;
        }
        test = cptr;
    }

    EXPECT_NE(test.get(), raw);
}
