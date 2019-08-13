#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CE::StaticFunctionTest"

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>

#include <ce/shared_ptr.hpp>

using namespace ce;

BOOST_AUTO_TEST_CASE(copy_on_write)
{
    shared_ptr<float> ptr(std::make_shared<float>(5.0));
    shared_ptr<const float> cptr(ptr);
    BOOST_REQUIRE_EQUAL(ptr.get(), cptr.get());
    *ptr = 6;
    BOOST_REQUIRE_EQUAL(*ptr, *cptr);
    shared_ptr<float> cow(cptr);
    BOOST_REQUIRE_EQUAL(static_cast<const shared_ptr<float>&>(cow).get(), cptr.get());
    auto cow_ptr = cow.get();
    BOOST_REQUIRE_NE(cow_ptr, cptr.get());
    *cow = 10;
    BOOST_REQUIRE_NE(*ptr, 10);
    BOOST_REQUIRE_EQUAL(*cow, 10);
    BOOST_REQUIRE_EQUAL(cow.get(), cow_ptr);
}


BOOST_AUTO_TEST_CASE(const_handle_no_copy)
{
    // This tests a special case where a mutable handle is created from a const handle, but the mutable handle is the only
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

    BOOST_REQUIRE_EQUAL(test.get(), raw);
}

BOOST_AUTO_TEST_CASE(const_handle_copy)
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

    BOOST_REQUIRE_NE(test.get(), raw);
}
