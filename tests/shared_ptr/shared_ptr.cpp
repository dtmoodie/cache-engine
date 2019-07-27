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
