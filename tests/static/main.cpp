
#define CE_DEBUG_CACHE_USAGE
#include "gtest/gtest.h"
#include <algorithm>

#include <ce/execute.hpp>
#include <ct/StringView.hpp>
#include <ct/static_asserts.hpp>

#include <cmath>
#include <math.h>
#include <vector>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void foo0()
{
}

int foo1(int value)
{
    return value * 2;
}

double foo2(int value1, double value2)
{
    return std::pow(value2, value1);
}

double foo3(double a, double x, double b)
{
    return a * x + b;
}

void foo4(double a, double x, double b, double& y)
{
    y = a * x + b;
}

double foo5(int value1, int value2, int& out1, double& out2)
{
    out1 = value1 * value2;
    out2 = value1 + value2;
    return out1 + out2;
}

void foo6(int input, int* output, int opt1, int opt2, int opt3)
{
    *output = input * opt1 + opt2 + opt3;
}

void foo7(const std::vector<int>& input, double& output)
{
    output = 0;
    for (int val : input)
    {
        output += val;
    }
}

TEST(initialize, create)
{
    ce::ICacheEngine::setEngine(ce::ICacheEngine::create());
}

void format(char* buf, int val, int size)
{
    int written_chars = snprintf(buf, size, "%d", val);
    for (int i = written_chars; i < size; ++i)
    {
        buf[i] = ' ';
    }
}

void format(char* buf, long val, int size)
{
    snprintf(buf, size, "%ld", val);
    bool null_found = false;
    for (int i = 0; i < size; ++i)
    {
        if (buf[i] == '\n')
        {
            buf[i] = ' ';
            null_found = true;
        }
        if (null_found)
        {
            buf[i] = ' ';
        }
    }
}

void format(char* buf, float val, size_t size)
{
    snprintf(buf, size, "%f", val);
    bool null_found = false;
    for (size_t i = 0; i < size; ++i)
    {
        if (buf[i] == '\n')
        {
            buf[i] = ' ';
            null_found = true;
        }
        if (null_found)
        {
            buf[i] = ' ';
        }
    }
}

void formatImpl(const char* fmt, char* buf)
{
    char* ptr = buf;
    while (*fmt != '\0')
    {
        *ptr = *fmt;
        ++ptr;
        ++fmt;
    }
}

template <class FormatArg, class... FormatArgs>
void formatImpl(const char* fmt, char* buf, FormatArg&& arg, FormatArgs&&... args)
{
    char* ptr = buf;
    while (*fmt != '\0')
    {
        if (*fmt == '{')
        {
            auto pos = ct::findFirst(fmt, '}');
            int len = ct::stoiRange(fmt + 1, fmt + pos - 1);
            format(ptr, std::forward<FormatArg>(arg), len);
            return formatImpl(fmt + pos, ptr + len, std::forward<FormatArgs>(args)...);
        }
        *ptr = *fmt;
        ++ptr;
        ++fmt;
    }
}

template <size_t StrLen, class... FormatArgs>
const char* format(const char* fmt, FormatArgs&&... args)
{
    char buf[StrLen];
    formatImpl(fmt, buf, std::forward<FormatArgs>(args)...);
    return buf;
}

#define fmt(spec, ...) format<ct::formatStringSize(spec)>(spec, __VA_ARGS__)

struct EmbeddedHash : public ce::HashedBase
{
    using type = EmbeddedHash;
    int val;
};

EmbeddedHash create(int x, int y)
{
    EmbeddedHash out;
    out.val = x * y;
    return out;
}

std::shared_ptr<const EmbeddedHash> createShared(int x, int y)
{
    auto out = std::make_shared<EmbeddedHash>();
    out->val = x * y;
    return out;
}

int scaleData(float& data, float mul, float bias)
{
    data = data * mul + bias;
    return 1;
}

static long GMULL = 0x9E3779B97F4A7C15L;
static long GDIVL = 0xF1DE83E19937733DL;
static long GADDL = 0x0123456789ABCDEFL;
uint64_t golden64fwd(uint64_t key)
{
    key *= GMULL;
    key += GADDL;
    return key;
}

uint64_t golden64rev(uint64_t key)
{
    key -= GADDL;
    key *= GDIVL;
    return key;
}

uint64_t combine(uint64_t v0, uint64_t v1)
{
    return v0 + v1;
}

uint64_t decouple(uint64_t v0, uint64_t v1)
{
    return v0 - v1;
}

TEST(test, foo1)
{
    auto val = golden64fwd(ce::generateHash(&foo1));
    auto rev = golden64rev(val);

    EXPECT_EQ(rev, ce::generateHash(&foo1));
    // auto spec_size = ct::specifierSize<2>("asdf {3} {4}");
    // fmt("asdf {3} {4}", 4, 5);
    EXPECT_EQ(ce::exec(foo1, 10), foo1(10));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    EXPECT_EQ(ce::exec(foo1, 11), foo1(11));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    // Get cached results
    EXPECT_EQ(ce::exec(foo1, 10), foo1(10));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(ce::exec(foo1, 11), foo1(11));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
}

TEST(test, foo2)
{
    EXPECT_EQ(ce::exec(foo2, 10, 5.0), foo2(10, 5.0));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    EXPECT_EQ(ce::exec(foo2, 11, 6.0), foo2(11, 6.0));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    // Get cached results
    EXPECT_EQ(ce::exec(foo2, 10, 5.0), foo2(10, 5.0));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(ce::exec(foo2, 11, 6.0), foo2(11, 6.0));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
}

TEST(test, foo3)
{
    EXPECT_EQ(ce::exec(foo3, 2, 3, 4), foo3(2, 3, 4));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    EXPECT_EQ(ce::exec(foo3, 5, 6, 7), foo3(5, 6, 7));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    // Get cached results
    EXPECT_EQ(ce::exec(foo3, 2, 3, 4), foo3(2, 3, 4));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(ce::exec(foo3, 5, 6, 7), foo3(5, 6, 7));
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
}

TEST(test, scale_vector)
{
    float data = 10;
    auto val = ce::exec(&scaleData, ce::makeOutput(data), 2.0, 3.0);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    EXPECT_EQ(val, 1);
    EXPECT_EQ(data, 10 * 2 + 3);
    data = 10;
    val = ce::exec(&scaleData, ce::makeOutput(data), 2.0, 3.0);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(val, 1);
    EXPECT_EQ(data, 10 * 2 + 3);
}

TEST(test, embedded_hash)
{
    auto eng = ce::ICacheEngine::instance();
    auto ret = eng->exec(&create, 4, 5);
    ct::StaticEqualTypes<decltype(ret), EmbeddedHash>{};
    EXPECT_NE(ret.hash(), 0);
    EXPECT_EQ(ret.val, 20);

    const auto hash = ret.hash();
    ret = eng->exec(&create, 4, 5);
    EXPECT_EQ(ret.hash(), hash);
    EXPECT_EQ(eng->wasCacheUsedLast(), true);
    EXPECT_EQ(ret.val, 20);
    ret = eng->exec(&create, 5, 5);
    EXPECT_EQ(eng->wasCacheUsedLast(), false);
    EXPECT_NE(ret.hash(), hash);
    EXPECT_EQ(ret.val, 25);

    auto shared_ret = eng->exec(&createShared, 4, 5);
    EXPECT_EQ(eng->wasCacheUsedLast(), false);
    shared_ret = eng->exec(&createShared, 4, 5);
    EXPECT_EQ(eng->wasCacheUsedLast(), true);
}

TEST(test, foo4)
{
    double result1, result2 = 0.0;
    auto out1 = ce::makeOutput(result1);
    ce::exec(foo4, 2, 3, 4, out1);
    foo4(2, 3, 4, result2);
    EXPECT_EQ(result1, result2);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);

    auto out2 = ce::makeOutput(result1);
    ce::exec(foo4, 5, 6, 7, out2);
    foo4(5, 6, 7, result2);
    EXPECT_EQ(result1, result2);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    EXPECT_NE(out1.hash(), out2.hash());

    // Get cached results
    auto out3 = ce::makeOutput(result1);
    ce::exec(foo4, 2, 3, 4, out3);
    foo4(2, 3, 4, result2);
    EXPECT_EQ(out1.hash(), out3.hash());
    EXPECT_EQ(result1, result2);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);

    auto out4 = ce::makeOutput(result1);
    ce::exec(foo4, 5, 6, 7, out4);
    foo4(5, 6, 7, result2);
    EXPECT_EQ(out2.hash(), out4.hash());
    EXPECT_EQ(result1, result2);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
}

TEST(test, chain)
{
    auto result1 = ce::exec(foo2, 5, 10);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    auto result2 = ce::exec(foo3, ce::makeInput(result1), 4, 5);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);

    auto result3 = ce::exec(foo2, 5, 10);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(result1.hash(), result3.hash());
    auto result4 = ce::exec(foo3, ce::makeInput(result3), 4, 5);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(result2.hash(), result4.hash());
}

TEST(test, multi_out)
{
    int val1 = 0.0;
    double val2 = 0.0;
    auto out1 = ce::makeOutput(val1);
    auto out2 = ce::makeOutput(val2);
    auto ret1 = ce::exec(foo5, 4, 5, out1, out2);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    EXPECT_NE(out1.hash(), out2.hash());
    EXPECT_NE(out1.hash(), ret1.hash());
    EXPECT_EQ(out1.data, 4 * 5);
    EXPECT_EQ(out2.data, 4 + 5);
    EXPECT_EQ(ret1, out1.data + out2.data);

    auto out3 = ce::makeOutput(val1);
    auto out4 = ce::makeOutput(val2);
    auto ret2 = ce::exec(foo5, 4, 5, out3, out4);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(out3.hash(), out1.hash());
    EXPECT_EQ(out4.hash(), out2.hash());
    EXPECT_EQ(ret1.hash(), ret2.hash());
    EXPECT_EQ(out3.data, 4 * 5);
    EXPECT_EQ(out4.data, 4 + 5);
    EXPECT_EQ(ret2, out3.data + out4.data);
    int in = 5;
    int out = 0;

    ce::exec(foo6, in, ce::makeOutput(&out), 4, 5, 6);
}

TEST(test, const_ref_input)
{
    auto input = ce::makeInput<std::vector<int>>(100, 5);
    double val = 0.0;
    auto out1 = ce::makeOutput(val);
    ce::exec(&foo7, input, out1);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    EXPECT_EQ(val, 100 * 5);
    auto out2 = ce::makeOutput(val);
    ce::exec(&foo7, input, out2);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
    EXPECT_EQ(out1.hash(), out2.hash());
    EXPECT_NE(out1.hash(), 0);
}

void foo7Wrapper(const std::vector<int>& input, double& output)
{
    ce::exec(&foo7, ce::wrapInput(input), ce::makeOutput(output));
}

// Less efficient because a hash is calculated on the whole vector every call inside of ce::wrapInput
// In most cases it is preferrable to use makeInput
TEST(test, wrapper_function)
{
    std::vector<int> in(100, 5);
    double out = 0.0;
    foo7Wrapper(in, out);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), false);
    foo7Wrapper(in, out);
    EXPECT_EQ(ce::ICacheEngine::instance()->wasCacheUsedLast(), true);
}
