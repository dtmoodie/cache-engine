#include <ce/Executor.hpp>
#include <ce/execute.hpp>
#include <ce/function_traits.hpp>
#include <ce/utils.hpp>

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
using namespace ce;

template <class T, class R, class... Args>
R exec(R (T::*func)(Args...), T* This, Args... args)
{
    return (*This.*func)(args...);
}

int foo1(int value)
{
    return 5;
}

int foo2(int value)
{
    return 6;
}

int foo3(int value)
{
    return value;
}

struct bar
{
    static int foo(int value)
    {
        return value;
    }
    int member(int value) const
    {
        return value;
    }
    void setter(int& value, int& value2)
    {
        value = 5;
        value2 = 10;
    }
    void apply(int value, int& output)
    {
        output = value;
    }
    void member2(int value1, int value2) const
    {
    }
    void overloaded(int)
    {
    }
    void overloaded(float)
    {
    }
    void set(int value)
    {
        m_member = value;
    }

    int get()
    {
        return m_member;
    }
    static int staticFunc(int value)
    {
        return 5;
    }
    int m_member = 0;
};

// unfortunately we need to use this macro to deal with not being able to hash member function pointers on windows. :(
#include <vector>
void foo(std::vector<int>&& vec)
{
    std::vector<int> other(std::move(vec));
}
void foo(int&& data)
{
}
int main()
{
    int test;
    std::vector<int> testvec;
    for (int i = 0; i < 100; ++i)
        testvec.push_back(i);
    foo(std::move(test));
    foo(std::move(testvec));
    std::cout << "Testing OutputPack detection\n";
    std::cout << OutputPack<int>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<int, int>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<HashedInput<int>, HashedInput<int>>::OUTPUT_COUNT << " == 0\n";
    std::cout << OutputPack<HashedOutput<int>>::OUTPUT_COUNT << " == 1\n";
    std::cout << OutputPack<HashedOutput<int>, HashedOutput<int>>::OUTPUT_COUNT << " == 2\n";

    // static_assert(!ce::function_traits::is_const(&bar::member), "member is non const");
    // static_assert(ce::function_traits::is_const(&bar::member2), "member is const");
    auto engine = ICacheEngine::create();
    ICacheEngine::setEngine(engine);
    auto hashed = makeInput<int>(5);
    std::cout << exec(foo1, 0) << std::endl;
    std::cout << exec(foo1, 0) << std::endl;
    std::cout << exec(foo2, 0) << std::endl;
    std::cout << exec(foo3, 10) << std::endl;
    std::cout << exec(bar::foo, 20) << std::endl;
    std::cout << exec(&bar::staticFunc, 0) << std::endl;
    bar cls;
    auto executor = makeExecutor(cls);
    std::cout << exec(&bar::member)(executor, hashed) << std::endl;
    std::cout << exec(&bar::member)(executor, hashed) << std::endl;
    int value1 = 20, value2 = 10;
    exec (&bar::setter)(executor, makeOutput(value1), makeOutput(value2));
    assert(value1 == 5);
    assert(value2 == 10);
    value1 = 100;
    value2 = 200;

    exec (&bar::setter)(executor, makeOutput(value2), makeOutput(value1));
    assert(value1 == 10);
    assert(value2 == 5);

    exec (&bar::apply)(executor, 10, makeOutput(value1));
    exec (&bar::apply)(executor, 10, makeOutput(value1));
    exec (&bar::apply)(executor, 5, makeOutput(value1));
    std::cout << "Testing setters and getters" << std::endl;
    int ret = exec(&bar::get)(executor);
    std::cout << ret << std::endl;
    // executor.set(&bar::set, 15);
    std::cout << exec(&bar::get)(executor) << std::endl;
    std::cout << exec(&bar::get)(executor) << std::endl;

    exec (&bar::member2)(executor, 0, 1);
    exec(static_cast<void (bar::*)(int)>(&bar::overloaded))(executor, 1);
    exec(static_cast<void (bar::*)(float)>(&bar::overloaded))(executor, 1.f);

    return 0;
}
