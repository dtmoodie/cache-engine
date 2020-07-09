#include <ce/OutputPack.hpp>
#include <ct/static_asserts.hpp>

void foo(int& out, int in)
{
}

int main()
{
    // int out;
    // auto storage = ce::makeOutputStorage(&foo, ce::makeOutput(out), 5);
    // ct::StaticEqualTypes<decltype(storage), std::tuple<int>>{};
}
