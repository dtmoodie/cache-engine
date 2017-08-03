#include <ce/OutputPack.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

//#define STATIC_ASSERT_MACRO(expr) static_assert(expr, #expr)

int main(int argc, char** argv){
    static_assert(ce::OutputPack<void, int(int), int>::OUTPUT_COUNT == 0, "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, int>::OUTPUT_COUNT == 1, "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, ce::HashedOutput<int>>::OUTPUT_COUNT == 2, "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, ce::HashedInput<int>, ce::HashedOutput<int>>::OUTPUT_COUNT == 2, "asdf");

    return 0;
}