#include <ce/OutputPack.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/type_traits.hpp>
#include <ce/execute.hpp>

namespace as = ce::type_traits::argument_specializations;

void foo(int in, float& out){
    out = 20 * in;
}

template<int OUT_COUNT, class... Args, class R, class ... FArgs>
constexpr void argIterTester(R(*func)(FArgs...)){
    typedef ce::OutputPack<R(FArgs...), Args...> AI;
    static_assert(AI::OUTPUT_COUNT == OUT_COUNT, "asdf");
}
template<class SaveType, class... Args, class R, class ... FArgs>
constexpr void assertSavetype(R(*func)(FArgs...)){
    typedef ce::OutputPack<R(FArgs...), Args...> AI;
    static_assert(std::is_same<SaveType, AI::SaveTuple>::value, "asdf");
}

template<class... Args, class R, class ... FArgs>
constexpr void printSavetype(R(*func)(FArgs...)) {
    typedef ce::OutputPack<R(FArgs...), Args...> AI;
    AI::debugPrint();
}

int main(int argc, char** argv){
    {
        typedef ce::OutputPack<void(float), float> AI;
        static_assert(AI::IS_OUTPUT == 0, "adf");
        static_assert(AI::OUTPUT_COUNT == 0, "adf");
    }
    {
        typedef ce::OutputPack<void(ce::HashedOutput<float>), ce::HashedOutput<float>> AI;
        static_assert(AI::IS_OUTPUT == 1, "adsf");
        static_assert(std::is_same<AI::SaveTuple, std::tuple<float>>::value, "adsf");
    }
    {
        argIterTester<1, int, ce::HashedOutput<float>>(foo);
        //assertSavetype<std::tuple<float>, int, ce::HashedOutput<float>>(foo);
        printSavetype<int, ce::HashedOutput<float>>(foo);
    }
    return 0;
}