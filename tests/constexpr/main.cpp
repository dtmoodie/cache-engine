#include <ce/OutputPack.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

//#define STATIC_ASSERT_MACRO(expr) static_assert(expr, #expr)

// Test specializing on function signature

/*namespace ce{
template<class T1, class T, class R, class ... FArgs, class ... Args>
    struct OutputPack<typename std::enable_if<!OutputPack<void, Args...>::OUTPUT_COUNT && !outputDetector<T>()>::type, R(T1, FArgs...), T, Args...> : public OutputPack<void, R(FArgs...), Args...> {
        enum {
            OUTPUT_COUNT = OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT
        };
        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& result, T& out, Args&... args) {
            OutputPack<void, R(FArgs...), Args...>::setOutputs(hash, result, args...);
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T& out, Args&... args) {
            OutputPack<void, R(FArgs...), Args...>::saveOutputs(hash, result, args...);
        };
    };
}*/

int main(int argc, char** argv){
    static_assert(ce::OutputPack<void, int(int), int>::OUTPUT_COUNT == 0, "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, int>::OUTPUT_COUNT == 1, "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, ce::HashedOutput<int>>::OUTPUT_COUNT == 2, "asdf");
    static_assert(!ce::outputDetector<float>(), "asdf");
    static_assert(ce::outputDetector<ce::HashedOutput<float>>(), "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, ce::HashedInput<int>, ce::HashedOutput<int>>::OUTPUT_COUNT == 2, "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, ce::HashedInput<int>, ce::HashedOutput<int>, ce::HashedInput<float>>::OUTPUT_COUNT == 2, "asdf");
    static_assert(ce::OutputPack<void, int(int), ce::HashedOutput<int>, ce::HashedInput<int>, ce::HashedOutput<int>, ce::HashedInput<float>, ce::HashedOutput<float>>::OUTPUT_COUNT == 3, "asdf");


    return 0;
}