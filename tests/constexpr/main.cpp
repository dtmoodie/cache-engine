#include <ce/OutputPack.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/type_traits.hpp>
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

namespace ce{
namespace type_traits{
namespace argument_specializations {
    template<class T>
    constexpr int countInputsImpl(ce::HashedInput<T>* ptr = nullptr) {
        return 1;
    }

    template<class T>
    constexpr int countInputsImpl(ce::HashedOutput<T>* ptr = nullptr) {
        return 1;
    }
}
}
}
namespace as = ce::type_traits::argument_specializations;
void foo1(int, int, int){
}

void foo2(ce::HashedInput<int>, int, int) {
}

template<class T>
constexpr void outputTypeImpl(T* ptr = nullptr){
}
template<class T>
constexpr T outputTypeImpl(ce::HashedOutput<T>* ptr = nullptr){
    return {};
}



template<class T, class... Args>
constexpr auto outputType(std::tuple<T, Args...>* ptr = nullptr) -> 
    decltype(as::appendTypes(decltype(as::outputTypeImpl(static_cast<T*>(nullptr)))*(0), decltype(as::outputTypeImpl<Args...>()>())*(0))) {
    return {};
}



int main(int argc, char** argv){
    static_assert(as::countInputs<int, int, int>() == 0, "asdf");
    static_assert(as::countOutputs<int,int,int>() == 0, "asdf");
    static_assert(as::countInputs<ce::HashedInput<float>>() == 1, "asdf");
    static_assert(as::countInputs<ce::HashedInput<float>, int, int>() == 1, "asdf");
    static_assert(as::countInputs<ce::HashedInput<float>, int, int, ce::HashedInput<float>>() == 2, "asdf");
    static_assert(as::countInputs<ce::HashedInput<float>, int, int, ce::HashedInput<float>>() == 2, "asdf");

    static_assert(as::hasDefaultSpecialization<int>() == 1, "asdf");
    static_assert(as::hasDefaultSpecialization<float>() == 1, "asdf");
    static_assert(as::hasDefaultSpecialization<double>() == 1, "asdf");
    static_assert(as::hasDefaultSpecialization<std::tuple<int,int>>() == 1, "asdf");

    static_assert(as::hasDefaultSpecialization<ce::HashedInput<float>>() == 0, "asdf");
    static_assert(as::hasDefaultSpecialization<ce::HashedOutput<float>>() == 0, "asdf");

    static_assert(as::countOutputs(&foo1) == 0, "asdf");
    static_assert(as::countOutputs(&foo2) == 0, "asdf");
    static_assert(as::countInputs(&foo1) == 0, "asdf");
    static_assert(as::countInputs(&foo2) == 1, "asdf");
       
    decltype(as::appendTypes<void, int, int>()) test;
    decltype(as::appendTypes<int, float, void, int, int>()) test2;
    decltype(as::appendTypes<int, float, void, ce::HashedOutput<int>, int>()) test3;

    outputTypeImpl(static_cast<int*>(nullptr));
    
    
    decltype(as::appendTypes<decltype(outputTypeImpl<int>()), decltype(outputTypeImpl<float>())>()) testoutput;
    as::AppendTypes<int, float> testoutput2;
    return 0;
}