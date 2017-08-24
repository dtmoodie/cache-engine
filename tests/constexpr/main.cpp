#include <ce/OutputPack.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/type_traits.hpp>

namespace ce{
namespace type_traits{
namespace argument_specializations {
    template<class T>
    constexpr int countInputsImpl(const ce::HashedInput<T>* ptr = nullptr) {
        return 1;
    }

    template<class T>
    constexpr int countOutputsImpl(const ce::HashedOutput<T>* ptr = nullptr) {
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
    as::enable_if_not_output<int, int>* ptr = nullptr;
    //static_assert(ce::OutputPack<void, void(int), int>::OUTPUT_COUNT == 0, "asdf");
    as::hasOutputSpecialization<ce::HashedOutput<int>>();
    static_assert(as::hasOutputSpecialization<ce::HashedOutput<int>>() == true, "asdf");
    static_assert((!as::hasOutputSpecialization<int>() && !as::hasOutputSpecialization<ce::HashedOutput<int>>()) == false, "asdf");
    static_assert((as::hasOutputSpecialization<int>() || as::hasOutputSpecialization<ce::HashedOutput<int>>()) == true, "asdf");
    {
        as::enable_if_output<int, ce::HashedOutput<int>>* ptr = nullptr;
        ce::ArgumentIterator<int(int), int>::printTypes();
        ce::ArgumentIterator<int(int, float), int, float>::printTypes();
        ce::ArgumentIterator<int(int, float), int, float>::printOutputTypes();
        ce::ArgumentIterator<int(int, ce::HashedOutput<float>), int, float>::printOutputTypes();
        ce::ArgumentIterator<int(int, float), int, ce::HashedOutput<int>>::printOutputTypes();
        static_assert(ce::ArgumentIterator<int(int, ce::HashedOutput<float>), int, float>::OUTPUT_COUNT == 1, "asdf");
    }
    //static_assert(ce::OutputPack<void, void(int), ce::HashedOutput<int>>::OUTPUT_COUNT == 1, "asdf");
    return 0;
}