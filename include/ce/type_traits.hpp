#pragma once
#include <utility>
namespace ce{
namespace type_traits{
namespace argument_specializations{

    template<class T>
    constexpr bool hasDefaultSpecialization(const T* ptr = nullptr){
        return (!hasOutputSpecialization<T>() && !hasInputSpecialization<T>());
    }

    template<class T1, class T2>
    constexpr bool hasDefaultSpecialization(const T1* ptr = nullptr, const T2* ptr2 = nullptr){
        return hasDefaultSpecialization<T1>() && hasDefaultSpecialization<T2>();
    }

    template<class T>
    constexpr bool hasOutputSpecialization(const T* ptr = nullptr){
        return countOutputs<T>() != 0;
    }

    template<class T>
    constexpr bool hasInputSpecialization(const T* ptr = nullptr){
        return countInputs<T>() != 0;
    }

    template<class T>
    constexpr int countInputsImpl(const T* ptr = nullptr){
        return 0;
    }

    template<class T, class ... Args>
    constexpr int countInputsImpl(std::tuple<T, Args...>* ptr = nullptr){
        return countInputsImpl(static_cast<T*>(nullptr)) +
            countInputsImpl(static_cast<std::tuple<Args...>*>(nullptr));
    }
    
    template<class T, class ... Args>
    constexpr int countInputs() {
        return countInputsImpl(static_cast<std::tuple<T, Args...>*>(nullptr));
    }

    template<class R, class ... Args>
    constexpr int countInputs(R(*func)(Args...)) {
        return countInputsImpl(static_cast<std::tuple<R, Args...>*>(nullptr));
    }

    template<class T>
    constexpr int countOutputsImpl(const T* ptr = nullptr){
        return 0;
    }

    template<class T, class ... Args>
    constexpr int countOutputsImpl(std::tuple<T, Args...>* ptr = nullptr){
        return countOutputsImpl(static_cast<T*>(nullptr)) +
            countOutputsImpl(static_cast<std::tuple<Args...>*>(nullptr));
    }

    template<class T, class ... Args>
    constexpr int countOutputs() {
        return countOutputsImpl(static_cast<std::tuple<T, Args...>*>(nullptr));
    }

    template<class R, class ... Args>
    constexpr int countOutputs(R(*func)(Args...)){
        return countOutputsImpl(static_cast<std::tuple<R, Args...>*>(nullptr));
    }

    template<class T>
    ce::variadic_typedef<T> appendTypes(T* ptr = nullptr){
        return {};
    }

    template<class T1, class T2>
    ce::variadic_typedef<T1, T2> appendTypes(T1* ptr1, T2* ptr2){
        return {};
    }

    template<class T, class ... Args> 
    typename ce::append_to_tupple<T, decltype(appendTypes<Args...>())>::type appendTypes(T* ptr = nullptr, std::tuple<Args...>* args = nullptr){
        return {};
    }

    template<class T, class ... Args>
    decltype(appendTypes<Args...>()) appendTypes(std::enable_if_t<std::is_same_v<T, void>>* ptr = nullptr, std::tuple<Args...>* args = nullptr) {
        return{};
    }

    template<class T1, class T2>
    using AppendTypes = std::remove_pointer_t<decltype(appendTypes(static_cast<T1*>(nullptr), static_cast<T2*>(nullptr)))>;

} // namespace ce::type_traits::argument_specializations
} // namespace ce::type_traits
} // namespace ce

#include "detail/type_traits.hpp"