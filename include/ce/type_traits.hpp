#pragma once
#include <utility>
namespace ce{
namespace type_traits{
namespace argument_specializations{

    template<class T, bool Select>
    struct SaveTypeImpl{
    };

    template<class T>
    struct SaveTypeImpl<T, true> {
        typedef T type;
    };

    template<class T>
    struct SaveTypeImpl<T, false> {
        typedef void type;
    };


    template<class T1, class T2>
    struct SelectSaveType{};

    template<class T>
    struct SelectSaveType<T, void>{
        typedef T type;
    };

    template<class T>
    struct SelectSaveType<void, T> {
        typedef T type;
    };

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

    template<class ... Args>
    constexpr int countInputs(void(*func)(Args...)) {
        return countInputsImpl(static_cast<std::tuple<Args...>*>(nullptr));
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
        return countOutputsImpl(static_cast<std::tuple<Args...>*>(nullptr)) + 1;
    }

    template<class ... Args>
    constexpr int countOutputs(void(*func)(Args...)) {
        return countOutputsImpl(static_cast<std::tuple<Args...>*>(nullptr));
    }

    template<class T>
    struct SaveType {
        typedef typename SaveTypeImpl<T, hasOutputSpecialization<T>()>::type type;
    };

    template<class T>
    using SaveType_t = typename SaveType<T>::type;

    template<class T, class F>
    using enable_if_output = typename std::enable_if<hasOutputSpecialization<T>() || hasOutputSpecialization<F>()>::type;

    template<class T, class F>
    using enable_if_not_output = typename std::enable_if<(!hasOutputSpecialization<T>() && !hasOutputSpecialization<F>())>::type;

} // namespace ce::type_traits::argument_specializations
} // namespace ce::type_traits
} // namespace ce