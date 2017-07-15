#pragma once
#include <ce/VariadicTypedef.hpp>

template<class Enable, class T, class...Args> struct OutputPack : public OutputPack<void, Args...> {
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
    };
};

template<class T> struct OutputPack<void, T> {
    enum {
        OUTPUT_COUNT = 0
    };
};

template<class T> struct OutputPack<void, HashedOutput<T>> {
    enum {
        OUTPUT_COUNT = 1
    };
    typedef variadic_typedef<T> types;

    template<class TupleType>
    static void setOutputs(TupleType& result, HashedOutput<T>& out) {
        get(out) = std::get<std::tuple_size<TupleType>::value - 1>(result);
    }
    template<class TupleType>
    static void saveOutputs(TupleType& result, HashedOutput<T>& out) {
        std::get<std::tuple_size<TupleType>::value - 1>(result) = get(out);
    }
};

template<class T, class ... Args> struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0>::type, HashedOutput<T>, Args...> : public OutputPack<void, Args...> {
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT + 1
    };
    typedef typename append_to_tupple<T, typename OutputPack<void, Args...>::types>::type types;

    template<typename TupleType>
    static void setOutputs(TupleType& result, HashedOutput<T>& out, Args&... args) {
        get(out) = std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT >(result);
        OutputPack<void, Args...>::setOutputs(result, args...);
    }

    template<typename TupleType>
    static void saveOutputs(TupleType& result, HashedOutput<T>& out, Args&... args) {
        std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT >(result) = get(out);
        OutputPack<void, Args...>::saveOutputs(result, args...);
    }
};

template<class T, class ... Args>
struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type, HashedOutput<T>, Args...> : public OutputPack<void, Args...> {
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT + 1
    };
    typedef variadic_typedef<T> types;

    template<class TupleType>
    static void setOutputs(TupleType& result, HashedOutput<T>& out, Args&... args) {
        get(out) = std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result);
        OutputPack<void, Args...>::setOutputs(result, args...);
    }

    template<class TupleType>
    static void saveOutputs(TupleType& result, HashedOutput<T>& out, Args&... args) {
        std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result) = get(out);
        OutputPack<void, Args...>::saveOutputs(result, args...);
    }
};

template<class T, class ... Args>
struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0>::type, T, Args...> : public OutputPack<void, Args...> {
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
    };
    typedef typename OutputPack<void, Args...>::types types;
    typedef typename convert_in_tuple<types>::type TupleType;

    static void setOutputs(TupleType& result, T&, Args&... args) {
        OutputPack<void, Args...>::setOutputs(result, args...);
    }
    static void saveOutputs(TupleType& result, T&, Args&... args) {
        OutputPack<void, Args...>::saveOutputs(result, args...);
    }
};

template<class T, class ... Args>
struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type, T, Args...> : public OutputPack<void, Args...> {
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
    };
};