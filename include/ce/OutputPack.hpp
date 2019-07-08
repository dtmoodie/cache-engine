#pragma once
#include <ce/VariadicTypedef.hpp>
#include <ce/output.hpp>
#include <ct/hash.hpp>

#include <memory>
#include <tuple>

namespace ce
{
    template <class T>
    using decay_t = typename std::decay<T>::type;

    template <class Enable, class T, class... Args>
    struct OutputPack : public OutputPack<void, Args...>
    {
        enum
        {
            OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
        };
    };

    template <class T>
    T deepCopy(T data)
    {
        return data;
    }

    template <class T>
    std::shared_ptr<T> deepCopy(std::shared_ptr<T> data)
    {
        if (data)
        {
            return std::make_shared<T>(*data);
        }
        return {};
    }

    template <class T>
    std::shared_ptr<const T> deepCopy(std::shared_ptr<const T> data)
    {
        return data;
    }

    template <class T>
    struct OutputPack<void, HashedOutput<T>>
    {
        enum
        {
            OUTPUT_COUNT = 1
        };
        using types = ct::VariadicTypedef<decay_t<T>>;

        template <class TupleType>
        static void setOutputs(size_t hash, const TupleType& result, HashedOutput<T>& out)
        {
            ce::get(out) = deepCopy(std::get<std::tuple_size<TupleType>::value - 1>(result));
            out.m_hash = ct::combineHash(hash, std::tuple_size<TupleType>::value - 1);
        }
        template <class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, HashedOutput<T>& out)
        {
            std::get<std::tuple_size<TupleType>::value - 1>(result) = deepCopy(ce::get(out));
            out.m_hash = ct::combineHash(hash, std::tuple_size<TupleType>::value - 1);
        }
    };

    template <class T, class... Args>
    struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0>::type,
                      HashedOutput<T>,
                      Args...> : public OutputPack<void, Args...>
    {
        enum
        {
            OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT + 1
        };
        using types = typename ct::Append<decay_t<T>, typename OutputPack<void, Args...>::types>::type;

        template <typename TupleType>
        static void setOutputs(size_t hash, TupleType& result, HashedOutput<T>& out, Args&... args)
        {
            ce::get(out) = deepCopy(std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT>(result));
            out.m_hash = ct::combineHash(hash, std::tuple_size<TupleType>::value - OUTPUT_COUNT);
            OutputPack<void, Args...>::setOutputs(hash, result, args...);
        }

        template <typename TupleType>
        static void saveOutputs(size_t hash, TupleType& result, HashedOutput<T>& out, Args&... args)
        {
            std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT>(result) = deepCopy(ce::get(out));
            out.m_hash = ct::combineHash(hash, std::tuple_size<TupleType>::value - OUTPUT_COUNT);
            OutputPack<void, Args...>::saveOutputs(hash, result, args...);
        }
    };

    template <class T, class... Args>
    struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type,
                      HashedOutput<T>,
                      Args...> : public OutputPack<void, Args...>
    {
        enum
        {
            OUTPUT_COUNT = 1
        };
        using types = ct::VariadicTypedef<decay_t<T>>;

        template <class TupleType>
        static void setOutputs(size_t hash, const TupleType& result, HashedOutput<T>& out, Args&... args)
        {
            ce::get(out) = deepCopy(
                std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result));
            out.m_hash =
                ct::combineHash(hash, std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1);
            OutputPack<void, Args...>::setOutputs(hash, result, args...);
        }

        template <class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, HashedOutput<T>& out, Args&... args)
        {
            std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result) =
                deepCopy(ce::get(out));
            out.m_hash =
                ct::combineHash(hash, std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1);
            OutputPack<void, Args...>::saveOutputs(hash, result, args...);
        }
    };

    template <class T, class... Args>
    struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0>::type, T, Args...>
        : public OutputPack<void, Args...>
    {
        enum
        {
            OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
        };
        using types = typename OutputPack<void, Args...>::types;

        template <typename TupleType>
        static void setOutputs(size_t hash, TupleType& result, T&, Args&... args)
        {
            OutputPack<void, Args...>::setOutputs(hash, result, args...);
        }

        template <typename TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T&, Args&... args)
        {
            OutputPack<void, Args...>::saveOutputs(hash, result, args...);
        }
    };

    template <class T>
    struct OutputPack<void, T>
    {
        enum
        {
            OUTPUT_COUNT = 0
        };
        template <class TupleType>
        static void setOutputs(size_t, TupleType&, T&)
        {
        }
        template <class TupleType>
        static void saveOutputs(size_t, TupleType&, T&)
        {
        }
    };

    template <class T, class... Args>
    struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type, T, Args...>
        : public OutputPack<void, Args...>
    {
        enum
        {
            OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT
        };
        template <class TupleType>
        static void setOutputs(size_t hash, TupleType& result, T&, Args&... args)
        {
            OutputPack<void, Args...>::setOutputs(hash, result, args...);
        }
        template <class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T&, Args&... args)
        {
            OutputPack<void, Args...>::saveOutputs(hash, result, args...);
        }
    };
}
