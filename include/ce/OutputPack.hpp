#pragma once
#include "IResult.hpp"

#include <ce/VariadicTypedef.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ct/hash.hpp>

#include <memory>
#include <tuple>

namespace ce
{
    template <class T>
    using decay_t = typename std::decay<T>::type;

    template <class T, class... Args>
    struct OutputPackImpl;

    template <class T, class E = void, int32_t I = 10>
    struct OutputParameterHandler;

    template <class T>
    struct OutputParameterHandler<T, void, 0>
    {
        static constexpr const bool IS_OUTPUT = false;
        using result_storage_type = ct::VariadicTypedef<>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t, const TupleType&, T&, Args&&...)
        {
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t, TupleType&, T&, Args&&...)
        {
        }
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

    template <class T, class E, int32_t I>
    struct OutputParameterHandler : public OutputParameterHandler<T, E, I - 1>
    {
    };

    template <class T>
    struct OutputParameterHandler<HashedOutput<T>, void, 3>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<decay_t<T>>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t hash, const TupleType& result, HashedOutput<T>& out, Args&&...)
        {
            ce::get(out) = deepCopy(std::get<IDX>(result));
            out.setHash(ct::combineHash(hash, IDX));
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, HashedOutput<T>& out, Args&&...)
        {
            std::get<IDX>(result) = deepCopy(ce::get(out));
            out.setHash(ct::combineHash(hash, IDX));
        }
    };

    template <class T>
    struct OutputParameterHandler<HashedInput<T>, void, 4>
    {
        static constexpr const bool IS_OUTPUT = false;
        using result_storage_type = ct::VariadicTypedef<>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t, const TupleType&, HashedInput<T>&, Args&&...)
        {
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t, TupleType&, HashedInput<T>&, Args&&...)
        {
        }
    };

    template <class T>
    struct OutputParameterHandler<
        T,
        typename std::enable_if<std::is_base_of<HashedBase, T>::value && !std::is_const<T>::value>::type,
        2>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<decay_t<T>>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t hash, const TupleType& result, T& out, Args&&...)
        {
            ce::get(out) = deepCopy(std::get<IDX>(result));
            out.setHash(ct::combineHash(hash, IDX));
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, T& out, Args&&...)
        {
            std::get<IDX>(result) = deepCopy(ce::get(out));
            out.setHash(ct::combineHash(hash, IDX));
        }
    };

    template <class T>
    struct OutputParameterHandler<std::shared_ptr<T>,
                                  typename std::enable_if<std::is_base_of<HashedBase, T>::value>::type,
                                  2>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<std::shared_ptr<decay_t<T>>>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t hash, const TupleType& result, std::shared_ptr<T>& out, Args&&...)
        {
            ce::get(out) = deepCopy(std::get<IDX>(result));
            out->setHash(ct::combineHash(hash, IDX));
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, std::shared_ptr<T>& out, Args&&...)
        {
            std::get<IDX>(result) = deepCopy(ce::get(out));
            out->setHash(ct::combineHash(hash, IDX));
        }
    };

    template <class T>
    struct OutputPackImpl<T>
    {
        static constexpr const size_t OUTPUT_COUNT = (OutputParameterHandler<T>::IS_OUTPUT ? 1 : 0);
        using result_storage_types = typename OutputParameterHandler<T>::result_storage_type;

        template <class TupleType>
        static void getOutputs(size_t hash, TupleType& result, T& out)
        {
            OutputParameterHandler<T>::template getOutput<std::tuple_size<TupleType>::value - 1>(hash, result, out);
        }

        template <class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T& out)
        {
            OutputParameterHandler<T>::template saveOutput<std::tuple_size<TupleType>::value - 1>(hash, result, out);
        }
    };

    template <class T, class... Args>
    struct OutputPackImpl
    {
        static constexpr const size_t OUTPUT_COUNT =
            OutputPackImpl<Args...>::OUTPUT_COUNT + (OutputParameterHandler<T>::IS_OUTPUT ? 1 : 0);
        using storage_type = typename OutputParameterHandler<T>::result_storage_type;
        using result_storage_types =
            typename ct::Append<storage_type, typename OutputPackImpl<Args...>::result_storage_types>::type;

        template <class TupleType>
        static void getOutputs(size_t hash, TupleType& result, T& out, Args&... args)
        {
            constexpr const size_t OUTPUT_COUNT = OutputPackImpl<Args...>::OUTPUT_COUNT;
            constexpr const size_t IDX = std::tuple_size<TupleType>::value - OUTPUT_COUNT - 1;
            OutputParameterHandler<T>::template getOutput<IDX>(hash, result, out, args...);
            OutputPackImpl<Args...>::getOutputs(hash, result, args...);
        }

        template <class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T& out, Args&... args)
        {
            constexpr const size_t OUTPUT_COUNT = OutputPackImpl<Args...>::OUTPUT_COUNT;
            constexpr const size_t IDX = std::tuple_size<TupleType>::value - OUTPUT_COUNT - 1;
            OutputParameterHandler<T>::template saveOutput<IDX>(hash, result, out, args...);
            OutputPackImpl<Args...>::saveOutputs(hash, result, args...);
        }
    };

    template <class... Args>
    struct OutputPack : public OutputPackImpl<Args...>, public IResult
    {
        typename OutputPackImpl<Args...>::result_storage_types::tuple_type values;

        void saveOutputs(Args&... args)
        {
            OutputPackImpl<Args...>::saveOutputs(hash(), values, args...);
        }

        void getOutputs(Args&... args)
        {
            OutputPackImpl<Args...>::getOutputs(hash(), values, args...);
        }
    };

    template <class... Args>
    struct OutputPack<void, Args...> : public OutputPackImpl<Args...>, public IResult
    {
        typename OutputPackImpl<Args...>::result_storage_types::tuple_type values;

        void saveOutputs(Args&... args)
        {
            OutputPackImpl<Args...>::saveOutputs(hash(), values, args...);
        }

        void getOutputs(Args&... args)
        {
            OutputPackImpl<Args...>::getOutputs(hash(), values, args...);
        }
    };
}
