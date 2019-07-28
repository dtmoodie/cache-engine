#pragma once
#include "IResult.hpp"

#include <ce/VariadicTypedef.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/shared_ptr.hpp>
#include <ct/hash.hpp>

#include <memory>
#include <tuple>

namespace ce
{
    template <class T>
    using decay_t = typename std::decay<T>::type;

    template <class F, class T>
    struct OutputPackImpl;

    template <class T, class E = void, int32_t P = 10, int32_t J = 0>
    struct OutputParameterHandler;

    template <class T, int32_t J>
    struct OutputParameterHandler<T, void, 0, J>
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

    template <class T, class E, int32_t P, int32_t J>
    struct OutputParameterHandler : public OutputParameterHandler<T, E, P - 1, J>
    {
    };

    template <class T, int32_t J>
    struct OutputParameterHandler<
        T,
        typename std::enable_if<std::is_base_of<HashedInputBase, decay_t<T>>::value && !std::is_const<T>::value>::type,
        4,
        J>
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

    template <class T, int32_t J>
    struct OutputParameterHandler<
        T,
        ct::EnableIf<std::is_base_of<HashedOutputBase, decay_t<T>>::value && !std::is_const<T>::value>,
        2,
        J>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<typename decay_t<T>::type>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t hash, const TupleType& result, T& out, Args&&...)
        {
            ce::get(out) = deepCopy(ce::get(std::get<IDX>(result)));
            out.setHash(ct::combineHash(hash, IDX));
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, T& out, Args&&...)
        {
            std::get<IDX>(result) = deepCopy(ce::get(out));
            out.setHash(ct::combineHash(hash, IDX));
        }
    };

    // Standalone object inheriting from HashedBase
    template <class T, int32_t J>
    struct OutputParameterHandler<T,
                                  ct::EnableIf<std::is_base_of<HashedBase, decay_t<T>>::value &&
                                               !std::is_const<typename std::remove_reference<T>::type>::value>,
                                  1,
                                  J>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<decay_t<T>>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t, const TupleType& result, T& out, Args&&...)
        {
            out = deepCopy(std::get<IDX>(result));
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, T& out, Args&&...)
        {
            out.setHash(ct::combineHash(hash, IDX));
            std::get<IDX>(result) = out;
        }
    };

    template <class T, int32_t J>
    struct OutputParameterHandler<
        T,
        ct::EnableIf<std::is_base_of<HashedOutputBase, decay_t<T>>::value && !std::is_const<T>::value &&
                     std::is_pointer<typename decay_t<T>::type>::value>,
        3,
        J>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<typename std::remove_pointer<typename decay_t<T>::type>::type>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t hash, const TupleType& result, T& out, Args&&...)
        {
            *out.data = deepCopy(ce::get(std::get<IDX>(result)));
            out.setHash(ct::combineHash(hash, IDX));
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, T& out, Args&&...)
        {
            out.setHash(ct::combineHash(hash, IDX));
            std::get<IDX>(result) = deepCopy(*ce::get(out));
        }
    };

    template <class T, int32_t J>
    struct OutputParameterHandler<std::shared_ptr<T>,
                                  typename std::enable_if<std::is_base_of<HashedBase, T>::value>::type,
                                  2,
                                  J>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<shared_ptr<const decay_t<T>>>;

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
    struct OutputParameterHandler<std::shared_ptr<const T>,
                                  typename std::enable_if<std::is_base_of<HashedBase, T>::value>::type,
                                  3,
                                  0>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<shared_ptr<const T>>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t, const TupleType& result, std::shared_ptr<const T>& out, Args&&...)
        {
            ce::get(out) = std::get<IDX>(result);
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t, TupleType& result, std::shared_ptr<const T>& out, Args&&...)
        {
            std::get<IDX>(result) = ce::get(out);
        }
    };

    template <>
    struct OutputPackImpl<ct::VariadicTypedef<>, ct::VariadicTypedef<>>
    {
        static constexpr const size_t OUTPUT_COUNT = 0;
        using result_storage_types = ct::VariadicTypedef<>;

        template <class TupleType>
        static void getOutputs(size_t, TupleType&)
        {
        }

        template <class TupleType>
        static void saveOutputs(size_t, TupleType&)
        {
        }
    };

    template <class F, class T>
    struct OutputPackImpl<ct::VariadicTypedef<F>, ct::VariadicTypedef<T>>
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

    template <class F, class T, class... FARGS, class... ARGS>
    struct OutputPackImpl<ct::VariadicTypedef<F, FARGS...>, ct::VariadicTypedef<T, ARGS...>>
    {
        using Super = OutputPackImpl<ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>;

        static constexpr const size_t OUTPUT_COUNT =
            Super::OUTPUT_COUNT + (OutputParameterHandler<T>::IS_OUTPUT ? 1 : 0);
        using storage_type = typename OutputParameterHandler<T>::result_storage_type;
        using result_storage_types = typename ct::Append<storage_type, typename Super::result_storage_types>::type;

        template <class TupleType>
        static void getOutputs(size_t hash, TupleType& result, T& out, ARGS&... args)
        {
            constexpr const size_t OUTPUT_COUNT = Super::OUTPUT_COUNT;
            constexpr const size_t IDX = std::tuple_size<TupleType>::value - OUTPUT_COUNT - 1;
            OutputParameterHandler<T>::template getOutput<IDX>(hash, result, out, args...);
            Super::getOutputs(hash, result, args...);
        }

        template <class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T& out, ARGS&... args)
        {
            constexpr const size_t OUTPUT_COUNT = Super::OUTPUT_COUNT;
            constexpr const size_t IDX = std::tuple_size<TupleType>::value - OUTPUT_COUNT - 1;
            OutputParameterHandler<T>::template saveOutput<IDX>(hash, result, out, args...);
            Super::saveOutputs(hash, result, args...);
        }
    };

    template <class R, class FARGS, class ARGS>
    struct OutputPack;

    template <class R, class... FARGS, class... ARGS>
    struct OutputPack<R, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>> : IResult
    {
        using Super = OutputPackImpl<ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>;
        typename ct::Append<typename OutputParameterHandler<R>::result_storage_type,
                            typename Super::result_storage_types>::type::tuple_type values;
        static constexpr const size_t OUTPUT_COUNT = Super::OUTPUT_COUNT + 1;

        void saveOutputs(R& ret, ARGS&... args)
        {
            const auto hsh = hash();
            OutputParameterHandler<R>::template saveOutput<0>(hsh, values, ret);
            Super::saveOutputs(hsh, values, args...);
        }

        void getOutputs(R& ret, ARGS&... args)
        {
            const auto hsh = hash();
            OutputParameterHandler<R>::template getOutput<0>(hsh, values, ret);
            Super::getOutputs(hsh, values, args...);
        }
    };

    template <class... FARGS, class... ARGS>
    struct OutputPack<void, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>> : IResult
    {
        using Super = OutputPackImpl<ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>;
        typename Super::result_storage_types::tuple_type values;
        static constexpr const size_t OUTPUT_COUNT = Super::OUTPUT_COUNT;

        void saveOutputs(ARGS&... args)
        {
            Super::saveOutputs(hash(), values, args...);
        }

        void getOutputs(ARGS&... args)
        {
            Super::getOutputs(hash(), values, args...);
        }
    };
}
