#pragma once
#include "IResult.hpp"

#include <ce/VariadicTypedef.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/shared_ptr.hpp>

#include <ct/hash.hpp>
#include <ct/static_asserts.hpp>
#include <ct/type_traits.hpp>

#include <memory>
#include <tuple>

namespace ce
{

    namespace result_traits
    {
        template <class T>
        using Decay_t = typename std::decay<T>::type;

        template <class T>
        using RemoveRef_t = typename std::remove_reference<T>::type;

        template <bool VAL, class U = void>
        using EnableIf = ct::EnableIf<VAL, U>;

        // This is a template metafunction used for infering if the parameter to a function is in fact an output
        template <class FArg, class CArg, class E = void, int32_t P = 10>
        struct IsOutput;

        template <class T, class U>
        struct IsOutput<T, U, void, 0>
        {
            static constexpr const bool value = false;
        };

        template <class T>
        T& get(std::shared_ptr<T>& ptr)
        {
            if (ptr == nullptr)
            {
                ptr = std::make_shared<T>();
            }
            return *ptr;
        }

        template <class T>
        T& get(T& val)
        {
            return val;
        }

        template <class T>
        T& get(T* val)
        {
            assert(val);
            return *val;
        }

        // default implementation
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
        struct DefaultStoragePolicy
        {
            template <size_t IDX, class T, class ResultStorage, class... Args>
            static void saveResult(const size_t hash, ResultStorage& storage, T& out, Args&&... args)
            {
                std::get<IDX>(storage) = deepCopy(ce::get(out));
                setHash(out, ct::combineHash(hash, IDX));
            }

            template <size_t IDX, class T, class ResultStorage, class... Args>
            static void getResult(const size_t hash, const ResultStorage& storage, T& out, Args&&... args)
            {
                get(out) = deepCopy(std::get<IDX>(storage));
                setHash(out, ct::combineHash(hash, IDX));
            }
        };

        template <class T, class U, class Enable = void, int Priority = 10>
        struct Storage : Storage<T, U, void, Priority - 1>
        {
        };

        template <class T, class U>
        struct Storage<T, U, void, 0> : DefaultStoragePolicy
        {
            using type = Decay_t<T>;
        };

        template <class T>
        struct Storage<std::shared_ptr<T>, std::shared_ptr<T>, void, 1> : DefaultStoragePolicy
        {
            using type = T;
        };

        template <class T>
        struct Storage<std::shared_ptr<const T>, std::shared_ptr<const T>, void, 1>
        {
            using type = std::shared_ptr<const T>;

            template <size_t IDX, class ResultStorage, class... Args>
            static void
            saveResult(const size_t hash, ResultStorage& storage, std::shared_ptr<const T>& out, Args&&... args)
            {
                std::get<IDX>(storage) = deepCopy(ce::get(out));
                /*if (out)
                {
                    setHash(*out, ct::combineHash(hash, IDX));
                }*/
            }

            template <size_t IDX, class ResultStorage, class... Args>
            static void
            getResult(const size_t hash, const ResultStorage& storage, std::shared_ptr<const T>& out, Args&&... args)
            {
                out = deepCopy(std::get<IDX>(storage));
                /*if (out)
                {
                    setHash(*out, ct::combineHash(hash, IDX));
                }*/
            }
        };

        // If the output to a function is a T*, store the value
        // U may be a wrapper type and thus we are not specializating on <T*, T*>
        template <class T>
        struct Storage<T*, T*, void, 1> : DefaultStoragePolicy
        {
            using type = T;
        };

        template <class T, class U, class E, int32_t P>
        struct IsOutput : IsOutput<T, U, E, P - 1>
        {
        };

        template <class T>
        struct DerivedFromHashedInput
        {
            static constexpr const bool value = std::is_base_of<HashedInputBase, Decay_t<T>>::value;
        };
        template <class T>
        struct IsConst : std::is_const<T>
        {
        };

        // Inputs are not outputs
        template <class T, class U>
        struct IsOutput<T, U, EnableIf<(DerivedFromHashedInput<T>::value) || (DerivedFromHashedInput<U>::value)>, 4>
        {
            static constexpr const bool value = false;
        };

        template <class T>
        struct DerivedFromHashedOutput
        {
            static constexpr const bool value = std::is_base_of<HashedOutputBase, Decay_t<T>>::value;
        };

        template <class T, class U>
        struct IsOutput<T,
                        U,
                        ct::EnableIf<(DerivedFromHashedOutput<T>::value) || (DerivedFromHashedOutput<U>::value)>,
                        3>
        {
            static constexpr const bool value = true;
        };

        template <class U, class T>
        struct Storage<U, HashedOutput<T>, void, 4> : DefaultStoragePolicy
        {
            using type = Decay_t<typename Storage<Decay_t<U>, Decay_t<T>>::type>;
        };

        template <class T>
        struct DerivedFromHashedBase
        {
            static constexpr const bool value = std::is_base_of<HashedBase, Decay_t<T>>::value;
        };

        // Specialization for a hashable output wrapped in a HashedOutput
        template <class U, class T>
        struct Storage<U, HashedOutput<T>, EnableIf<DerivedFromHashedBase<T>::value>, 5>
        {
            using type = Decay_t<typename Storage<Decay_t<U>, Decay_t<T>>::type>;

            template <size_t IDX, class ResultStorage, class... Args>
            static void saveResult(const size_t hash, ResultStorage& storage, HashedOutput<T>& out, Args&&... args)
            {
                std::get<IDX>(storage) = deepCopy(ce::get(out));
                setHash(static_cast<T&>(out), ct::combineHash(hash, IDX));
            }

            template <size_t IDX, class ResultStorage, class... Args>
            static void getResult(const size_t hash, const ResultStorage& storage, HashedOutput<T>& out, Args&&... args)
            {
                get(out) = deepCopy(std::get<IDX>(storage));
                setHash(static_cast<T&>(out), ct::combineHash(hash, IDX));
            }
        };

        // Standalone object inheriting from HashedBase
        template <class T, class U>
        struct IsOutput<T,
                        U,
                        EnableIf<(DerivedFromHashedBase<T>::value && !IsConst<RemoveRef_t<T>>::value) ||
                                 (DerivedFromHashedBase<U>::value && !IsConst<RemoveRef_t<U>>::value)>,
                        1>
        {
            static constexpr const bool value = true;
        };

        template <class T, class U>
        struct IsOutput<T&, U, EnableIf<!IsConst<T>::value>, 2>
        {
            static constexpr const bool value = true;
        };

        template <class T>
        void setHash(T&, EnableIf<!DerivedFromHashedBase<T>::value, size_t>)
        {
        }

    } // namespace result_traits

    template <class F, class T>
    struct FunctionArgumentRecurse;

    template <>
    struct FunctionArgumentRecurse<ct::VariadicTypedef<>, ct::VariadicTypedef<>>
    {
        static constexpr const size_t OUTPUT_COUNT = 0;
        using result_storage_types = ct::VariadicTypedef<>;

        template <size_t IDX, class TupleType>
        static void getOutputs(size_t, TupleType&)
        {
        }

        template <size_t IDX, class TupleType>
        static void saveOutputs(size_t, TupleType&)
        {
        }
    };

    template <class FunctionArg,
              class CallsiteArg,
              bool IS_OUTPUT = result_traits::IsOutput<FunctionArg, CallsiteArg>::value>
    struct ResultStorage
    {
        using type = ct::VariadicTypedef<>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t hash, TupleType& result, CallsiteArg& out, Args&&... args)
        {
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, CallsiteArg& out, Args&&... args)
        {
        }
    };

    template <class FunctionArg, class CallsiteArg>
    struct ResultStorage<FunctionArg, CallsiteArg, true>
    {
        using FArg = result_traits::Decay_t<FunctionArg>;
        using CArg = result_traits::Decay_t<CallsiteArg>;
        using StoragePolicy = result_traits::Storage<FArg, CArg>;

        using type = ct::VariadicTypedef<typename StoragePolicy::type>;

        template <size_t IDX, class TupleType, class... Args>
        static void getOutput(size_t hash, TupleType& result, CallsiteArg& out, Args&&... args)
        {
            constexpr size_t TUPLE_SIZE = std::tuple_size<TupleType>::value;
            ct::StaticGreater<size_t, TUPLE_SIZE, IDX>{};
            StoragePolicy::template getResult<IDX>(hash, result, out, std::forward<Args>(args)...);
        }

        template <size_t IDX, class TupleType, class... Args>
        static void saveOutput(size_t hash, TupleType& result, CallsiteArg& out, Args&&... args)
        {
            StoragePolicy::template saveResult<IDX>(hash, result, out, std::forward<Args>(args)...);
        }
    };

    template <class F, class T>
    struct FunctionArgumentRecurse<ct::VariadicTypedef<F>, ct::VariadicTypedef<T>>
    {
        static constexpr const bool IS_OUTPUT = result_traits::IsOutput<F, T>::value;
        static constexpr const size_t OUTPUT_COUNT = (IS_OUTPUT ? 1 : 0);
        using result_storage_types = typename ResultStorage<F, T>::type;

        // U ~= T
        template <size_t IDX, class TupleType, class U>
        static void getOutputs(size_t hash, TupleType& result, U&& out)
        {
            ResultStorage<F, T>::template getOutput<IDX>(hash, result, out);
        }

        // U ~= T
        template <size_t IDX, class TupleType, class U>
        static void saveOutputs(size_t hash, TupleType& result, U&& out)
        {
            ResultStorage<F, T>::template saveOutput<IDX>(hash, result, out);
        }
    };

    template <class F, class T, class... FARGS, class... ARGS>
    struct FunctionArgumentRecurse<ct::VariadicTypedef<F, FARGS...>, ct::VariadicTypedef<T, ARGS...>>
    {
        using Super = FunctionArgumentRecurse<ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>;

        static constexpr const bool IS_OUTPUT = result_traits::IsOutput<F, T>::value;
        static constexpr const size_t OUTPUT_COUNT = Super::OUTPUT_COUNT + (IS_OUTPUT ? 1 : 0);

        using storage_type = typename ResultStorage<F, T>::type;
        using super_storage_types = typename Super::result_storage_types;
        using result_storage_types = ct::append<storage_type, super_storage_types>;

        // U ~= T
        template <size_t IDX, class TupleType, class U, class... Args>
        static void getOutputs(size_t hash, TupleType& result, U&& out, Args&&... args)
        {
            static constexpr const size_t NEXT_IDX = IDX + (IS_OUTPUT ? 1 : 0);
            ResultStorage<F, T>::template getOutput<IDX>(hash, result, out, std::forward<Args>(args)...);
            Super::template getOutputs<NEXT_IDX>(hash, result, std::forward<Args>(args)...);
        }

        template <size_t IDX, class TupleType, class U, class... Args>
        static void saveOutputs(size_t hash, TupleType& result, U&& out, Args&&... args)
        {
            static constexpr const size_t NEXT_IDX = IDX + (IS_OUTPUT ? 1 : 0);
            ResultStorage<F, T>::template saveOutput<IDX>(hash, result, out, std::forward<Args>(args)...);
            Super::template saveOutputs<NEXT_IDX>(hash, result, std::forward<Args>(args)...);
        }
    };

    // Output pack is a structure for storing the output of a computation
    // It is templated on the return type of a function, the arguments to the function and the arguments passed into the
    // function The return type is used for the void specialization since with a void returning function, we must pass
    // in the result of the function Function args can be used to infer if an input to a function is an input or an
    // output Furthermore so can the arguments passed into the function IE a function can be declared as such: void
    // foo(ce::Output<T>, params...); or it can be used as such: void foo(T, params...); cache->exec(&foo,
    // ce::makeOutput(), params...); Thus you can wrap external functions easily without having to decorate
    template <class R, class FARGS, class ARGS>
    struct OutputPack;

    template <class R, class... FARGS, class... ARGS>
    struct OutputPack<R, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>> : IResult
    {
        using function_args = ct::VariadicTypedef<FARGS...>;
        using callsite_args = ct::VariadicTypedef<ARGS...>;
        using Super = FunctionArgumentRecurse<function_args, callsite_args>;

        using return_storage_type = typename ResultStorage<R, R, true>::type;
        using argument_storage_type = typename Super::result_storage_types;
        using storage_type = typename ct::append<return_storage_type, argument_storage_type>::tuple_type;

        storage_type values;

        static constexpr const size_t OUTPUT_COUNT = Super::OUTPUT_COUNT + 1;

        void saveOutputs(R& ret, ARGS&... args)
        {
            const auto hsh = hash();
            ResultStorage<R, R, true>::template saveOutput<0>(hsh, values, ret);
            Super::template saveOutputs<1>(hsh, values, args...);
        }

        void getOutputs(R& ret, ARGS&... args)
        {
            const auto hsh = hash();
            ResultStorage<R, R, true>::template getOutput<0>(hsh, values, ret);
            Super::template getOutputs<1>(hsh, values, args...);
        }
    };

    template <class... FARGS, class... ARGS>
    struct OutputPack<void, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>> : IResult
    {
        using function_args = ct::VariadicTypedef<FARGS...>;
        using callsite_args = ct::VariadicTypedef<ARGS...>;
        using Super = FunctionArgumentRecurse<function_args, callsite_args>;
        using argument_storage_type = typename Super::result_storage_types;
        using storage_type = typename argument_storage_type::tuple_type;

        storage_type values;

        static constexpr const size_t OUTPUT_COUNT = Super::OUTPUT_COUNT;

        template <class... Args>
        void saveOutputs(Args&&... args)
        {
            const auto hsh = hash();
            Super::template saveOutputs<0>(hsh, values, std::forward<Args>(args)...);
        }

        template <class... Args>
        void getOutputs(Args&&... args)
        {
            const auto hsh = hash();
            Super::template getOutputs<0>(hsh, values, std::forward<Args>(args)...);
        }
    };
} // namespace ce
