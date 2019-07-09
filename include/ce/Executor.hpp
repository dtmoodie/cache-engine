#pragma once
#include <ce/output.hpp>

#include <ce/ICacheEngine.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/input.hpp>

#include <cstring>
#include <ct/StringView.hpp>

#include <iostream>
#include <type_traits>

// This whole file is full of helper functions for executing member functions on objects.
// The trivial case is when it's a const member function and thus we can just invoke it and memoize the results
// however the non trivial case invoves potential state change :/
namespace ce
{

    template <class T>
    ExecutorBase<T, ExecutorRef<T>> makeExecutor(T& obj);

    template <class T, class... Args>
    ExecutorBase<T, ExecutorOwner<T>> makeExecutor(Args&&... args);

    template <class T>
    ExecutorBase<T, ExecutorOwner<T>> makeExecutor();

    template <class T>
    ExecutorBase<T, ExecutorRef<T>> makeExecutor(HashedOutput<T&>& obj);

    template <class T, class R, class... FArgs>
    struct ExecutionToken
    {
        ExecutionToken(size_t fhash, R (T::*func)(FArgs...));

        template <class T2, class... Args>
        HashedOutput<R> operator()(T2& object, Args&&... args);

        R (T::*m_func)(FArgs...);
        size_t m_fhash;
    };

    template <class T, class... FArgs>
    struct ExecutionToken<T, void, FArgs...>
    {
        ExecutionToken(size_t fhash, void (T::*func)(FArgs...));

        template <class T2, class... Args>
        typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
        operator()(T2& object, Args&&... args);

        template <class T2, class... Args>
        typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
        operator()(T2& object, Args&&... args);

        void (T::*m_func)(FArgs...);
        size_t m_fhash;
    };

    template <class T, class R, class... FArgs>
    struct ConstExecutionToken
    {
        ConstExecutionToken(size_t fhash, R (T::*func)(FArgs...) const);

        template <class T2, class... Args>
        HashedOutput<R> operator()(const T2& object, Args&&... args);

        R (T::*m_func)(FArgs...) const;
        size_t m_fhash;
    };

    template <class T, class... FArgs>
    struct ConstExecutionToken<T, void, FArgs...>
    {
        ConstExecutionToken(size_t fhash, void (T::*func)(FArgs...) const);

        template <class T2, class... Args>
        typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
        operator()(const T2& object, Args&&... args);

        template <class T2, class... Args>
        typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
        operator()(const T2& object, Args&&... args);

        void (T::*m_func)(FArgs...) const;
        size_t m_fhash;
    };

    template <class T, class R, class... FArgs>
    ExecutionToken<T, R, FArgs...> exec(R (T::*func)(FArgs...));

    template <class T, class R, class... FArgs>
    ConstExecutionToken<T, R, FArgs...> exec(R (T::*func)(FArgs...) const);
}

namespace ce
{

    template <class Token, class Executor, class T, class R, class... FArgs>
    struct ExecutorHelper;

    template <class Token, class Executor, class T, class R, class... FArgs>
    ExecutorHelper<Token, Executor, T, R, FArgs...>::ExecutorHelper(Token&& token, Executor& executor)
        : m_token(std::move(token))
        , m_executor(executor)
    {
    }

    template <class Token, class Executor, class T, class R, class... FArgs>
    template <class... Args>
    R ExecutorHelper<Token, Executor, T, R, FArgs...>::operator()(Args&&... args)
    {
        return m_token(m_executor, std::forward<Args>(args)...);
    }

    template <class T, class Derived>
    struct ExecutorBase;

    template <class T, class Derived>
    template <class... Args>
    ExecutorBase<T, Derived>::ExecutorBase(Args&&... args)
        : Derived(std::forward<Args>(args)...)
    {
    }

    template <class T, class Derived>
    template <class R, class... FArgs>
    ExecutorHelper<ExecutionToken<T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...>
    ExecutorBase<T, Derived>::exec(R (T::*func)(FArgs...))
    {
        return {std::move(ExecutionToken<T, R, FArgs...>(memberFunctionPointerValue(func), func)), *this};
    }

    template <class T, class Derived>
    template <class R, class... FArgs>
    ExecutorHelper<ConstExecutionToken<T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...>
    ExecutorBase<T, Derived>::exec(R (T::*func)(FArgs...) const)
    {
        return {ConstExecutionToken<T, R, FArgs...>(memberFunctionPointerValue(func), func), *this};
    }

    template <class T>
    struct ExecutorRef;
    template <class T>
    ExecutorRef<T>::ExecutorRef(T& obj)
        : m_obj(obj)
    {
    }

    template <class T>
    struct ExecutorOwner;

    template <class T>
    template <class... Args>
    ExecutorOwner<T>::ExecutorOwner(Args&&... args)
        : m_obj(std::forward<Args>(args)...)
    {
    }

    template <class T>
    ExecutorBase<T, ExecutorRef<T>> makeExecutor(T& obj)
    {
        return ExecutorBase<T, ExecutorRef<T>>(obj);
    }

    template <class T, class... Args>
    ExecutorBase<T, ExecutorOwner<T>> makeExecutor(Args&&... args)
    {
        ExecutorBase<T, ExecutorOwner<T>> executor(std::forward<Args>(args)...);
        executor.m_hash = combineHash(classHash<T>(), std::forward<Args>(args)...);
        return executor;
    }

    template <class T>
    ExecutorBase<T, ExecutorOwner<T>> makeExecutor()
    {
        ExecutorBase<T, ExecutorOwner<T>> executor;
        executor.m_hash = classHash<T>();
        return executor;
    }

    template <class T>
    ExecutorBase<T, ExecutorRef<T>> makeExecutor(HashedOutput<T&>& obj)
    {
        ExecutorBase<T, ExecutorRef<T>> ret(obj.m_ref);
        ret.m_hash = obj.m_hash;
        return ret;
    }

    template <class T, class R, class... FArgs>
    struct ExecutionToken;

    template <class T, class R, class... FArgs>
    ExecutionToken<T, R, FArgs...>::ExecutionToken(size_t fhash, R (T::*func)(FArgs...))
        : m_func(func)
        , m_fhash(fhash)
    {
    }

    template <class T, class R, class... FArgs>
    template <class T2, class... Args>
    HashedOutput<R> ExecutionToken<T, R, FArgs...>::operator()(T2& object, Args&&... args)
    {
        T& obj = getObjectRef(object);
        size_t& obj_hash = getObjectHash(object);
        auto eng = ICacheEngine::instance();
        if (eng)
        {
            using PackType = OutputPack<HashedOutput<R>, ct::remove_reference_t<Args>...>;
            using TupleType = typename PackType::result_storage_types::tuple_type;
            const auto arg_hash = generateHash(obj_hash, std::forward<Args>(args)...);
            const size_t combined_hash = generateHash(m_fhash, arg_hash);
            std::shared_ptr<IResult> result = eng->getCachedResult(m_fhash, arg_hash);
            if (result)
            {
                std::shared_ptr<TResult<TupleType>> tresult = std::dynamic_pointer_cast<TResult<TupleType>>(result);
                if (tresult)
                {
                    HashedOutput<R> ret;
#ifdef CE_DEBUG_CACHE_USAGE
                    std::cout << "Found result in cache" << std::endl;
#endif
                    PackType::getOutputs(combined_hash, tresult->values, ret, args...);
                    return ret;
                }
            }
            TupleType results;
            HashedOutput<R> out((obj.*m_func)(ce::get(std::forward<Args>(args))...), combined_hash);
            PackType::saveOutputs(combined_hash, results, out, args...);
            result.reset(new TResult<TupleType>(std::move(results)));
            eng->pushCachedResult(result, m_fhash, arg_hash);
            return out;
        }
        HashedOutput<R> out((obj.*m_func)(ce::get(std::forward<Args>(args))...));
        return out;
    }

    template <class T, class... FArgs>
    ExecutionToken<T, void, FArgs...>::ExecutionToken(size_t fhash, void (T::*func)(FArgs...))
        : m_func(func)
        , m_fhash(fhash)
    {
    }

    template <class T>
    void printArgHashImpl(T&& arg)
    {
        std::cout << generateHash(arg) << " ";
    }

    template <class T, class... ARGS>
    void printArgHashImpl(T&& arg, ARGS&&... args)
    {
        std::cout << generateHash(arg) << " ";
        printArgHashImpl(std::forward<ARGS>(args)...);
    }

    template <class... ARGS>
    void printArgHash(ARGS&&... args)
    {
        printArgHashImpl(std::forward<ARGS>(args)...);
    }

    template <class T, class... FArgs>
    template <class T2, class... Args>
    typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
    ExecutionToken<T, void, FArgs...>::operator()(T2& object, Args&&... args)
    {
        using PackType = OutputPack<ct::remove_reference_t<Args>...>;
        using TupleType = typename PackType::result_storage_types::tuple_type;

        T& obj = getObjectRef(object);
        size_t& obj_hash = getObjectHash(object);
        auto eng = ICacheEngine::instance();
        if (eng)
        {
            const size_t arg_hash = generateHash(obj_hash, args...);
            const auto combined_hash = combineHash(m_fhash, arg_hash);
            if (eng->printDebug())
            {
                std::cout << "arghash: (";
                printArgHash(std::forward<Args>(args)...);
                std::cout << ") ";
                std::cout << "fhash: " << m_fhash << " Hash: " << arg_hash << std::endl;
            }
            std::shared_ptr<IResult> result = eng->getCachedResult(m_fhash, arg_hash);
            if (result)
            {
                std::shared_ptr<TResult<TupleType>> tresult = std::dynamic_pointer_cast<TResult<TupleType>>(result);
                if (tresult)
                {
                    if (eng->printDebug())
                    {
                        std::cout << "Found result in cache" << std::endl;
                    }
                    eng->setCacheWasUsed(true);
                    PackType::getOutputs(combined_hash, tresult->values, args...);
                    return;
                }
            }
            eng->setCacheWasUsed(false);
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
            TupleType results;
            PackType::saveOutputs(combined_hash, results, args...);
            result.reset(new TResult<TupleType>(std::move(results)));
            eng->pushCachedResult(result, m_fhash, arg_hash);
        }
        else
        {
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        }
    }

    template <class T, class... FArgs>
    template <class T2, class... Args>
    typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
    ExecutionToken<T, void, FArgs...>::operator()(T2& object, Args&&... args)
    {
        T& obj = getObjectRef(object);
        size_t& obj_hash = getObjectHash(object);
        size_t arg_hash = generateHash(obj_hash, args...);
        const auto combined_hash = combineHash(m_fhash, arg_hash);
        obj_hash = combined_hash;
        (obj.*m_func)(ce::get(std::forward<Args>(args))...);
    }

    template <class T, class R, class... FArgs>
    struct ConstExecutionToken;
    template <class T, class R, class... FArgs>
    ConstExecutionToken<T, R, FArgs...>::ConstExecutionToken(size_t fhash, R (T::*func)(FArgs...) const)
        : m_func(func)
        , m_fhash(fhash)
    {
    }

    template <class T, class R, class... FArgs>
    template <class T2, class... Args>
    HashedOutput<R> ConstExecutionToken<T, R, FArgs...>::operator()(const T2& object, Args&&... args)
    {
        const T& obj = getObjectRef(object);
        auto eng = ICacheEngine::instance();
        if (eng)
        {
            return eng->exec(m_func, object, std::forward<Args>(args)...);
        }
        R ret = (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        HashedOutput<R> out(std::move(ret));
        return out;
    }

    template <class T, class... FArgs>
    ConstExecutionToken<T, void, FArgs...>::ConstExecutionToken(size_t fhash, void (T::*func)(FArgs...) const)
        : m_func(func)
        , m_fhash(fhash)
    {
    }

    template <class T, class... FArgs>
    template <class T2, class... Args>
    typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
    ConstExecutionToken<T, void, FArgs...>::operator()(const T2& object, Args&&... args)
    {
        const T& obj = getObjectRef(object);
        auto eng = ICacheEngine::instance();
        if (eng)
        {
            eng->exec(m_func, object, std::forward<Args>(args)...);
        }
        else
        {
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        }
    }

    template <class T, class... FArgs>
    template <class T2, class... Args>
    typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
    ConstExecutionToken<T, void, FArgs...>::operator()(const T2& object, Args&&... args)
    {
        const T& obj = getObjectRef(object);
        (obj.*m_func)(ce::get(std::forward<Args>(args))...);
    }

    template <class T, class R, class... FArgs>
    ExecutionToken<T, R, FArgs...> exec(R (T::*func)(FArgs...))
    {
        return ExecutionToken<T, R, FArgs...>(memberFunctionPointerValue(func), func);
    }

    template <class T, class R, class... FArgs>
    ConstExecutionToken<T, R, FArgs...> exec(R (T::*func)(FArgs...) const)
    {
        return ConstExecutionToken<T, R, FArgs...>(memberFunctionPointerValue(func), func);
    }
}
