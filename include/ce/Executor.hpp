#pragma once
#include <ce/CacheEngine.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/hash.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

#include <ct/StringView.hpp>
#include <ct/types/TArrayView.hpp>
#include <cstring>
#include <iostream>
#include <type_traits>

namespace ce
{

    template <class T, class R, class... FArgs>
    struct ConstExecutionToken;

    template <class T, class R, class... FArgs>
    struct ExecutionToken;

    template <class Token, class Executor, class T, class R, class... FArgs>
    struct ExecutorHelper
    {
        ExecutorHelper(Token&& token, Executor& executor);
        template <class... Args>
        R operator()(Args&&... args);

        Executor& m_executor;
        Token m_token;
    };

    template <class T, class Derived>
    struct ExecutorBase : public Derived
    {
        template <class... Args>
        ExecutorBase(Args&&... args);

        template <class R, class... FArgs>
        ExecutorHelper<ExecutionToken<T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...>
        exec(R (T::*func)(FArgs...));

        template <class R, class... FArgs>
        ExecutorHelper<ConstExecutionToken<T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...>
        exec(R (T::*func)(FArgs...) const);

        size_t m_hash = generateHash();
    };

    template <class T>
    struct ExecutorRef
    {
        ExecutorRef(T& obj);
        T& m_obj;
    };

    template <class T>
    struct ExecutorOwner
    {
        template <class... Args>
        ExecutorOwner(Args&&... args);

        T m_obj;
    };

    template <class T>
    ExecutorBase<T, ExecutorRef<T>> makeExecutor(T& obj);

    template <class T, class... Args>
    ExecutorBase<T, ExecutorOwner<T>> makeExecutor(Args&&... args);

    template <class T>
    ExecutorBase<T, ExecutorOwner<T>> makeExecutor();

    template <class T>
    ExecutorBase<T, ExecutorRef<T>> makeExecutor(HashedOutput<T&>& obj);

    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorOwner<T>>& executor);

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorOwner<T>>& executor);

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorOwner<T>>& executor);

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorOwner<T>>& executor);

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
        typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
        operator()(T2& object, Args&&... args);

        template <class T2, class... Args>
        typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
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
        typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
        operator()(const T2& object, Args&&... args);

        template <class T2, class... Args>
        typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
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
    template <class R, class T, class... ARGS>
    constexpr size_t memberFunctionPointerValue(R (T::*ptr)(ARGS...))
    {
        return *ct::ptrCast<size_t>(&ptr);
    }

    template <class R, class T, class... ARGS>
    constexpr size_t memberFunctionPointerValue(R (T::*ptr)(ARGS...) const)
    {
        return *ct::ptrCast<size_t>(&ptr);
    }

    template <class R, class T>
    constexpr size_t memberFunctionPointerValue(R (T::*ptr)() const)
    {
        return *ct::ptrCast<size_t>(&ptr);
    }

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

    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_hash;
    }

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_hash;
    }

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_hash;
    }

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_hash;
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
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng)
        {
            typedef OutputPack<void, HashedOutput<R>, ct::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, m_fhash, args...);
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result)
            {
                std::shared_ptr<TResult<output_tuple_type>> tresult =
                    std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult)
                {
                    HashedOutput<R> ret;
#ifdef CE_DEBUG_CACHE_USAGE
                    std::cout << "Found result in cache" << std::endl;
                    setCacheUsedLast(true);
#endif
                    PackType::setOutputs(hash, tresult->values, ret, args...);
                    return ret;
                }
            }
#ifdef CE_DEBUG_CACHE_USAGE
            setCacheUsedLast(false);
#endif
            R ret = (obj.*m_func)(ce::get(std::forward<Args>(args))...);
            output_tuple_type results;
            HashedOutput<R> out(std::move(ret), hash);
            PackType::saveOutputs(hash, results, out, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
            return out;
        }
        R ret = (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        HashedOutput<R> out(std::move(ret));
        return out;
    }

    template <class T, class... FArgs>
    ExecutionToken<T, void, FArgs...>::ExecutionToken(size_t fhash, void (T::*func)(FArgs...))
        : m_func(func)
        , m_fhash(fhash)
    {
    }


    template<class T>
    void printArgHashImpl(T&& arg)
    {
        std::cout << generateHash(arg) << " ";
    }

    template<class T, class ... ARGS>
    void printArgHashImpl(T&& arg, ARGS&& ... args)
    {
        std::cout << generateHash(arg) << " ";
        printArgHashImpl(std::forward<ARGS>(args)...);
    }

    template<class ... ARGS>
    void printArgHash(ARGS&&... args)
    {
        printArgHashImpl(std::forward<ARGS>(args)...);
        std::cout << std::endl;
    }

    template <class T, class... FArgs>
    template <class T2, class... Args>
    typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
    ExecutionToken<T, void, FArgs...>::operator()(T2& object, Args&&... args)
    {
        T& obj = getObjectRef(object);
        size_t& obj_hash = getObjectHash(object);
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng)
        {
            typedef OutputPack<void, ct::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, args...);
            hash = combineHash(hash, m_fhash);
#ifdef CE_DEBUG_CACHE_USAGE
            std::cout << "arghash: ";
            printArgHash(std::forward<Args>(args)...);
            std::cout << "fhash: " << m_fhash << " Hash: " << hash << std::endl;
#endif
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result)
            {
                std::shared_ptr<TResult<output_tuple_type>> tresult =
                    std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult)
                {
#ifdef CE_DEBUG_CACHE_USAGE
                    std::cout << "Found result in cache" << std::endl;
                    setCacheUsedLast(true);
#endif
                    PackType::setOutputs(hash, tresult->values, args...);
                    return;
                }
            }
#ifdef CE_DEBUG_CACHE_USAGE
            setCacheUsedLast(false);
#endif
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
            output_tuple_type results;
            PackType::saveOutputs(hash, results, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
        }
        else
        {
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        }
    }

    template <class T, class... FArgs>
    template <class T2, class... Args>
    typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
    ExecutionToken<T, void, FArgs...>::operator()(T2& object, Args&&... args)
    {
        T& obj = getObjectRef(object);
        size_t& obj_hash = getObjectHash(object);
        size_t hash = generateHash(obj_hash, args...);
        hash = combineHash(hash, m_fhash);
        obj_hash = hash;
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
        size_t obj_hash = getObjectHash(object);
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng)
        {
            typedef OutputPack<void, HashedOutput<R>, ct::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, m_fhash, args...);
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result)
            {
                std::shared_ptr<TResult<output_tuple_type>> tresult =
                    std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult)
                {
                    HashedOutput<R> ret;
#ifdef CE_DEBUG_CACHE_USAGE
                    std::cout << "Found result in cache" << std::endl;
                    setCacheUsedLast(true);
#endif
                    PackType::setOutputs(hash, tresult->values, ret, args...);
                    return ret;
                }
            }
#ifdef CE_DEBUG_CACHE_USAGE
            setCacheUsedLast(false);
#endif
            R ret = (obj.*m_func)(ce::get(std::forward<Args>(args))...);
            output_tuple_type results;
            HashedOutput<R> out(std::move(ret), hash);
            PackType::saveOutputs(hash, results, out, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
            return out;
        }
#ifdef CE_DEBUG_CACHE_USAGE
        setCacheUsedLast(false);
#endif
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
    typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
    ConstExecutionToken<T, void, FArgs...>::operator()(const T2& object, Args&&... args)
    {
        const T& obj = getObjectRef(object);
        size_t obj_hash = getObjectHash(object);
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng)
        {
            typedef OutputPack<void, ct::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, args...);
            hash = combineHash(hash, m_fhash);
#ifdef CE_DEBUG_CACHE_USAGE
            std::cout << "Hash: " << hash << std::endl;
#endif
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result)
            {
                std::shared_ptr<TResult<output_tuple_type>> tresult =
                    std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult)
                {
#ifdef CE_DEBUG_CACHE_USAGE
                    std::cout << "Found result in cache" << std::endl;
                    setCacheUsedLast(true);
#endif
                    PackType::setOutputs(hash, tresult->values, args...);
                    return;
                }
            }
#ifdef CE_DEBUG_CACHE_USAGE
            setCacheUsedLast(false);
#endif
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
            output_tuple_type results;
            PackType::saveOutputs(hash, results, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
        }
        else
        {
#ifdef CE_DEBUG_CACHE_USAGE
            setCacheUsedLast(false);
#endif
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        }
    }

    template <class T, class... FArgs>
    template <class T2, class... Args>
    typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
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
