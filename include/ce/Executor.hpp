#pragma once
#include <ce/CacheEngine.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/hash.hpp>

#include <ct/String.hpp>

#include <type_traits>
#include <iostream>

namespace ce {

template<Hash_t FHash, class T, class R, class... FArgs>
struct ConstExecutionToken;

template<Hash_t FHash, class T, class R, class... FArgs>
struct ExecutionToken;

template<Hash_t FHash, class Token, class Executor, class T, class R, class... FArgs>
struct ExecutorHelper {
    ExecutorHelper(Token&& token, Executor& executor) : m_token(std::move(token)), m_executor(executor) {}
    template<class...Args>
    R operator()(Args&&... args) {
        return m_token(m_executor, std::forward<Args>(args)...);
    }
    Executor& m_executor;
    Token m_token;
};

template<class T, class Derived> 
struct ExecutorBase: public Derived {
    template<class ... Args>
    ExecutorBase(Args&&... args):
    Derived(std::forward<Args>(args)...){
    }
    
    template<uint32_t fhash, class R, class...FArgs>
    ExecutorHelper<fhash, ExecutionToken<fhash, T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...> exec(R(T::*func)(FArgs...)) {
        return {std::move(ExecutionToken<fhash, T, R, FArgs...>(func)), *this};
    }

    template<uint32_t fhash, class R, class...FArgs>
    ExecutorHelper<fhash, ConstExecutionToken<fhash, T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...> exec(R(T::*func)(FArgs...) const) {
        return{ std::move(ConstExecutionToken<fhash, T, R, FArgs...>(func)), *this };
    }

    std::size_t m_hash = generateHash();
};

template<class T> 
struct ExecutorRef{
    ExecutorRef(T& obj) :m_obj(obj) {}
    T& m_obj;
};

template<class T>
struct ExecutorOwner{
    template<class... Args>
    ExecutorOwner(Args&&... args):
        m_obj(std::forward<Args>(args)...){
    }

    T m_obj;
};

template<class T>
ExecutorBase<T, ExecutorRef<T>> makeExecutor(T& obj) {
    return ExecutorBase<T, ExecutorRef<T>>(obj);
}

template<class T, class ... Args>
ExecutorBase<T, ExecutorOwner<T>> makeExecutor(Args&&... args){
    ExecutorBase<T, ExecutorOwner<T>> executor(std::forward<Args>(args)...);
    executor.m_hash = combineHash(classHash<T>(), std::forward<Args>(args)...);
    return executor;
}

template<class T>
ExecutorBase<T, ExecutorOwner<T>> makeExecutor() {
    ExecutorBase<T, ExecutorOwner<T>> executor;
    executor.m_hash = classHash<T>();
    return executor;
}

template<class T>
ExecutorBase<T, ExecutorRef<T>> makeExecutor(HashedOutput<T&>& obj){
    ExecutorBase<T, ExecutorRef<T>> ret(obj.m_ref);
    ret.m_hash = obj.m_hash;
    return ret;
}
//ExecutionToken(R(T::*func)(FArgs...), T& obj, Hash_t obj_hash) :
// R(T::*m_func)(FArgs...);
template<class T> T& getObjectRef(ExecutorBase<T, ExecutorRef<T>>& executor){
    return executor.m_obj;
}

template<class T> const T& getObjectRef(const ExecutorBase<T, ExecutorRef<T>>& executor) {
    return executor.m_obj;
}

template<class T> T& getObjectRef(ExecutorBase<T, ExecutorOwner<T>>& executor) {
    return executor.m_obj;
}

template<class T> const T& getObjectRef(const ExecutorBase<T, ExecutorOwner<T>>& executor) {
    return executor.m_obj;
}

template<class T> Hash_t& getObjectHash(ExecutorBase<T, ExecutorRef<T>>& executor) {
    return executor.m_hash;
}
template<class T> Hash_t& getObjectHash(ExecutorBase<T, ExecutorOwner<T>>& executor) {
    return executor.m_hash;
}
template<class T> Hash_t getObjectHash(const ExecutorBase<T, ExecutorRef<T>>& executor) {
    return executor.m_hash;
}
template<class T> Hash_t getObjectHash(const ExecutorBase<T, ExecutorOwner<T>>& executor) {
    return executor.m_hash;
}

template<Hash_t FHash, class T, class R, class... FArgs>
struct ExecutionToken{
    ExecutionToken(R(T::*func)(FArgs...)): m_func(func){
    }

    template<class T2, class ... Args>
    HashedOutput<R> operator()(T2& object, Args&&...args) {
        T& obj = getObjectRef(object);
        Hash_t& obj_hash = getObjectHash(object);
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng) {
            typedef OutputPack<void, HashedOutput<R>, std::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, FHash, args...);
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result) {
                std::shared_ptr<TResult<output_tuple_type>> tresult = std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult) {
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
            HashedOutput<R> out(ret, hash);
            PackType::saveOutputs(hash, results, out, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
            return out;
        }
        return (obj.*m_func)(ce::get(std::forward<Args>(args))...);
    }

    R(T::*m_func)(FArgs...);
};

template<Hash_t FHash, class T, class... FArgs>
struct ExecutionToken<FHash, T, void, FArgs...> {
    ExecutionToken(void(T::*func)(FArgs...)) : m_func(func) {
    }

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, std::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type operator()(T2& object, Args&&...args) {
        T& obj = getObjectRef(object);
        Hash_t& obj_hash = getObjectHash(object);
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng) {
            typedef OutputPack<void, std::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, args...);
            hash = combineHash(hash, FHash);
#ifdef CE_DEBUG_CACHE_USAGE 
            std::cout << "Hash: " << hash << std::endl;
#endif
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result) {
                std::shared_ptr<TResult<output_tuple_type>> tresult = std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult) {
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
        else {
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        }
    }

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, std::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type 
    operator()(T2& object, Args&&...args) {
        T& obj = getObjectRef(object);
        Hash_t& obj_hash = getObjectHash(object);
        size_t hash = generateHash(obj_hash, args...);
        hash = combineHash(hash, FHash);
        obj_hash = hash;
        (obj.*m_func)(ce::get(std::forward<Args>(args))...);
    }

    void(T::*m_func)(FArgs...);
};

template<Hash_t FHash, class T, class R, class... FArgs>
struct ConstExecutionToken {
    ConstExecutionToken(R(T::*func)(FArgs...) const) : m_func(func) {
    }

    template<class T2, class ... Args>
    HashedOutput<R> operator()(const T2& object, Args&&...args) {
        const T& obj = getObjectRef(object);
        Hash_t obj_hash = getObjectHash(object);
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng) {
            typedef OutputPack<void, HashedOutput<R>, std::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, FHash, args...);
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result) {
                std::shared_ptr<TResult<output_tuple_type>> tresult = std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult) {
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
            HashedOutput<R> out(ret, hash);
            PackType::saveOutputs(hash, results, out, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
            return out;
        }
#ifdef CE_DEBUG_CACHE_USAGE 
        setCacheUsedLast(false);
#endif
        return (obj.*m_func)(ce::get(std::forward<Args>(args))...);
    }

    R(T::*m_func)(FArgs...) const;
};

template<Hash_t FHash, class T, class... FArgs>
struct ConstExecutionToken<FHash, T, void, FArgs...> {
    ConstExecutionToken(void(T::*func)(FArgs...) const) : m_func(func) {
    }

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, std::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type 
    operator()(const T2& object, Args&&...args) {
        const T& obj = getObjectRef(object);
        Hash_t obj_hash = getObjectHash(object);
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng) {
            typedef OutputPack<void, std::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(obj_hash, args...);
            hash = combineHash(hash, FHash);
#ifdef CE_DEBUG_CACHE_USAGE 
            std::cout << "Hash: " << hash << std::endl;
#endif
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result) {
                std::shared_ptr<TResult<output_tuple_type>> tresult = std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult) {
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
        else {
#ifdef CE_DEBUG_CACHE_USAGE 
            setCacheUsedLast(false);
#endif
            (obj.*m_func)(ce::get(std::forward<Args>(args))...);
        }
    }

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, std::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type 
    operator()(const T2& object, Args&&...args) {
        const T& obj = getObjectRef(object);
        (obj.*m_func)(ce::get(std::forward<Args>(args))...);
    }

    void(T::*m_func)(FArgs...) const;
};

template<Hash_t FHash, class T, class R, class... FArgs>
ExecutionToken<FHash, T, R, FArgs...> exec(R(T::*func)(FArgs...)){
    return ExecutionToken<FHash, T, R, FArgs...>(func);
}

template<Hash_t FHash, class T, class R, class... FArgs>
ConstExecutionToken<FHash, T, R, FArgs...> exec(R(T::*func)(FArgs...)const) {
    return ConstExecutionToken<FHash, T, R, FArgs...>(func);
}

}
#define EXEC_MEMBER(func) exec<ct::ctcrc32_ignore_whitespace(#func)>(func)
