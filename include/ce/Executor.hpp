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

template<size_t FHash, class T, class R, class... FArgs>
struct ConstExecutionToken;

template<size_t FHash, class T, class R, class... FArgs>
struct ExecutionToken;

template<size_t FHash, class Token, class Executor, class T, class R, class... FArgs>
struct ExecutorHelper {
    ExecutorHelper(Token&& token, Executor& executor);
    template<class...Args>
    R operator()(Args&&... args);

    Executor& m_executor;
    Token m_token;
};

template<class T, class Derived> 
struct ExecutorBase: public Derived {
    template<class ... Args>
    ExecutorBase(Args&&... args);
    
    template<uint32_t fhash, class R, class...FArgs>
    ExecutorHelper<fhash, ExecutionToken<fhash, T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...> exec(R(T::*func)(FArgs...));

    template<uint32_t fhash, class R, class...FArgs>
    ExecutorHelper<fhash, ConstExecutionToken<fhash, T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...> exec(R(T::*func)(FArgs...) const);

    size_t m_hash = generateHash();
};

template<class T> 
struct ExecutorRef{
    ExecutorRef(T& obj);
    T& m_obj;
};

template<class T>
struct ExecutorOwner{
    template<class... Args>
    ExecutorOwner(Args&&... args);

    T m_obj;
};

template<class T>
ExecutorBase<T, ExecutorRef<T>> makeExecutor(T& obj);

template<class T, class ... Args>
ExecutorBase<T, ExecutorOwner<T>> makeExecutor(Args&&... args);

template<class T>
ExecutorBase<T, ExecutorOwner<T>> makeExecutor();

template<class T>
ExecutorBase<T, ExecutorRef<T>> makeExecutor(HashedOutput<T&>& obj);

template<class T> 
T& getObjectRef(ExecutorBase<T, ExecutorRef<T>>& executor);

template<class T> 
const T& getObjectRef(const ExecutorBase<T, ExecutorRef<T>>& executor);

template<class T> 
T& getObjectRef(ExecutorBase<T, ExecutorOwner<T>>& executor);

template<class T> 
const T& getObjectRef(const ExecutorBase<T, ExecutorOwner<T>>& executor);

template<class T> 
size_t& getObjectHash(ExecutorBase<T, ExecutorRef<T>>& executor);

template<class T> 
size_t& getObjectHash(ExecutorBase<T, ExecutorOwner<T>>& executor);

template<class T> 
size_t getObjectHash(const ExecutorBase<T, ExecutorRef<T>>& executor);

template<class T> 
size_t getObjectHash(const ExecutorBase<T, ExecutorOwner<T>>& executor);

template<size_t FHash, class T, class R, class... FArgs>
struct ExecutionToken{
    ExecutionToken(R(T::*func)(FArgs...));

    template<class T2, class ... Args>
    HashedOutput<R> operator()(T2& object, Args&&...args);

    R(T::*m_func)(FArgs...);
};

template<size_t FHash, class T, class... FArgs>
struct ExecutionToken<FHash, T, void, FArgs...> {
    ExecutionToken(void(T::*func)(FArgs...));

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, void(FArgs...), std::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type operator()(T2& object, Args&&...args);

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, void(FArgs...), std::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type 
    operator()(T2& object, Args&&...args);

    void(T::*m_func)(FArgs...);
};

template<size_t FHash, class T, class R, class... FArgs>
struct ConstExecutionToken {
    ConstExecutionToken(R(T::*func)(FArgs...) const);

    template<class T2, class ... Args>
    HashedOutput<R> operator()(const T2& object, Args&&...args);

    R(T::*m_func)(FArgs...) const;
};

template<size_t FHash, class T, class... FArgs>
struct ConstExecutionToken<FHash, T, void, FArgs...> {
    ConstExecutionToken(void(T::*func)(FArgs...) const);

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, void(FArgs...), std::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type 
    operator()(const T2& object, Args&&...args);

    template<class T2, class ... Args>
    typename std::enable_if<OutputPack<void, void(FArgs...), std::remove_reference_t<Args>...>::OUTPUT_COUNT == 0>::type
    operator()(const T2& object, Args&&...args);

    void(T::*m_func)(FArgs...) const;
};

template<size_t FHash, class T, class R, class... FArgs>
ExecutionToken<FHash, T, R, FArgs...> exec(R(T::*func)(FArgs...));

template<size_t FHash, class T, class R, class... FArgs>
ConstExecutionToken<FHash, T, R, FArgs...> exec(R(T::*func)(FArgs...)const);

}
#define EXEC_MEMBER(func) exec<ct::ctcrc32_ignore_whitespace(#func)>(func)
#include "detail/Executor.hpp"
