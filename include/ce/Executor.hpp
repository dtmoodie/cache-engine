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
template<class T, class Derived> struct ExecutorBase: public Derived {
    template<class ... Args>
    ExecutorBase(Args&&... args):
    Derived(std::forward<Args>(args)...){
    }
    
    template<uint32_t fhash, class R, class...FArgs, class... Args>
    R exec(R(T::*func)(FArgs...), Args&&... args) {
        ICacheEngine* eng = ICacheEngine::instance();
        if(eng){
            size_t hash = generateHash(m_hash, args...);
            hash = combineHash(hash, fhash);
            std::cout << "Hash : " << hash << std::endl;
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result) {
                std::shared_ptr<TResult<R>> tresult = std::dynamic_pointer_cast<TResult<R>>(result);
                if (tresult) {
                    std::cout << "Found result in cache" << std::endl;
                    return std::get<0>(tresult->values);
                }
            }

            R ret = (this->m_obj.*func)(ce::get(std::forward<Args>(args))...);
            result.reset(new TResult<R>(std::forward<R>(ret)));
            return ret;
        }
        return (this->m_obj.*func)(ce::get(std::forward<Args>(args))...);
    }

    template<uint32_t fhash, class...FArgs, class... Args>
    typename std::enable_if<OutputPack<void, std::decay_t<Args>...>::OUTPUT_COUNT == 0>::type exec(void(T::*func)(FArgs...) const, Args&&... args) {
        // no output but it's a const call soooo execute
        (this->m_obj.*func)(ce::get(std::forward<Args>(args))...);
    }

    template<uint32_t fhash, class...FArgs, class... Args>
    typename std::enable_if<OutputPack<void, std::decay_t<Args>...>::OUTPUT_COUNT == 0>::type exec(void(T::*func)(FArgs...), Args&&... args) {
        // Assuming this modifies the object since there is no output
        //size_t hash = generateHash(m_hash, args...);
        //m_hash = combineHash(hash, fhash);
        (this->m_obj.*func)(ce::get(std::forward<Args>(args))...);
    }

    template<uint32_t fhash, class...FArgs, class...Args>
    typename std::enable_if<OutputPack<void, std::decay_t<Args>...>::OUTPUT_COUNT != 0>::type exec(void(T::*func)(FArgs...), Args&&... args) {
        ICacheEngine* eng = ICacheEngine::instance();
        if(eng){
            typedef OutputPack<void, std::decay_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(m_hash, args...);
            hash = combineHash(hash, fhash);
            std::cout << "Hash: " << hash << std::endl;
            std::shared_ptr<IResult>& result = eng->getCachedResult(hash);
            if (result) {
                std::shared_ptr<TResult<output_tuple_type>> tresult = std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
                if (tresult) {
                    std::cout << "Found result in cache" << std::endl;
                    PackType::setOutputs(hash, tresult->values, args...);
                    return;
                }
            }
            (this->m_obj.*func)(ce::get(std::forward<Args>(args))...);
            output_tuple_type results;
            PackType::saveOutputs(hash, results, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
        }else{
            (this->m_obj.*func)(ce::get(std::forward<Args>(args))...);
        }
    }

    template<class... Args>
    void set(void(T::*func)(Args...), Args&&...args) {
        m_hash = generateHash(m_hash, args...);
        (this->m_obj.*func)(ce::get(std::forward<Args>(args))...);
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
ExecutorBase<T, ExecutorRef<T>> make_executor(T& obj) {
    return ExecutorBase<T, ExecutorRef<T>>(obj);
}

template<class T, class ... Args>
ExecutorBase<T, ExecutorOwner<T>> make_executor(Args&&... args){
    return ExecutorBase<T, ExecutorOwner<T>>(std::forward<Args>(args)...);

}

}
#define EXEC(func) exec<ct::ctcrc32(#func)>(func