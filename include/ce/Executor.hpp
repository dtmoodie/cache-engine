#pragma once
#include <ce/CacheEngine.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/hash.hpp>

#include <ct/String.hpp>

#include <type_traits>

template<class T> struct Executor {
    Executor(T& obj) :m_obj(obj) {}

    template<uint32_t fhash, class R, class...FArgs, class... Args>
    R exec(R(T::*func)(FArgs...), Args&&... args) {
        size_t hash = generateHash(m_hash, args...);
        hash = combineHash(hash, fhash);
        std::cout << "Hash : " << hash << std::endl;
        std::shared_ptr<IResult>& result = ICacheEngine::instance().getCachedResult(hash);
        if (result) {
            std::shared_ptr<TResult<R>> tresult = std::dynamic_pointer_cast<TResult<R>>(result);
            if (tresult) {
                std::cout << "Found result in cache" << std::endl;
                return std::get<0>(tresult->values);
            }
        }

        R ret = (m_obj.*func)(get(std::forward<Args>(args))...);
        result.reset(new TResult<R>(std::forward<R>(ret)));
        return ret;
    }

    template<uint32_t fhash, class...FArgs, class... Args>
    typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type exec(void(T::*func)(FArgs...) const, Args&&... args) {
        // no output but it's a const call soooo execute
        (m_obj.*func)(get(std::forward<Args>(args))...);
    }

    template<uint32_t fhash, class...FArgs, class... Args>
    typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type exec(void(T::*func)(FArgs...), Args&&... args) {
        // Assuming this modifies the object since there is no output
        size_t hash = generateHash(m_hash, args...);
        m_hash = combineHash(hash, fhash);
        (m_obj.*func)(get(std::forward<Args>(args))...);
    }

    template<uint32_t fhash, class...FArgs, class...Args>
    typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0>::type exec(void(T::*func)(FArgs...), Args&&... args) {
        typedef OutputPack<void, Args...> PackType;
        typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
        size_t hash = generateHash(m_hash, args...);
        hash = combineHash(hash, fhash);
        std::cout << "Hash: " << hash << std::endl;
        std::shared_ptr<IResult>& result = ICacheEngine::instance().getCachedResult(hash);
        if (result) {
            std::shared_ptr<TResult<output_tuple_type>> tresult = std::dynamic_pointer_cast<TResult<output_tuple_type>>(result);
            if (tresult) {
                std::cout << "Found result in cache" << std::endl;
                PackType::setOutputs(tresult->values, args...);
                return;
            }
        }
        (m_obj.*func)(get(std::forward<Args>(args))...);
        output_tuple_type results;
        PackType::saveOutputs(results, args...);
        result.reset(new TResult<output_tuple_type>(std::move(results)));
    }

    template<class... Args>
    void set(void(T::*func)(Args...), Args&&...args) {
        m_hash = generateHash(m_hash, args...);
        (m_obj.*func)(get(std::forward<Args>(args))...);
    }

    T& m_obj;
    std::size_t m_hash = generateHash();
};

template<class T>
Executor<T> make_executor(T& obj) {
    return Executor<T>(obj);
}
#define EXEC(func) exec<ct::ctcrc32(#func)>(func