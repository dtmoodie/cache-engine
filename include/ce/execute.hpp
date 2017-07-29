#pragma once
#include <ce/ICacheEngine.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/hash.hpp>
namespace ce {

template<class ... FArgs, class... Args>
typename std::enable_if<OutputPack<void, std::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type exec(void(*func)(FArgs...), Args&&...args) {
    ICacheEngine* eng = ICacheEngine::instance();
    if(eng){
        typedef OutputPack<void, std::remove_reference_t<Args>...> PackType;
        typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
        size_t hash = generateHash(func);
        hash = generateHash(hash, std::forward<Args>(args)...);
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
        func(ce::get(std::forward<Args>(args))...);
        output_tuple_type results;
        PackType::saveOutputs(hash, results, args...);
        result.reset(new TResult<output_tuple_type>(std::move(results)));
    }else{
#ifdef CE_DEBUG_CACHE_USAGE 
        setCacheUsedLast(false);
#endif
        return func(ce::get(std::forward<Args>(args))...);
    }
}

template<class R, class ... FArgs, class... Args>
HashedOutput<R> exec(R(*func)(FArgs...), Args&&...args) {
    ICacheEngine* eng = ICacheEngine::instance();
    if(eng){
        typedef OutputPack<void, HashedOutput<R>, std::remove_reference_t<Args>...> PackType;
        typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
        size_t hash = generateHash(func);
        hash = generateHash(hash, std::forward<Args>(args)...);
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
                HashedOutput<R> ret;
                PackType::setOutputs(hash, tresult->values, ret, args...);
                return ret;
            }
        }
#ifdef CE_DEBUG_CACHE_USAGE 
        setCacheUsedLast(false);
#endif
        R ret = func(ce::get(std::forward<Args>(args))...);
        output_tuple_type results;
		HashedOutput<R> out(std::move(ret), hash);
        PackType::saveOutputs(hash, results, out, args...);
        result.reset(new TResult<output_tuple_type>(std::move(results)));
		return out;
    }
#ifdef CE_DEBUG_CACHE_USAGE 
    setCacheUsedLast(false);
#endif
    R ret = func(ce::get(std::forward<Args>(args))...);
    return HashedOutput<R>(std::move(ret), 0);
}

}