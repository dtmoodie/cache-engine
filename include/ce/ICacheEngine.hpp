#pragma once
#include <ce/IResult.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/export.hpp>
#include <ce/hash.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

#include <ct/type_traits.hpp>

#include <memory>
namespace ce
{
    // was the last call to a ce::exec or ce::Executor::exec a cashed executation or a non cached executation, only used
    // if CE_DEBUG_CACHE_USAGE is defined
    bool CE_EXPORT wasCacheUsedLast();
    void CE_EXPORT setCacheUsedLast(bool value);

    struct CE_EXPORT ICacheEngine
    {
        virtual ~ICacheEngine();
        static ICacheEngine* instance();
        static std::unique_ptr<ICacheEngine> create();
        static void setEngine(std::unique_ptr<ICacheEngine>&& engine, bool is_thread_local = false);
        static void releaseEngine(bool thread_engine = false);

        virtual std::shared_ptr<IResult>& getCachedResult(size_t hash) = 0;

        // Function doesn't return
        template <class... FArgs, class... Args>
        typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
        exec(void (*func)(FArgs...), Args&&... args)
        {
            typedef OutputPack<void, ct::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(func);
            hash = generateHash(hash, std::forward<Args>(args)...);
#ifdef CE_DEBUG_CACHE_USAGE
            std::cout << "Hash: " << hash << std::endl;
#endif
            std::shared_ptr<IResult>& result = this->getCachedResult(hash);
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
            func(ce::get(std::forward<Args>(args))...);
            output_tuple_type results;
            PackType::saveOutputs(hash, results, args...);
            result.reset(new TResult<output_tuple_type>(std::move(results)));
        }

        // function returns
        template <class R, class... FArgs, class... Args>
        HashedOutput<R> exec(R (*func)(FArgs...), Args&&... args)
        {
            typedef OutputPack<void, HashedOutput<R>, ct::remove_reference_t<Args>...> PackType;
            typedef typename convert_in_tuple<typename PackType::types>::type output_tuple_type;
            size_t hash = generateHash(func);
            hash = generateHash(hash, std::forward<Args>(args)...);
#ifdef CE_DEBUG_CACHE_USAGE
            std::cout << "Hash: " << hash << std::endl;
#endif
            std::shared_ptr<IResult>& result = this->getCachedResult(hash);
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
    };
}
