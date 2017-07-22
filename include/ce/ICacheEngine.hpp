#pragma once
#include <ce/IResult.hpp>
#include <ce/export.hpp>
#include <memory>
namespace ce {
    // was the last call to a ce::exec or ce::Executor::exec a cashed executation or a non cached executation, only used if CE_DEBUG_CACHE_USAGE is defined
    bool CE_EXPORT wasCacheUsedLast();
    void CE_EXPORT setCacheUsedLast(bool value);

struct CE_EXPORT ICacheEngine {
    virtual ~ICacheEngine();
    static ICacheEngine* instance();
    static std::unique_ptr<ICacheEngine> create();
    static void setEngine(std::unique_ptr<ICacheEngine>&& engine, bool is_thread_local = false);
    static void releaseEngine(bool thread_engine = false);
    virtual std::shared_ptr<IResult>& getCachedResult(size_t hash) = 0;
};
}