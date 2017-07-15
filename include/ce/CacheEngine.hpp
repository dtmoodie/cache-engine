#pragma once
#include <ce/IResult.hpp>
#include <memory>
#include <map>

struct CacheEngine {
    static CacheEngine& instance() {
        static CacheEngine inst;
        return inst;
    }

    std::shared_ptr<IResult>& getCachedResult(size_t hash) {
        return m_result_cache[hash];
    }
private:
    CacheEngine() {}
    std::map<size_t, std::shared_ptr<IResult>> m_result_cache;
};