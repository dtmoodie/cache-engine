#pragma once
#include <ce/ICacheEngine.hpp>
#include <ce/IResult.hpp>
#include <memory>
#include <map>

namespace ce{
struct CE_EXPORT CacheEngine : public ICacheEngine{
    CacheEngine();

    std::shared_ptr<IResult>& getCachedResult(size_t hash);
private:
    
    std::map<size_t, std::shared_ptr<IResult>> m_result_cache;
};
}