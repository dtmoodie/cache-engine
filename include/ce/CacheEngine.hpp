#pragma once
#include <ce/ICacheEngine.hpp>
#include <ce/IResult.hpp>
#include <memory>
#include <unordered_map>

namespace ce
{
    struct CE_EXPORT CacheEngine : public ICacheEngine
    {
        CacheEngine(bool debug_print = false);
        ~CacheEngine() override;
        std::shared_ptr<IResult> getCachedResult(size_t fhash, size_t hash) const override;
        void pushCachedResult(std::shared_ptr<IResult>, size_t fhash, size_t arg_hash) override;

        bool printDebug() const override;
        void printDebug(bool val) override;
        bool wasCacheUsedLast() const override;
        void setCacheWasUsed(bool) override;
        void clearCache() override;

      private:
        std::unordered_map<size_t, std::shared_ptr<IResult>> m_result_cache;
        bool m_print_debug;
        mutable bool m_was_used = false;
    };
}
