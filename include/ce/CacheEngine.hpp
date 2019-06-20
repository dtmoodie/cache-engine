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
        virtual ~CacheEngine() override;
        virtual std::shared_ptr<IResult>& getCachedResult(size_t hash) override;

        bool printDebug() const override;
        bool wasCacheUsedLast() const override;
        void setCacheWasUsed(bool) override;

      private:
        std::unordered_map<size_t, std::shared_ptr<IResult>> m_result_cache;
        bool m_print_debug;
        bool m_was_used = false;
    };
}
