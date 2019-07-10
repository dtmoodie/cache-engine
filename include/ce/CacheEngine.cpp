#include <ce/CacheEngine.hpp>

namespace ce
{
    namespace
    {
        static std::shared_ptr<ICacheEngine> g_engine;
    }

    std::shared_ptr<ICacheEngine> ICacheEngine::instance()
    {
        if (g_engine == nullptr)
        {
            g_engine = create();
        }
        return g_engine;
    }

    std::shared_ptr<ICacheEngine> ICacheEngine::create()
    {
        return std::make_shared<CacheEngine>();
    }

    void ICacheEngine::setEngine(std::shared_ptr<ICacheEngine> engine)
    {
        g_engine = std::move(engine);
    }

    ICacheEngine::~ICacheEngine()
    {
    }

    CacheEngine::~CacheEngine()
    {
        m_result_cache.clear();
    }

    CacheEngine::CacheEngine(bool debug)
        : m_print_debug(debug)
    {
    }

    bool CacheEngine::printDebug() const
    {
        return m_print_debug;
    }

    bool CacheEngine::wasCacheUsedLast() const
    {
        return m_was_used;
    }

    void CacheEngine::setCacheWasUsed(bool val)
    {
        m_was_used = val;
    }

    std::shared_ptr<IResult> CacheEngine::getCachedResult(size_t fhash, size_t hash) const
    {
        auto itr = m_result_cache.find(combineHash(fhash, hash));
        if (itr != m_result_cache.end())
        {
            m_was_used = true;
            return itr->second;
        }
        m_was_used = false;
        return {};
    }

    void CacheEngine::pushCachedResult(std::shared_ptr<IResult> result, size_t fhash, size_t arg_hash)
    {
        m_result_cache[combineHash(fhash, arg_hash)] = result;
    }

    void CacheEngine::clearCache()
    {
        m_result_cache.clear();
    }
}
