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

    void CacheEngine::printDebug(bool val)
    {
        m_print_debug = val;
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
        if (m_print_debug)
        {
            std::cout << "\n  arghash = " << hash << "  fhash = " << fhash << std::endl;
        }
        auto itr = m_result_cache.find(combineHash(fhash, hash));
        if (itr != m_result_cache.end())
        {
            return itr->second;
        }
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
