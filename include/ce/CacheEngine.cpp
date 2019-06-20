#include <ce/CacheEngine.hpp>

namespace ce
{
    extern thread_local std::unique_ptr<ICacheEngine> t_engine;
    extern thread_local bool t_cache_used_last;

    static std::unique_ptr<ICacheEngine> g_engine;
    thread_local std::unique_ptr<ICacheEngine> t_engine;
    thread_local bool t_cache_used_last = false;

    bool wasCacheUsedLast()
    {
        return t_cache_used_last;
    }

    void setCacheUsedLast(bool value)
    {
        t_cache_used_last = value;
    }

    ICacheEngine* ICacheEngine::instance()
    {
        if (t_engine)
            return t_engine.get();
        return g_engine.get();
    }

    void ICacheEngine::setEngine(std::unique_ptr<ICacheEngine>&& engine, bool is_thread_local)
    {
        if (is_thread_local)
            t_engine = std::move(engine);
        else
            g_engine = std::move(engine);
    }

    void ICacheEngine::releaseEngine(bool thread_engine)
    {
        if (thread_engine)
        {
            t_engine.reset();
        }
        else
        {
            g_engine.reset();
        }
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

    std::shared_ptr<IResult>& CacheEngine::getCachedResult(size_t hash)
    {
        return m_result_cache[hash];
    }

    std::unique_ptr<ICacheEngine> ICacheEngine::create()
    {
        return std::unique_ptr<ICacheEngine>(new CacheEngine());
    }
}
