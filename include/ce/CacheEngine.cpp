#include <ce/CacheEngine.hpp>

static std::unique_ptr<ICacheEngine> g_engine;
thread_local std::unique_ptr<ICacheEngine> t_engine;

ICacheEngine& ICacheEngine::instance() {
    if(t_engine)
        return *t_engine;
    if(!g_engine){
        g_engine.reset(new CacheEngine());
    }
    return *g_engine;
}

void ICacheEngine::setEngine(std::unique_ptr<ICacheEngine>&& engine, bool is_thread_local){
    if(is_thread_local)
        t_engine = std::move(engine);
    else
        g_engine = std::move(engine);
}

CacheEngine::CacheEngine() {
}

std::shared_ptr<IResult>& CacheEngine::getCachedResult(size_t hash) {
    return m_result_cache[hash];
}
