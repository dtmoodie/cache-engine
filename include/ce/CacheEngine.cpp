#include <ce/CacheEngine.hpp>

namespace ce{
static std::unique_ptr<ICacheEngine> g_engine;
thread_local std::unique_ptr<ICacheEngine> t_engine;
thread_local bool t_cache_used_last = false;

bool wasCacheUsedLast(){
    return t_cache_used_last;
}

void setCacheUsedLast(bool value){
    t_cache_used_last = value;
}

ICacheEngine* ICacheEngine::instance() {
    if(t_engine)
        return t_engine.get();
    return g_engine.get();
}

void ICacheEngine::setEngine(std::unique_ptr<ICacheEngine>&& engine, bool is_thread_local){
    if(is_thread_local)
        t_engine = std::move(engine);
    else
        g_engine = std::move(engine);
}

void ICacheEngine::releaseEngine(bool thread_engine){
    if(thread_engine){
        t_engine.reset();
    }else{
        g_engine.reset();
    }
}

ICacheEngine::~ICacheEngine(){
}

CacheEngine::~CacheEngine() {
    m_result_cache.clear();
}

CacheEngine::CacheEngine() {
}

std::shared_ptr<IResult>& CacheEngine::getCachedResult(size_t hash) {
    return m_result_cache[hash];
}

std::unique_ptr<ICacheEngine> ICacheEngine::create(){
    return std::make_unique<CacheEngine>();
}
}