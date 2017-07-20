#include "sync.hpp"

namespace ce{
    EventPool::EventPool(){
        cudaEvent_t ev = nullptr;
        cudaEventCreate(&ev);
        m_pool.push_back(ev);
    }

    EventPool::~EventPool(){
        for(auto ev : m_pool){
            cudaEventDestroy(ev);
        }
    }

    EventPool* EventPool::instance(){
        static std::unique_ptr<EventPool> g_inst;
        if(!g_inst){
            g_inst.reset(new EventPool());
        }
        return g_inst.get();
    }
    cudaEvent_t EventPool::request(){
        if(m_pool.size()){
            auto ret = m_pool.back();
            m_pool.pop_back();
            return ret;
        }
        cudaEvent_t ev = nullptr;
        cudaEventCreate(&ev);
        return ev;
    }
    void EventPool::release(cudaEvent_t ev){
        m_pool.push_back(ev);
    }
    EventPool::EventPtr::EventPtr() :
        std::shared_ptr<CUevent_st>(EventPool::instance()->request(), [](cudaEvent_t ev) {
        EventPool::instance()->release(ev);
    }) {}
}