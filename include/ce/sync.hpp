#pragma once
#include <ce/export.hpp>
#include <ce/output.hpp>
#include <ce/OutputPack.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <list>
namespace ce{
    struct CE_EXPORT EventPool{
        ~EventPool();
        static EventPool* instance();
        cudaEvent_t request();
        void release(cudaEvent_t ev);

        struct CE_EXPORT EventPtr: public std::shared_ptr<CUevent_st>{
            EventPtr():
            std::shared_ptr<CUevent_st>(EventPool::instance()->request(), [](cudaEvent_t ev){
                EventPool::instance()->release(ev);
            }){}
        };
    private:
        EventPool();
        
        std::list<cudaEvent_t> m_pool;
    };
template<> struct OutputPack<void, cudaStream_t> {
    enum {
        OUTPUT_COUNT = 1
    };
    typedef variadic_typedef<EventPool::EventPtr> types;

    template<class TupleType>
    static void setOutputs(TupleType& result, ::cudaStream_t& stream) {
        EventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
        cudaStreamWaitEvent(stream, ev.get(), 0);
    }
    template<class TupleType>
    static void saveOutputs(TupleType& result, ::cudaStream_t& stream) {
        if(stream){
            EventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
            cudaEventRecord(ev.get(), stream);
        }
    }
};

template<class ... Args>
struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT == 0>::type, cudaStream_t, Args...> : public OutputPack<void, Args...> {
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT + 1
    };
    typedef variadic_typedef<cudaStream_t> types;

    template<class TupleType>
    static void setOutputs(TupleType& result, cudaStream_t& out, Args&... args) {
        //ce::get(out) = std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result);
        OutputPack<void, Args...>::setOutputs(result, args...);
    }

    template<class TupleType>
    static void saveOutputs(TupleType& result, cudaStream_t& out, Args&... args) {
        //std::get<std::tuple_size<TupleType>::value - OutputPack<void, Args...>::OUTPUT_COUNT - 1>(result) = ce::get(out);
        OutputPack<void, Args...>::saveOutputs(result, args...);
    }
};

}