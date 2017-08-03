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
            EventPtr();
        };
    private:
        EventPool();
        
        std::list<cudaEvent_t> m_pool;
    };

template<class R, class ... FArgs> 
struct OutputPack<void, R(FArgs...), cudaStream_t> {
    enum {
        OUTPUT_COUNT = 1
    };
    typedef variadic_typedef<EventPool::EventPtr> types;

    template<class TupleType>
    static void setOutputs(size_t hash, TupleType& result, ::cudaStream_t& stream) {
        (void)hash;
        EventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
        cudaStreamWaitEvent(stream, ev.get(), 0);
    }

    template<class TupleType>
    static void saveOutputs(size_t hash, TupleType& result, ::cudaStream_t& stream) {
        (void)hash;
        if(stream){
            EventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
            cudaEventRecord(ev.get(), stream);
        }
    }
};

template<class R, class FArgs, class ... Args>
struct OutputPack<typename std::enable_if<OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT == 0>::type, R(FArgs...), cudaStream_t, Args...> : public OutputPack<void, R(FArgs...), Args...> {
    enum {
        OUTPUT_COUNT = OutputPack<void, Args...>::OUTPUT_COUNT + 1
    };
    typedef variadic_typedef<EventPool::EventPtr> types;

    template<class TupleType>
    static void setOutputs(size_t hash, TupleType& result, cudaStream_t& stream, Args&... args) {
        EventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
        cudaStreamWaitEvent(stream, ev.get(), 0);
        OutputPack<void, Args...>::setOutputs(hash, result, args...);
    }

    template<class TupleType>
    static void saveOutputs(size_t hash,TupleType& result, cudaStream_t& stream, Args&... args) {
        if (stream) {
            EventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
            cudaEventRecord(ev.get(), stream);
        }
        OutputPack<void, Args...>::saveOutputs(hash, result, args...);
    }
};

}