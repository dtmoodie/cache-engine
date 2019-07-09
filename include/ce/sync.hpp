#pragma once
#ifdef HAVE_CUDA
#include <ce/OutputPack.hpp>
#include <ce/export.hpp>
#include <ce/output.hpp>

#include <cuda_runtime_api.h>

#include <list>
#include <memory>

namespace ce
{
    struct CE_EXPORT EventPool
    {
        ~EventPool();
        static EventPool* instance();
        cudaEvent_t request();
        void release(cudaEvent_t ev);

        struct CE_EXPORT EventPtr : public std::shared_ptr<CUevent_st>
        {
            EventPtr();
        };

      private:
        EventPool();

        std::list<cudaEvent_t> m_pool;
    };


    template<>
    struct OutputParameterHandler<cudaStream_t, void, 9>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<EventPool::EventPtr>;

        template<size_t IDX, class TupleType, class ... Args>
        static void getOutput(size_t, const TupleType& result, cudaStream_t& stream, Args&& ...)
        {
            EventPool::EventPtr ev = std::get<IDX>(result);
            cudaStreamWaitEvent(stream, ev.get(), 0);
        }

        template<size_t IDX, class TupleType, class ... Args>
        static void saveOutput(size_t, TupleType& result, cudaStream_t& stream, Args&& ...)
        {
            if (stream)
            {
                EventPool::EventPtr ev = std::get<IDX>(result);
                cudaEventRecord(ev.get(), stream);
            }
        }
    };
}
#endif
