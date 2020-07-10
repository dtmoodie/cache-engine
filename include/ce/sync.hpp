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

    namespace result_traits
    {
        template <class U>
        struct IsOutput<cudaStream_t, U, void, 9>
        {
            static constexpr const bool value = true;
        };

        template <>
        struct Storage<cudaStream_t, cudaStream_t, void, 10>
        {
            using type = EventPool::EventPtr;

            template <size_t IDX, class ResultStorage, class... Args>
            static void saveResult(const size_t hash, ResultStorage& storage, cudaStream_t stream, Args&&... args)
            {
                if (stream)
                {
                    EventPool::EventPtr ev = std::get<IDX>(storage);
                    cudaEventRecord(ev.get(), stream);
                }
            }

            template <size_t IDX, class ResultStorage, class... Args>
            static void getResult(const size_t hash, const ResultStorage& storage, cudaStream_t stream, Args&&... args)
            {
                EventPool::EventPtr ev = std::get<IDX>(storage);
                cudaStreamWaitEvent(stream, ev.get(), 0);
            }
        };

    } // namespace result_traits

} // namespace ce
#endif
