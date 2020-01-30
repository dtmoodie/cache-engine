#pragma once
#include <ce/export.hpp>
#include <ce/hash.hpp>
#include <ce/sync.hpp>

#ifdef HAVE_OPENCV
#include <opencv2/core/cuda.hpp>

namespace ce
{

    size_t combineHash(size_t seed, const cv::cuda::Stream&)
    {
        return seed;
    }

    template <>
    struct HashSelector<cv::cuda::Stream, void, 1>
    {
        static size_t generateHash(const cv::cuda::Stream&)
        {
            return 0;
        }
    };

    struct CE_EXPORT CvEventPool
    {
        ~CvEventPool();
        static CvEventPool* instance();
        cv::cuda::Event* request();
        void release(cv::cuda::Event& ev);

        struct CE_EXPORT EventPtr : public std::shared_ptr<cv::cuda::Event>
        {
            EventPtr();
            std::unique_ptr<cv::cuda::Stream> m_stream;
        };

      private:
        CvEventPool();
        std::list<cv::cuda::Event> m_pool;
    };


    template<>
    struct OutputParameterHandler<cv::cuda::Stream, void, 9>
    {
        static constexpr const bool IS_OUTPUT = true;
        using result_storage_type = ct::VariadicTypedef<CvEventPool::EventPtr>;

        template<size_t IDX, class TupleType, class ... Args>
        static void getOutput(size_t, TupleType& result, cv::cuda::Stream& stream, Args&& ...)
        {
            CvEventPool::EventPtr& ev = std::get<IDX>(result);
            if (*(ev.m_stream) != stream)
            {
                stream.waitEvent(*ev);
            }
        }

        template<size_t IDX, class TupleType, class ... Args>
        static void saveOutput(size_t, TupleType& result, cv::cuda::Stream& stream, Args&& ...)
        {
            if (stream)
            {
                CvEventPool::EventPtr& ev = std::get<IDX>(result);
                ev.m_stream.reset(new cv::cuda::Stream(stream));
                ev->record(stream);
            }
        }
    };
    
}
#endif
