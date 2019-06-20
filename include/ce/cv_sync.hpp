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

    template <>
    struct OutputPack<void, cv::cuda::Stream>
    {
        enum
        {
            OUTPUT_COUNT = 1
        };
        using types = ct::VariadicTypedef<CvEventPool::EventPtr>;

        template <class TupleType>
        static void setOutputs(size_t /*hash*/, TupleType& result, cv::cuda::Stream& stream)
        {
            CvEventPool::EventPtr& ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
            if (*(ev.m_stream) != stream)
            {
                stream.waitEvent(*ev);
            }
        }

        template <class TupleType>
        static void saveOutputs(size_t /*hash*/, TupleType& result, cv::cuda::Stream& stream)
        {
            if (stream)
            {
                CvEventPool::EventPtr& ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
                ev.m_stream.reset(new cv::cuda::Stream(stream));
                ev->record(stream);
            }
        }
    };
}
#endif
