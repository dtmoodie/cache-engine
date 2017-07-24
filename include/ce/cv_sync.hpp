#pragma once

#include <ce/sync.hpp>
#include <ce/export.hpp>
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda.hpp>

namespace ce{
    struct CE_EXPORT CvEventPool {
        ~CvEventPool();
        static CvEventPool* instance();
        cv::cuda::Event* request();
        void release(cv::cuda::Event& ev);

        struct CE_EXPORT EventPtr : public std::shared_ptr<cv::cuda::Event> {
            EventPtr();
        };
    private:
        CvEventPool();
        std::list<cv::cuda::Event> m_pool;
    };

    template<> struct OutputPack<void, cv::cuda::Stream> {
        enum {
            OUTPUT_COUNT = 1
        };
        typedef variadic_typedef<CvEventPool::EventPtr> types;

        template<class TupleType>
        static void setOutputs(size_t hash,TupleType& result, cv::cuda::Stream& stream) {
            (void)hash;
            CvEventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
            stream.waitEvent(*ev);
        }

        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, cv::cuda::Stream& stream) {
            /*(void)hash;
            if (stream) {
                CvEventPool::EventPtr ev = std::get<std::tuple_size<TupleType>::value - 1>(result);
                ev->record(stream);
            }*/
        }
    };
}
#endif