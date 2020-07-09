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

    namespace result_traits
    {
        template <class U>
        struct IsOutput<cv::cuda::Stream, U, void, 9>
        {
            static constexpr const bool value = true;
        };

        template <>
        struct Storage<cv::_OutputArray, cv::cuda::GpuMat, 10> : DefaultStoragePolicy
        {
            using type = cv::cuda::GpuMat;
        };

        template <>
        struct Storage<cv::cuda::Stream, cv::cuda::Stream, 10>
        {
            using type = CvEventPool::EventPtr;

            template <size_t IDX, class ResultStorage, class... Args>
            static void saveResult(const size_t hash, ResultStorage& storage, cv::cuda::Stream& stream, Args&&... args)
            {
                if (stream)
                {
                    CvEventPool::EventPtr& ev = std::get<IDX>(storage);
                    ev.m_stream.reset(new cv::cuda::Stream(stream));
                    ev->record(stream);
                }
            }

            template <size_t IDX, class ResultStorage, class... Args>
            static void
            getResult(const size_t hash, const ResultStorage& storage, cv::cuda::Stream& stream, Args&&... args)
            {
                const CvEventPool::EventPtr& ev = std::get<IDX>(storage);
                if (*(ev.m_stream) != stream)
                {
                    stream.waitEvent(*ev);
                }
            }
        };

    } // namespace result_traits

} // namespace ce
#endif
