#pragma once

#include <ce/sync.hpp>
#include <ce/export.hpp>
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda.hpp>

namespace ce{
	
    size_t generateHash(const cv::cuda::Stream& data) {
        (void)data;
        return 0;
    }
    struct CE_EXPORT CvEventPool {
        ~CvEventPool();
        static CvEventPool* instance();
        cv::cuda::Event* request();
        void release(cv::cuda::Event& ev);

        struct CE_EXPORT EventPtr : public std::shared_ptr<cv::cuda::Event> {
            EventPtr();
            std::unique_ptr<cv::cuda::Stream> m_stream;
        };
    private:
        CvEventPool();
        std::list<cv::cuda::Event> m_pool;
    };
    template<int Idx, class Tuple, class T>
    void setOutput(size_t hash, Tuple& result, cv::cuda::Stream& arg) {
        (void)hash;
        CvEventPool::EventPtr& ev = std::get<Idx>(result);
        if(*(ev.m_stream) != arg)
            arg.waitEvent(*ev);
    }
    template<int Idx, class Tuple, class T>
    void saveOutput(size_t hash, Tuple& result, cv::cuda::Stream& arg) {
        (void)hash;
        if(stream){
            CvEventPool::EventPtr& ev = std::get<Idx>(result);
            ev.m_stream = std::make_unique<cv::cuda::Stream>(stream);
            ev->record(stream);
        }
    }
    namespace type_traits {
        namespace argument_specializations {
            template<class T>
            struct SaveType<cv::cuda::Stream&, T, 2>{
                enum{IS_OUTPUT = 1};
                typedef CvEventPool::EventPtr type;
                inline static size_t hash(const T& val) {
                    return 0;
                }
            };
        }
    }
    
}
#endif