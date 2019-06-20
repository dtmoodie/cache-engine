#ifdef HAVE_OPENCV
#include <ce/cv_sync.hpp>
namespace ce
{

    CvEventPool::CvEventPool()
    {
        m_pool.push_back(cv::cuda::Event());
    }

    CvEventPool::~CvEventPool()
    {
    }

    CvEventPool* CvEventPool::instance()
    {
        static std::unique_ptr<CvEventPool> g_inst;
        if (!g_inst)
            g_inst.reset(new CvEventPool());
        return g_inst.get();
    }
    cv::cuda::Event* CvEventPool::request()
    {
        if (m_pool.size())
        {
            cv::cuda::Event* out = new cv::cuda::Event(m_pool.back());
            m_pool.pop_back();
            return out;
        }
        return new cv::cuda::Event();
    }
    void CvEventPool::release(cv::cuda::Event& ev)
    {
        m_pool.push_back(std::move(ev));
    }

    CvEventPool::EventPtr::EventPtr()
        : std::shared_ptr<cv::cuda::Event>(CvEventPool::instance()->request(), [](cv::cuda::Event* ev) {
            CvEventPool::instance()->release(*ev);
            delete ev;
        })
    {
    }
}
#endif // HAVE_OPENCV
