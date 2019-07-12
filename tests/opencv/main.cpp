#include <ce/CacheEngine.hpp>
#include <ce/Executor.hpp>
#include <ce/execute.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/sync.hpp>
#include <ce/utils.hpp>

#ifdef HAVE_OPENCV
#include <ce/cv_sync.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

namespace ce
{
    template <class T>
    T& get(cv::Ptr<T>& ptr)
    {
        return *ptr;
    }

    template <class T>
    T& get(HashWrapper<cv::Ptr<T>&>& ptr)
    {
        return *ptr.obj;
    }
}

int main(int argc, char** argv)
{
    ce::ICacheEngine::setEngine(ce::ICacheEngine::create());
    if (argc == 2)
    {
        cv::Mat h_img = cv::imread(argv[1]);
        cv::cuda::Stream stream1, stream2;
        auto kp = cv::cuda::createGoodFeaturesToTrackDetector(CV_32F);
        auto executor = ce::wrapHash(kp);
        // auto executor = ce::makeExecutor(*kp);
        cv::cuda::GpuMat output_mat;
        cv::cuda::GpuMat output_mat2;
        cv::cuda::GpuMat float_mat;
        cv::cuda::GpuMat corners;
        auto input = ce::makeInput<cv::cuda::GpuMat>(h_img);
        if (!h_img.empty())
        {
            auto output = ce::makeOutput(output_mat);
            auto output2 = ce::makeOutput(output_mat2);
            ce::exec(cv::cuda::cvtColor, input, output, cv::COLOR_BGR2GRAY, -1, stream1);
            if (ce::ICacheEngine::instance()->wasCacheUsedLast())
            {
                std::cout << "Did not expect to retrive this value from cache" << std::endl;
                return 1;
            }
            ce::exec(cv::cuda::cvtColor, input, output2, cv::COLOR_BGR2GRAY, -1, stream2);
            if (!ce::ICacheEngine::instance()->wasCacheUsedLast())
            {
                std::cout << "Unable to retrieve result of cv::cuda::cvtColor out of result cache" << std::endl;
                return 1;
            }
            cv::_OutputArray out(float_mat);
            auto float_output = ce::makeOutput(out);
            ce::exec(static_cast<void (cv::cuda::GpuMat::*)(cv::OutputArray, int, double, cv::cuda::Stream&) const>(
                         &cv::cuda::GpuMat::convertTo),
                     output,
                     float_output,
                     CV_32F,
                     1.0,
                     stream2);

            cv::_OutputArray corners_out(corners);
            ce::exec(&cv::cuda::CornersDetector::detect,
                     executor,
                     ce::makeInput(float_output),
                     ce::makeOutput(corners_out),
                     ce::makeEmptyInput(cv::noArray()),
                     stream2);

            ce::exec(&cv::cuda::CornersDetector::detect,
                     executor,
                     ce::makeInput(float_output),
                     ce::makeOutput(corners_out),
                     ce::makeEmptyInput(cv::noArray()),
                     stream1);

            if (ce::ICacheEngine::instance()->wasCacheUsedLast())
            {
                std::cout << "Successfully pulled corner detection results out of cache" << std::endl;
            }
            else
            {
                std::cout << "Unable to pull corner detections out of result cache" << std::endl;
                return 1;
            }
        }
        else
        {
            std::cout << "Unable to load " << argv[1] << std::endl;
        }
        stream1.waitForCompletion();
        stream2.waitForCompletion();
    }
    else
    {
        std::cout << "Did not pass in an image as the arg to this application" << std::endl;
        return 1;
    }

    return 0;
}
