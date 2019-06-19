#define CE_DEBUG_CACHE_USAGE
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
    size_t combineHash(size_t seed, const cv::_InputOutputArray& v)
    {
        (void)v;
        return seed;
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
        auto executor = ce::makeExecutor(*kp);
        cv::cuda::GpuMat output_mat;
        cv::cuda::GpuMat float_mat;
        cv::cuda::GpuMat corners;
        auto input = ce::makeInput<cv::cuda::GpuMat>(h_img);
        if (!h_img.empty())
        {
            auto output = ce::makeOutput(output_mat);
            ce::exec(cv::cuda::cvtColor, input, output, cv::COLOR_BGR2GRAY, -1, stream1);
            ce::exec(cv::cuda::cvtColor, input, output, cv::COLOR_BGR2GRAY, -1, stream2);

            auto mat_executor = ce::makeExecutor(output);

            auto float_output = ce::makeOutput(float_mat);
            mat_executor.exec(
                static_cast<void (cv::cuda::GpuMat::*)(cv::OutputArray, int, double, cv::cuda::Stream&) const>(
                    &cv::cuda::GpuMat::convertTo))(float_output, CV_32F, 1.0, stream2);

            executor.exec (&cv::cuda::CornersDetector::detect)(
                ce::makeInput(float_output), ce::makeOutput(corners), ce::makeEmptyInput(cv::noArray()), stream2);

            executor.exec (&cv::cuda::CornersDetector::detect)(
                ce::makeInput(float_output), ce::makeOutput(corners), ce::makeEmptyInput(cv::noArray()), stream1);
            if (ce::wasCacheUsedLast())
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

    ce::ICacheEngine::releaseEngine();
    return 0;
}
