#define CE_DEBUG_CACHE_USAGE
#include <ce/utils.hpp>
#include <ce/CacheEngine.hpp>
#include <ce/sync.hpp>
#include <ce/execute.hpp>
#include <ce/Executor.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

#ifdef HAVE_OPENCV
#include <ce/cv_sync.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudawarping.hpp>
#endif



namespace ce{
    namespace type_traits {
        namespace argument_specializations {
            template<class T>
            struct remove_output{
                typedef T type;
            };

            template<class T>
            struct remove_output<ce::HashedOutput<T>>{
                typedef std::remove_reference_t<T> type;
            };


            template<class T>
            struct SaveType<cv::OutputArray, T> {
                enum { IS_OUTPUT = 1 };
                typedef typename remove_output<std::remove_reference_t<T>>::type type;
            };
        }
    }

size_t combineHash(size_t seed, const cv::_InputOutputArray& v) {
    (void)v;
    return seed;
}
namespace debug{
    template<class R, class... FArgs, class... Args>
    void debugExecute(R(*func)(FArgs...), Args&&...){
        typedef ce::OutputPack<R(FArgs...), Args...> AI;
        AI::debugPrint();
    }
}
}
int main(int argc, char** argv) {
	ce::ICacheEngine::setEngine(ce::ICacheEngine::create());
	if (argc == 2){
		cv::Mat h_img = cv::imread(argv[1]);
        cv::cuda::Stream stream1, stream2;
        auto kp = cv::cuda::createGoodFeaturesToTrackDetector(CV_32F);
        auto executor = ce::makeExecutor(*kp);
        cv::cuda::GpuMat output_mat;
        cv::cuda::GpuMat float_mat;
        cv::cuda::GpuMat corners;
        auto input = ce::makeInput<cv::cuda::GpuMat>(h_img);
		if (!h_img.empty()) {
            auto output = ce::makeOutput(output_mat);
            //ce::debug::debugExecute(cv::cuda::cvtColor, input, output, cv::COLOR_BGR2GRAY, -1, stream1);
            ce::exec(cv::cuda::cvtColor, input, output, cv::COLOR_BGR2GRAY, -1, stream1);
            /*ce::exec(cv::cuda::cvtColor, input, output, cv::COLOR_BGR2GRAY, -1, stream2);
            
            auto mat_executor = ce::makeExecutor(output);

            auto float_output = ce::makeOutput(float_mat);
            mat_executor.EXEC_MEMBER(static_cast<void(cv::cuda::GpuMat::*)(cv::OutputArray, int, double, cv::cuda::Stream&)const>(&cv::cuda::GpuMat::convertTo))(
                float_output, CV_32F, 1.0, stream2);

            executor.EXEC_MEMBER(&cv::cuda::CornersDetector::detect)(ce::makeInput(float_output), ce::makeOutput(corners), ce::makeEmptyInput(cv::noArray()), stream2);
            executor.EXEC_MEMBER(&cv::cuda::CornersDetector::detect)(ce::makeInput(float_output), ce::makeOutput(corners), ce::makeEmptyInput(cv::noArray()), stream1);
            */

		}
        stream1.waitForCompletion();
        stream2.waitForCompletion();
	}
    
    ce::ICacheEngine::releaseEngine();
	return 0;
}