#include "cuda_sync.hpp"
#include <ce/utils.hpp>
#include <ce/CacheEngine.hpp>
#include <ce/sync.hpp>
#include <ce/execute.hpp>
#include <ce/Executor.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifdef HAVE_OPENCV
#include <ce/cv_sync.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudawarping.hpp>
#endif
int main(int argc, char** argv) {
    thrust::host_vector<float> h_vec;
    h_vec.resize(1000);
    for(int i = 0; i < h_vec.size(); ++i)
        h_vec[i] = i;
    auto d_vec = ce::make_input<thrust::device_vector<float>>(h_vec);

    std::shared_ptr<thrust::device_vector<float>> d_out;
    auto exec = ce::make_executor<async_processor>();
    cudaStream_t stream = 0;
    cudaStreamCreate(&stream);

    std::cout << ce::countOutputs(d_vec, ce::make_output(d_out), stream) << std::endl;
    exec.EXEC(&async_processor::apply), d_vec, ce::make_output(d_out), stream);

#ifdef HAVE_OPENCV
    if(argc == 2)
    {
        cv::cuda::Stream stream1, stream2;
        cv::Mat h_img = cv::imread(argv[1]);
        if(!h_img.empty()){
            auto input = ce::make_input<cv::cuda::GpuMat>(h_img);
            cv::cuda::GpuMat output;
            
            ce::exec(cv::cuda::cvtColor, input, ce::make_output(output), cv::COLOR_BGR2GRAY, -1, stream1);
            //ce::exec(cv::cuda::cvtColor, input, ce::make_output(output), cv::COLOR_BGR2GRAY, -1, stream2);
            
        }
    }
    
    
#endif
    return 0;
}