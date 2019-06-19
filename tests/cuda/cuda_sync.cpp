#include "cuda_sync.hpp"
#include <ce/CacheEngine.hpp>
#include <ce/Executor.hpp>
#include <ce/execute.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <ce/sync.hpp>
#include <ce/utils.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char** argv)
{
    thrust::host_vector<float> h_vec;
    h_vec.resize(1000);
    for (int i = 0; i < h_vec.size(); ++i)
    {
        h_vec[i] = i;
    }
    auto d_vec = ce::makeInput<thrust::device_vector<float>>(h_vec);

    std::shared_ptr<thrust::device_vector<float>> d_out;
    auto exec = ce::makeExecutor<async_processor>();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    std::cout << ce::countOutputs(d_vec, ce::makeOutput(d_out), stream) << std::endl;
// exec.EXEC_MEMBER (&async_processor::apply)(d_vec, ce::makeOutput(d_out), stream);

#ifdef HAVE_OPENCV

#endif
    return 0;
}
