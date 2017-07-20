#include "cuda_sync.hpp"
#include <ce/utils.hpp>
#include <ce/CacheEngine.hpp>
#include <ce/sync.hpp>

#include <ce/Executor.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

    return 0;
}