#include "cuda_sync.hpp"

#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>

struct pow_op{
    int m_pow;
    pow_op(int pow): m_pow(pow){
    }
    __host__ __device__ float operator()(const float& x) const{
        return pow(x, m_pow);
    }
};

void async_processor::apply(const thrust::device_vector<float>& input, std::shared_ptr<thrust::device_vector<float>>& output, cudaStream_t stream) const{
    if(!output)
        output.reset(new thrust::device_vector<float>());
    output->resize(input.size());
    thrust::transform(thrust::system::cuda::par.on(stream), input.begin(), input.end(), output->begin(), pow_op(2));
}


