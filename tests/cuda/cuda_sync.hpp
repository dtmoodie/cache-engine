#pragma once
#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/device_vector.h>

struct async_processor
{
    void apply(const thrust::device_vector<float>& input,
               std::shared_ptr<thrust::device_vector<float>>& output,
               cudaStream_t stream) const;
};
