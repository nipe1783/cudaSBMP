#include "planners/KGMT.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"
#include "state/State.h"
#include "helper/helper.cuh"
#include "helper/helper.cu"
#include <curand_kernel.h>
#include <chrono>

KGMT::KGMT(float width, float height, int N, int numIterations, int numSamples, int numDisc, int sampleDim)
    : width_(width), height_(height), N_(N), sampleDim_(sampleDim), numIterations_(numIterations), numSamples_(numSamples), numDisc_(numDisc),
    cellSize_(width / N), grid_(width, height, N) {

    d_eOpen_ = thrust::device_vector<bool>(numSamples);
    d_eClosed_ = thrust::device_vector<bool>(numSamples);
    d_G_ = thrust::device_vector<bool>(numSamples);
    d_samples_ = thrust::device_vector<float>(numSamples * sampleDim);

    bool *d_eOpen_ptr_ = thrust::raw_pointer_cast(d_eOpen_.data());
    bool *d_eClosed_ptr_ = thrust::raw_pointer_cast(d_eClosed_.data());
    bool *d_G_ptr_ = thrust::raw_pointer_cast(d_G_.data());
    float *d_samples_ptr_ = thrust::raw_pointer_cast(d_samples_.data());

}

void KGMT::plan(float* initial, float* goal){
    printf("TEST\n");
}