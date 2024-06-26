#pragma once

#include "planners/Planner.cuh"
#include "occupancyMaps/OccupancyGrid.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class KGMT
{
    public:
        // constructor
        KGMT() = default;
        KGMT(float width, float height, int N, int numIterations, int numSamples, int numDisc, int sampleDim);

        // methods
        void plan(float* initial, float* goal);

        // fields
        int numIterations_, numSamples_, numDisc_, N_, sampleDim_;
        float width_, height_, cellSize_;
        OccupancyGrid grid_;
        thrust::device_vector<bool> d_eOpen_, d_eClosed_, d_G_;
        thrust::device_vector<float> d_samples_;
        bool *d_eOpen_ptr_, *d_eClosed_ptr_, *d_G_ptr_;
        float *d_samples_ptr_;
};