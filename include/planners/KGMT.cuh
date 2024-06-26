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
        int numIterations_, maxSamples_, numDisc_, N_, sampleDim_, treeSize_;
        float width_, height_, cellSize_;
        OccupancyGrid grid_;
        thrust::device_vector<bool> d_eOpen_, d_eClosed_, d_G_, d_edges_;
        thrust::device_vector<int> d_scanIdx_, d_activeGIdx_;
        thrust::device_vector<float> d_samples_;
        bool *d_eOpen_ptr_, *d_eClosed_ptr_, *d_G_ptr_, *d_edges_ptr_;
        float *d_samples_ptr_, *d_costGoal, costGoal;
        int *d_scanIdx_ptr_, *d_activeGIdx_ptr_;
};

__global__ void fillG(int numSamples, bool* G, int* scanIdx, int* activeGIdx);
__global__ void expandG(float* samples, int* activeG, int activeSize);