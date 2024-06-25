#pragma once
#include "state/State.h"
#include "agent/Agent.h"
#include <Eigen/Core>

class Planner {
    public:
        // methods
        virtual void plan(float* root, float* goal) = 0;
        virtual void generateRandomTree(const float* root, const int numSamples, float **samples) = 0;
        void checkCudaError(cudaError_t err, const char* msg);

};