#pragma once
#include "state/State.h"
#include "agent/Agent.h"
#include <Eigen/Core>

class Planner {
    public:
        // constructor
        Planner() = default;

        // methods
        virtual void plan(float* root, float* goal) = 0;
        void generateRandomSamples(const float* parent, const int numSamples, float **samples, float **controls);
        void generateRandomTree(const float* parent, const int numSamples, float **samples, float **controls);
        void generateRandomTreeV2(const float* root, const int numSamples, float **samples);

};