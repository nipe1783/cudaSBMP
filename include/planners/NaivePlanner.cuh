#pragma once

#include "planners/Planner.cuh"

class NaivePlanner : public Planner
{
    public:
        // constructor
        NaivePlanner() = default;

        // methods
        void plan(float* start, float* goal) override;
        void generateRandomTree(const float* root, const int numSamples, float **samples) override;
};