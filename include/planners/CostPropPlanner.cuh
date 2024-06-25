#pragma once

#include "planners/Planner.cuh"

class CostPropPlanner : public Planner
{
    public:
        // constructor
        CostPropPlanner() = default;

        // methods
        void plan(float* start, float* goal) override;
        void generateRandomTree(const float* root, const int numSamples, float **samples) override;
};