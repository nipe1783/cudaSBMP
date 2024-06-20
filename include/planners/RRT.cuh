#pragma once

#include "planners/Planner.cuh"

class RRT : public Planner
{
    public:
        // constructor
        RRT() = default;

        // methods
        void plan() override;
};