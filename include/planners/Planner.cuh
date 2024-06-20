#pragma once
#include "state/State.h"
#include "agent/Agent.h"
#include <Eigen/Core>

class Planner {
    public:
        // constructor
        Planner() = default;

        // methods
        virtual void plan() = 0;
        float* generateRandomSamples(const State &state, int numSamples);

};