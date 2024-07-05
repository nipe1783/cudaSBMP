#pragma once

__device__
bool broadPhaseCC(float *bbMin, float *bbMax, float *obs);

__device__
bool isMotionValid(float* x0, float* x1, int obstaclesCount, float* obstacles);