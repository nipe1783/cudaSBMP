#pragma once
#include <stdio.h>

__device__
bool isBroadPhaseValid(float *bbMin, float *bbMax, float *obs);

__device__
bool isMotionValid(float* x0, float* x1, float *bbMin, float *bbMax, float* obstacles, int obstaclesCount);