#include "collisionCheck/collisionCheck.cuh"
#include "config/config.h"

__device__ bool isBroadPhaseValid(float *bbMin, float *bbMax, float *obs)
{
    for(int d = 0; d < DIM; ++d)
        {
            if(bbMax[d] <= obs[d] || obs[DIM + d] <= bbMin[d]) return true;
        }
    return false;
}

__device__ bool isMotionValid(float *x0, float *x1, float *bbMin, float *bbMax, float *obstacles, int obstaclesCount)
{
    for(int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx)
        {
            float obs[2 * DIM];
            for(int d = 0; d < DIM; ++d)
                {
                    obs[d]       = obstacles[obsIdx * 2 * DIM + d];
                    obs[DIM + d] = obstacles[obsIdx * 2 * DIM + DIM + d];
                }
            if(!isBroadPhaseValid(bbMin, bbMax, obs)) return false;
        }
    return true;
}
