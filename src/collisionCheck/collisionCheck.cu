#include "collisionCheck/collisionCheck.cuh"

#define DIM_WORKSPACE 2


__device__
bool isBroadPhaseValid(float *bbMin, float *bbMax, float *obs) 
{
	for (int d = 0; d < DIM_WORKSPACE; ++d) {
		if (bbMax[d] <= obs[d] || obs[DIM_WORKSPACE+d] <= bbMin[d]) 
			return true;
	}
	return false;
}

__device__
bool isMotionValid(float* x0, float* x1, float *bbMin, float *bbMax, float* obstacles, int obstaclesCount){
    for (int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx) {
        float obs[2*DIM_WORKSPACE];
        for (int d = 0; d < DIM_WORKSPACE; ++d) {
            obs[d] = obstacles[obsIdx*2*DIM_WORKSPACE + d];
            obs[DIM_WORKSPACE+d] = obstacles[obsIdx*2*DIM_WORKSPACE + DIM_WORKSPACE + d];
        }
        if(!isBroadPhaseValid(bbMin, bbMax, obs))
            return false;
	}
	return true;
}

