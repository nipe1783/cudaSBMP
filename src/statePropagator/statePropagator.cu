#include "statePropagator/statePropagator.cuh"
#include "config/config.h"

#define WORKSPACE_DIM 2

__device__ bool propagateAndCheck(float* x0, float* x1, int numDisc, float agentLength, curandState* seed, float* obstacles,
                                  int obstaclesCount, float width, float height)
{
    float a        = UNI_MIN_ACC + curand_uniform(seed) * (UNI_MAX_ACC - UNI_MIN_ACC);
    float steering = UNI_MIN_STEERING + curand_uniform(seed) * (UNI_MAX_STEERING - UNI_MIN_STEERING);
    float duration = UNI_MIN_DT + curand_uniform(seed) * (UNI_MAX_DT - UNI_MIN_DT);
    float dt       = duration / NUM_DISC;

    float x     = x0[0];
    float y     = x0[1];
    float theta = x0[2];
    float v     = x0[3];

    float cosTheta, sinTheta, tanSteering;
    float bbMin[DIM], bbMax[DIM];

    bool motionValid = true;
    for(int i = 0; i < NUM_DISC; i++)
        {
            float x0State[DIM] = {x, y};
            cosTheta           = cos(theta);
            sinTheta           = sin(theta);
            tanSteering        = tan(steering);

            // --- State Propagation ---
            x += v * cosTheta * dt;
            y += v * sinTheta * dt;
            theta += (v / UNI_LENGTH) * tanSteering * dt;
            v += a * dt;
            float x1State[DIM] = {x, y};

            // --- Workspace Limit Check ---
            if(x < 0 || x > WS_SIZE || y < 0 || y > WS_SIZE)
                {
                    motionValid = false;
                    break;
                }

            // --- Obstacle Collision Check ---
            for(int d = 0; d < DIM; d++)
                {
                    if(x0State[d] > x1State[d])
                        {
                            bbMin[d] = x1State[d];
                            bbMax[d] = x0State[d];
                        }
                    else
                        {
                            bbMin[d] = x0State[d];
                            bbMax[d] = x1State[d];
                        }
                }

            motionValid = motionValid && isMotionValid(x0State, x1State, bbMin, bbMax, obstacles, obstaclesCount);
            if(!motionValid) break;
        }

    x1[0] = x, x1[1] = y, x1[2] = theta, x1[3] = v, x1[4] = a, x1[5] = steering, x1[6] = duration;
    return motionValid;
}

__device__ bool
propagateAndCheck_gb(float* x0, float* x1, int numDisc, float agentLength, curandState* seed, float* obstacles, int obstaclesCount,
                     float width, float height, int* selR1Edge, int* valR1Edge, float R1Size, int N, int* hashTable, int tableSize)
{
    // Generate random controls
    float a        = UNI_MIN_ACC + curand_uniform(seed) * (UNI_MAX_ACC - UNI_MIN_ACC);
    float steering = UNI_MIN_STEERING + curand_uniform(seed) * (UNI_MAX_STEERING - UNI_MIN_STEERING);
    float duration = UNI_MIN_DT + curand_uniform(seed) * (UNI_MAX_DT - UNI_MIN_DT);
    float dt       = duration / NUM_DISC;

    float x     = x0[0];
    float y     = x0[1];
    float theta = x0[2];
    float v     = x0[3];

    float cos_theta, sin_theta, tan_steering;
    float bbMin[WORKSPACE_DIM], bbMax[WORKSPACE_DIM];

    bool motionValid = true;

    int x0R1 = getR1_gb(x0[0], x0[1], R1Size, N);
    int x1R1;
    for(int i = 0; i < numDisc; ++i)
        {
            float v_state[WORKSPACE_DIM] = {x, y};
            cos_theta                    = cosf(theta);
            sin_theta                    = sinf(theta);
            tan_steering                 = tanf(steering);

            // Propagate the state
            x += v * cos_theta * dt;
            y += v * sin_theta * dt;
            x1R1 = getR1_gb(x, y, R1Size, N);

            if(x <= 0.0 || x >= width || y <= 0.0 || y >= height)
                {
                    motionValid = false;
                    break;
                }

            theta += (v / agentLength) * tan_steering * dt;
            v += a * dt;

            float w_state[WORKSPACE_DIM] = {x, y};

            for(int d = 0; d < WORKSPACE_DIM; ++d)
                {
                    if(v_state[d] > w_state[d])
                        {
                            bbMin[d] = w_state[d];
                            bbMax[d] = v_state[d];
                        }
                    else
                        {
                            bbMin[d] = v_state[d];
                            bbMax[d] = w_state[d];
                        }
                }

            motionValid   = motionValid && isMotionValid(v_state, w_state, bbMin, bbMax, obstacles, obstaclesCount);
            int edgeIndex = getEdgeIndex(x0R1, x1R1, hashTable, tableSize);
            if(edgeIndex != -1)
                {
                    atomicAdd(&selR1Edge[edgeIndex], 1);
                }
            x0R1 = x1R1;
            if(motionValid)
                {
                    if(edgeIndex != -1)
                        {
                            atomicAdd(&valR1Edge[edgeIndex], 1);
                        }
                }
        }

    x1[0] = x;
    x1[1] = y;
    x1[2] = theta;
    x1[3] = v;
    x1[4] = a;
    x1[5] = steering;
    x1[6] = duration;

    return motionValid;
}