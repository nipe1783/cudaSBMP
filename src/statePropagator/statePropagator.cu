#include "statePropagator/statePropagator.cuh"

#define WORKSPACE_DIM 2

__device__
bool propagateAndCheck(
    float* x0, 
    float* x1, 
    int numDisc, 
    float agentLength, 
    curandState* state, 
    float* obstacles, 
    int obstaclesCount,
    float width,
    float height) {
    // Generate random controls
    float a = curand_uniform(state) * 10.0f - 5.0f;  // a between -10 and 10
    float steering = curand_uniform(state) * 2.0f * M_PI - M_PI;  // steering between -π and π
    float duration = curand_uniform(state) * 1.0f + 0.05f;  // duration between 0.1 and 0.5

    float dt = duration / numDisc;
    float x = x0[0];
    float y = x0[1];
    float theta = x0[2];
    float v = x0[3];

    float cos_theta, sin_theta, tan_steering;
    float bbMin[WORKSPACE_DIM], bbMax[WORKSPACE_DIM];

    bool motionValid = true;
    for (int i = 0; i < numDisc; ++i) {
        
        float v_state[WORKSPACE_DIM] = {x, y};
        cos_theta = cosf(theta);
        sin_theta = sinf(theta);
        tan_steering = tanf(steering);

        // Propagate the state
        x += v * cos_theta * dt;
        y += v * sin_theta * dt;
        // TODO: update this
        if(x <= 0.0 || x >= width|| y <= 0.0 || y >= height) {
            motionValid = false;
            break;
        }
        theta += (v / agentLength) * tan_steering * dt;
        v += a * dt;

        float w_state[WORKSPACE_DIM] = {x, y};

        for (int d = 0; d < WORKSPACE_DIM; ++d) {
            if (v_state[d] > w_state[d]) {
                bbMin[d] = w_state[d];
                bbMax[d] = v_state[d];
            } else {
                bbMin[d] = v_state[d];
                bbMax[d] = w_state[d];
            }
        }

        motionValid = motionValid && isMotionValid(v_state, w_state, bbMin, bbMax, obstacles, obstaclesCount);
        if (!motionValid) {
            break;
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

__device__
bool propagateAndCheck_gb(
    float* x0, 
    float* x1, 
    int numDisc, 
    float agentLength, 
    curandState* state, 
    float* obstacles, 
    int obstaclesCount,
    float width,
    float height,
    int* selR1Edge,
    int* valR1Edge,
    float R1Size,
    int N,
    int* hashTable,
    int tableSize) {
    
    // Generate random controls
    float a = curand_uniform(state) * 10.0f - 5.0f;  // a between -10 and 10
    float steering = curand_uniform(state) * 2.0f * M_PI - M_PI;  // steering between -π and π
    float duration = curand_uniform(state) * 1.0f + 0.05f;  // duration between 0.1 and 0.5

    float dt = duration / numDisc;
    float x = x0[0];
    float y = x0[1];
    float theta = x0[2];
    float v = x0[3];

    float cos_theta, sin_theta, tan_steering;
    float bbMin[WORKSPACE_DIM], bbMax[WORKSPACE_DIM];

    bool motionValid = true;

    int x0R1 = getR1_gb(x0[0], x0[1], R1Size, N);
    int x1R1;
    for (int i = 0; i < numDisc; ++i) {
        
        float v_state[WORKSPACE_DIM] = {x, y};
        cos_theta = cosf(theta);
        sin_theta = sinf(theta);
        tan_steering = tanf(steering);

        // Propagate the state
        x += v * cos_theta * dt;
        y += v * sin_theta * dt;
        x1R1 = getR1_gb(x, y, R1Size, N);
        
        if(x <= 0.0 || x >= width || y <= 0.0 || y >= height) {
            motionValid = false;
            break;
        }
        
        theta += (v / agentLength) * tan_steering * dt;
        v += a * dt;

        float w_state[WORKSPACE_DIM] = {x, y};

        for (int d = 0; d < WORKSPACE_DIM; ++d) {
            if (v_state[d] > w_state[d]) {
                bbMin[d] = w_state[d];
                bbMax[d] = v_state[d];
            } else {
                bbMin[d] = v_state[d];
                bbMax[d] = w_state[d];
            }
        }

        motionValid = motionValid && isMotionValid(v_state, w_state, bbMin, bbMax, obstacles, obstaclesCount);
        int edgeIndex = getEdgeIndex(x0R1, x1R1, hashTable, tableSize);
        if(edgeIndex != -1) {
            atomicAdd(&selR1Edge[edgeIndex], 1);
        }
        if (!motionValid) {
            break;
        }
        if(edgeIndex != -1) {
            atomicAdd(&valR1Edge[edgeIndex], 1);
        }
        x0R1 = x1R1;
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