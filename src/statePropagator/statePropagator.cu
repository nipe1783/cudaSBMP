#include "statePropagator/statePropagator.cuh"

#define WORKSPACE_DIM 2

__device__
bool propagateAndCheck(float* x0, float* x1, int numDisc, float agentLength, curandState* state, float* obstacles, int obstaclesCount) {
    // Generate random controls
    float a = curand_uniform(state) * 10.0f - 5.0f;  // a between -10 and 10
    float steering = curand_uniform(state) * 2.0f * M_PI - M_PI;  // steering between -π and π
    float duration = curand_uniform(state) * 0.4f + 0.1f;  // duration between 0.1 and 0.5

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