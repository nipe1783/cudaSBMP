#include "statePropagator/statePropagator.cuh"

__device__
bool propagateAndCheck(float* x0, float* x1, int numDisc, float agentLength, curandState* state){
    // Generate random controls
    float a = curand_uniform(state) * 20.0f - 10.0f;  // a between -5 and 5
    float steering = curand_uniform(state) * 2.0f * M_PI - M_PI;  // steering between -π and π
    float duration = curand_uniform(state) * .4f + 0.1f;  // duration between 0.1 and 0.5

    float dt = duration / numDisc;
    float x = x0[0];
    float y = x0[1];
    float theta = x0[2];
    float v = x0[3];

    float cos_theta, sin_theta, tan_steering;

    for (int i = 0; i < numDisc; i++) {
        cos_theta = cosf(theta);
        sin_theta = sinf(theta);
        tan_steering = tanf(steering);

        x += v * cos_theta * dt;
        y += v * sin_theta * dt;
        theta += (v / agentLength) * tan_steering * dt;
        v += a * dt;
    }

    x1[0] = x;
    x1[1] = y;
    x1[2] = theta;
    x1[3] = v;
    x1[4] = a;
    x1[5] = steering;
    x1[6] = duration;
    //TODO: Update this.

    return true;
}