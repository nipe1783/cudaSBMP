#include "helper/helper.cuh"

__device__ void printSample(float* x, int sampleDim) {
    for (int i = 0; i < sampleDim; ++i) {
        printf("%f ", x[i]);
    }
    printf("\n");
}