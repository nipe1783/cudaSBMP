#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "planners/Planner.cuh"
#include "planners/RRT.cuh"

int main(void){

    // Create a planner
    RRT rrt;
    float* samples1 = rrt.generateRandomSamples(State(0, 0, 0, 0), 256);

    // Print samples
    printf("Samples1:\n");
    for (int i = 0; i < 256; i++){
        std::cout << samples1[i * 4] << " " << samples1[i * 4 + 1] << " " << samples1[i * 4 + 2] << " " << samples1[i * 4 + 3] << std::endl;
    }

    float* samples2 = rrt.generateRandomSamples(State(1, 1, 1, 1), 256);

    // Print samples
    printf("Samples2:\n");
    for (int i = 0; i < 256; i++){
        std::cout << samples2[i * 4] << " " << samples2[i * 4 + 1] << " " << samples2[i * 4 + 2] << " " << samples2[i * 4 + 3] << std::endl;
    }

    // Print Samples 1:
    printf("Samples1:\n");
    for (int i = 0; i < 256; i++){
        std::cout << samples1[i * 4] << " " << samples1[i * 4 + 1] << " " << samples1[i * 4 + 2] << " " << samples1[i * 4 + 3] << std::endl;
    }
    return 0;
}