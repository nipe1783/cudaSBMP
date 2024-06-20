#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

int main(void){

    const int N = 6;
    int h_data[N] = {1, 2, 3, 4, 5, 6};

    // Transfer data to the device
    thrust::device_vector<int> d_data(h_data, h_data + N);

    // Apply a transformation (multiply each element by 2)
    thrust::transform(d_data.begin(), d_data.end(), d_data.begin(), thrust::placeholders::_1 * 2);

    // Transfer data back to the host
    thrust::copy(d_data.begin(), d_data.end(), h_data);

    // Print the results
    std::cout << "Result: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    printf("TEST2\n");
    return 0;
}