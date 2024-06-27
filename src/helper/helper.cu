#include "helper/helper.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

// Templated function to print contents of a thrust device vector given a raw pointer and size
template <typename T>
void printDeviceVector(const T* d_ptr, int size) {
    thrust::host_vector<T> h_vec(size);
    cudaMemcpy(thrust::raw_pointer_cast(h_vec.data()), d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
}

__device__ void printSample(float* x, int sampleDim) {
    for (int i = 0; i < sampleDim; ++i) {
        printf("%f ", x[i]);
    }
    printf("\n");
}