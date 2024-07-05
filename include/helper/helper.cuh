#pragma once

#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <math.h>
#include <fstream>
#include "cuda.h"
#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#define _USE_MATH_DEFINES

// A macro for checking the error codes of cuda runtime calls
#define CUDA_ERROR_CHECK(expr) \
  {                            \
    cudaError_t err = expr;    \
    if (err != cudaSuccess)    \
    {                          \
      printf("CUDA call failed!\n\t%s\n", cudaGetErrorString(err)); \
      exit(1);                 \
    }                          \
  }

template <typename T>
void printDeviceVector(const T* d_ptr, int size);

__device__ void printSample(float* x, int sampleDim);

template <typename T>
void writeVectorToCSV(const thrust::host_vector<T>& vec, const std::string& filename, int rows, int cols);

template <typename T>
void copyAndWriteVectorToCSV(const thrust::device_vector<T>& d_vec, const std::string& filename, int rows, int cols);

// Implement the template functions in the header file

template <typename T>
void printDeviceVector(const T* d_ptr, int size) {
    thrust::host_vector<T> h_vec(size);
    cudaMemcpy(thrust::raw_pointer_cast(h_vec.data()), d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void writeVectorToCSV(const thrust::host_vector<T>& vec, const std::string& filename, int rows, int cols) {
    std::ofstream file;
    file.open(filename);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << vec[i * cols + j];
            if (j < cols - 1) {
                file << ",";
            }
        }
        file << std::endl;
    }

    file.close();
}

template <typename T>
void copyAndWriteVectorToCSV(const thrust::device_vector<T>& d_vec, const std::string& filename, int rows, int cols) {
    thrust::host_vector<T> h_vec(d_vec.size());
    cudaMemcpy(thrust::raw_pointer_cast(h_vec.data()), thrust::raw_pointer_cast(d_vec.data()), d_vec.size() * sizeof(T), cudaMemcpyDeviceToHost);
    writeVectorToCSV(h_vec, filename, rows, cols);
}
