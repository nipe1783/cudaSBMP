#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

class OccupancyGrid 
{
    public:

    // constructor
    OccupancyGrid() = default;
    OccupancyGrid(float width, float height, int N);


    // methods:
    __host__ __device__ int getCellIndex(float x, float y) const;
    __host__ __device__ int getOccupancy(int row, int col) const;
    void updateOccupancy(int row, int col, int n);

    // fields:
    float width_, height_, cellSize_;
    int N_;
    thrust::device_vector<int> grid_;
};