#include "occupancyMaps/OccupancyGrid.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

OccupancyGrid::OccupancyGrid(float width, float height, int N) 
        : width_(width), height_(height), 
          N_(N), cellSize_(width / N),
          grid_(N * N, 0) {}

__host__ __device__
int OccupancyGrid::getCellIndex(float x, float y) const {
    int cellX = static_cast<int>(x / cellSize_);
    int cellY = static_cast<int>(y / cellSize_);
    if (cellX >= 0 && cellX < N_ && cellY >= 0 && cellY < N_) {
        return cellY * N_ + cellX;
    }
    return -1;
}

__host__ __device__
int OccupancyGrid::getOccupancy(int row, int col) const {
    int cellIndex = getCellIndex(row, col);
    if (cellIndex != -1) {
        return grid_[cellIndex];
    }
    return -1;
}

void OccupancyGrid::updateOccupancy(int row, int col, int n) {
    int cellIndex = getCellIndex(row, col);
    if (cellIndex != -1) {
        grid_[cellIndex] = grid_[cellIndex] + n;
    }
}