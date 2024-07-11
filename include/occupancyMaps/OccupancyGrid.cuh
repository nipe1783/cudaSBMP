#pragma once

#include <iostream>
#include <vector>
#include <cub/cub.cuh>

class OccupancyGrid
{
public:
    // Constructors
    OccupancyGrid();
    OccupancyGrid(float width, float height, int N);

    // Methods
    std::vector<int> constructFromNodes();
    std::vector<int> constructToNodes();
    std::vector<int> constructVertexArray();
    std::vector<int> constructEdgeAndWeightArrays();

private:
    // Fields
    float width_;
    float height_;
    int N_;
    int numEdges_;
    float cellSize_;
    std::vector<int> grid_;
};

// Global functions
__host__ __device__ int getR1_gb(float x, float y, float R1Size, int N);
__host__ __device__ int getR2_gb(float x, float y, int r1, float R1Size, int N, float R2Size, int n);
__device__ int hashFunction(int key, int size);
__global__ void initHashMap(int* fromNodes, int* toNodes, int* edgeIndices, int* hashTable, int tableSize, int numEdges);
__device__ int getEdgeIndex(int fromNode, int toNode, int* hashTable, int tableSize);
