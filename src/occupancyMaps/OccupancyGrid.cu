#include "occupancyMaps/OccupancyGrid.cuh"

// Default constructor
OccupancyGrid::OccupancyGrid() : width_(0), height_(0), N_(0), numEdges_(0), cellSize_(0) {}

// Parameterized constructor
OccupancyGrid::OccupancyGrid(float width, float height, int N) : width_(width), height_(height), N_(N), cellSize_(width / N) {
    numEdges_ = N * N * 4;  // Max 4 edges per cell
    grid_.resize(N * N, 0);
}

// Method to construct fromNodes
std::vector<int> OccupancyGrid::constructFromNodes() {
    std::vector<int> fromNodes;

    for(int row = 0; row < N_; ++row)
        {
            for(int col = 0; col < N_; ++col)
                {
                    int currentNode = row * N_ + col;

                    // Add edge to the cell above if not on the top edge
                    if(row > 0)
                        {
                            fromNodes.push_back(currentNode);
                        }

                    // Add edge to the cell to the left if not on the left edge
                    if(col > 0)
                        {
                            fromNodes.push_back(currentNode);
                        }

                    // Add edge to the cell below if not on the bottom edge
                    if(row < N_ - 1)
                        {
                            fromNodes.push_back(currentNode);
                        }

                    // Add edge to the cell to the right if not on the right edge
                    if(col < N_ - 1)
                        {
                            fromNodes.push_back(currentNode);
                        }
                }
        }

    return fromNodes;
}

// Method to construct toNodes
std::vector<int> OccupancyGrid::constructToNodes() {
    std::vector<int> toNodes;

    for(int row = 0; row < N_; ++row)
        {
            for(int col = 0; col < N_; ++col)
                {
                    int currentNode = row * N_ + col;

                    // Add edge to the cell above if not on the top edge
                    if(row > 0)
                        {
                            toNodes.push_back((row - 1) * N_ + col);
                        }

                    // Add edge to the cell to the left if not on the left edge
                    if(col > 0)
                        {
                            toNodes.push_back(row * N_ + (col - 1));
                        }

                    // Add edge to the cell below if not on the bottom edge
                    if(row < N_ - 1)
                        {
                            toNodes.push_back((row + 1) * N_ + col);
                        }

                    // Add edge to the cell to the right if not on the right edge
                    if(col < N_ - 1)
                        {
                            toNodes.push_back(row * N_ + (col + 1));
                        }
                }
        }

    return toNodes;
}

// Method to construct vertexArray
std::vector<int> OccupancyGrid::constructVertexArray() {
    std::vector<int> vertexArray(N_ * N_);
    int edgeIdx = 0;

    for(int row = 0; row < N_; ++row)
        {
            for(int col = 0; col < N_; ++col)
                {
                    int currentNode = row * N_ + col;
                    vertexArray[currentNode] = edgeIdx;

                    // Increment edge index by the number of edges for the current node
                    if(row > 0) edgeIdx++;
                    if(col > 0) edgeIdx++;
                    if(row < N_ - 1) edgeIdx++;
                    if(col < N_ - 1) edgeIdx++;
                }
        }

    return vertexArray;
}

// Method to construct edgeArray and weightArray
std::vector<int> OccupancyGrid::constructEdgeAndWeightArrays() {
    std::vector<int> edgeArray;
    std::vector<float> weightArray;

    for(int row = 0; row < N_; ++row)
        {
            for(int col = 0; col < N_; ++col)
                {
                    int currentNode = row * N_ + col;

                    // Add edge to the cell above if not on the top edge
                    if(row > 0)
                        {
                            edgeArray.push_back((row - 1) * N_ + col);
                            weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
                        }

                    // Add edge to the cell to the left if not on the left edge
                    if(col > 0)
                        {
                            edgeArray.push_back(row * N_ + (col - 1));
                            weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
                        }

                    // Add edge to the cell below if not on the bottom edge
                    if(row < N_ - 1)
                        {
                            edgeArray.push_back((row + 1) * N_ + col);
                            weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
                        }

                    // Add edge to the cell to the right if not on the right edge
                    if(col < N_ - 1)
                        {
                            edgeArray.push_back(row * N_ + (col + 1));
                            weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
                        }
                }
        }

    return edgeArray;
}

__host__ __device__ int getR1_gb(float x, float y, float R1Size, int N) {
    int cellX = static_cast<int>(x / R1Size);
    int cellY = static_cast<int>(y / R1Size);
    if(cellX >= 0 && cellX < N && cellY >= 0 && cellY < N)
        {
            return cellY * N + cellX;
        }
    return -1;  // Invalid cell
}

__host__ __device__ int getR2_gb(float x, float y, int r1, float R1Size, int N, float R2Size, int n) {
    if(r1 == -1)
        {
            return -1;  // Invalid R1 cell, so R2 is also invalid
        }

    int cellY_R1 = r1 / N;
    int cellX_R1 = r1 % N;

    // Calculate the local coordinates within the R1 cell
    float localX = x - cellX_R1 * R1Size;
    float localY = y - cellY_R1 * R1Size;

    int cellX_R2 = static_cast<int>(localX / R2Size);
    int cellY_R2 = static_cast<int>(localY / R2Size);
    if(cellX_R2 >= 0 && cellX_R2 < n && cellY_R2 >= 0 && cellY_R2 < n)
        {
            int localR2 = cellY_R2 * n + cellX_R2;
            return r1 * (n * n) + localR2;  // Flattened index
        }
    return -1;  // Invalid subcell
}

__device__ int hashFunction(int key, int size) {
    return key % size;
}

__global__ void initHashMap(int* fromNodes, int* toNodes, int* edgeIndices, int* hashTable, int tableSize, int numEdges) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= numEdges) return;

    int key = fromNodes[tid] * 100000 + toNodes[tid];
    int hash = hashFunction(key, tableSize);

    while(atomicCAS(&hashTable[2 * hash], -1, key) != -1)
        {
            hash = (hash + 1) % tableSize;
        }
    hashTable[2 * hash + 1] = edgeIndices[tid];
}

__device__ int getEdgeIndex(int fromNode, int toNode, int* hashTable, int tableSize) {
    int key = fromNode * 100000 + toNode;
    int hash = hashFunction(key, tableSize);

    while(hashTable[2 * hash] != key)
        {
            if(hashTable[2 * hash] == -1)
                {
                    return -1;  // Edge not found
                }
            hash = (hash + 1) % tableSize;
        }
    return hashTable[2 * hash + 1];
}
