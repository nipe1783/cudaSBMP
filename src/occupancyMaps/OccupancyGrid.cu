#include "occupancyMaps/OccupancyGrid.cuh"

// Default constructor
OccupancyGrid::OccupancyGrid() : width_(0), height_(0), N_(0), numEdges_(0), cellSize_(0) {}

// Parameterized constructor
OccupancyGrid::OccupancyGrid(float width, float height, int N)
    : width_(width), height_(height), N_(N), cellSize_(width / N) {
    numEdges_ = N * N * 4; // Max 4 edges per cell
    grid_.resize(N * N, 0);
}

// Method to construct fromNodes
std::vector<int> OccupancyGrid::constructFromNodes() {
    std::vector<int> fromNodes;

    for (int row = 0; row < N_; ++row) {
        for (int col = 0; col < N_; ++col) {
            int currentNode = row * N_ + col;

            // Add edge to the cell above if not on the top edge
            if (row > 0) {
                fromNodes.push_back(currentNode);
            }

            // Add edge to the cell to the left if not on the left edge
            if (col > 0) {
                fromNodes.push_back(currentNode);
            }

            // Add edge to the cell below if not on the bottom edge
            if (row < N_ - 1) {
                fromNodes.push_back(currentNode);
            }

            // Add edge to the cell to the right if not on the right edge
            if (col < N_ - 1) {
                fromNodes.push_back(currentNode);
            }
        }
    }

    return fromNodes;
}

// Method to construct toNodes
std::vector<int> OccupancyGrid::constructToNodes() {
    std::vector<int> toNodes;

    for (int row = 0; row < N_; ++row) {
        for (int col = 0; col < N_; ++col) {
            int currentNode = row * N_ + col;

            // Add edge to the cell above if not on the top edge
            if (row > 0) {
                toNodes.push_back((row - 1) * N_ + col);
            }

            // Add edge to the cell to the left if not on the left edge
            if (col > 0) {
                toNodes.push_back(row * N_ + (col - 1));
            }

            // Add edge to the cell below if not on the bottom edge
            if (row < N_ - 1) {
                toNodes.push_back((row + 1) * N_ + col);
            }

            // Add edge to the cell to the right if not on the right edge
            if (col < N_ - 1) {
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

    for (int row = 0; row < N_; ++row) {
        for (int col = 0; col < N_; ++col) {
            int currentNode = row * N_ + col;
            vertexArray[currentNode] = edgeIdx;

            // Increment edge index by the number of edges for the current node
            if (row > 0) edgeIdx++;
            if (col > 0) edgeIdx++;
            if (row < N_ - 1) edgeIdx++;
            if (col < N_ - 1) edgeIdx++;
        }
    }

    return vertexArray;
}

// Method to construct edgeArray and weightArray
std::pair<std::vector<int>, std::vector<float>> OccupancyGrid::constructEdgeAndWeightArrays() {
    std::vector<int> edgeArray;
    std::vector<float> weightArray;

    for (int row = 0; row < N_; ++row) {
        for (int col = 0; col < N_; ++col) {
            int currentNode = row * N_ + col;

            // Add edge to the cell above if not on the top edge
            if (row > 0) {
                edgeArray.push_back((row - 1) * N_ + col);
                weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
            }

            // Add edge to the cell to the left if not on the left edge
            if (col > 0) {
                edgeArray.push_back(row * N_ + (col - 1));
                weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
            }

            // Add edge to the cell below if not on the bottom edge
            if (row < N_ - 1) {
                edgeArray.push_back((row + 1) * N_ + col);
                weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
            }

            // Add edge to the cell to the right if not on the right edge
            if (col < N_ - 1) {
                edgeArray.push_back(row * N_ + (col + 1));
                weightArray.push_back(static_cast<float>(rand() % 1000) / 1000.0f);
            }
        }
    }

    return std::make_pair(edgeArray, weightArray);
}
