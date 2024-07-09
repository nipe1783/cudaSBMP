#include "occupancyMaps/OccupancyGrid.h"

// Default constructor
OccupancyGrid::OccupancyGrid() : width_(0), height_(0), N_(0), numEdges_(0), cellSize_(0) {}

// Parameterized constructor
OccupancyGrid::OccupancyGrid(float width, float height, int N)
    : width_(width), height_(height), N_(N), cellSize_(width / N) {
    numEdges_ = N * N * N * N; // Max N*N edges per cell
    grid_.resize(N * N, 0);
}

// Method to construct fromNodes
std::vector<int> OccupancyGrid::constructFromNodes() {
    std::vector<int> fromNodes;

    for (int row = 0; row < N_; ++row) {
        for (int col = 0; col < N_; ++col) {
            int currentNode = row * N_ + col;
            for (int r = 0; r < N_; ++r) {
                for (int c = 0; c < N_; ++c) {
                    fromNodes.push_back(currentNode);
                }
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
            for (int r = 0; r < N_; ++r) {
                for (int c = 0; c < N_; ++c) {
                    toNodes.push_back(r * N_ + c);
                }
            }
        }
    }

    return toNodes;
}