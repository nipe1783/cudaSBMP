#pragma once

#include <iostream>
#include <vector>

class OccupancyGrid {
public:
    // Constructor
    OccupancyGrid();
    OccupancyGrid(float width, float height, int N);

    // Methods
    std::vector<int> constructFromNodes();
    std::vector<int> constructToNodes();

private:
    // Fields
    float width_;
    float height_;
    int N_;
    int numEdges_;
    float cellSize_;
    std::vector<int> grid_;
};
