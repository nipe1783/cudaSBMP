#pragma once
#include <vector>
#include <Eigen/Core>

// kinematic bicycle model:
class Agent {
    public:
        // Constructors
        Agent() = default;
        Agent(std::vector<Eigen::Vector2d>& verticesCCW);
        Agent(float x, float y, float length, float theta = 0.0f, float v = 0.0f);

        // Methods
        void updateState(float a, float delta, float dt);
        
        // Fields
        std::vector<Eigen::Vector2d> verticesCCW_;
        float x_;       // Position x
        float y_;       // Position y
        float theta_;   // Orientation
        float v_;       // Velocity
        float length_;  // Wheelbase
        float width_;   // Width

};