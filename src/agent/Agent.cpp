#include "agent/Agent.h"
#include <cmath>

Agent::Agent(std::vector<Eigen::Vector2d>& verticesCCW) : verticesCCW_(verticesCCW) {}

Agent::Agent(float x, float y, float length, float theta, float v) {
    verticesCCW_.push_back(Eigen::Vector2d(x, y));
    verticesCCW_.push_back(Eigen::Vector2d(x + length, y));
    verticesCCW_.push_back(Eigen::Vector2d(x + length, y + length));
    verticesCCW_.push_back(Eigen::Vector2d(x, y + length));

    x_ = x;
    y_ = y;
    length_ = length;
    theta_ = theta;
    v_ = v;
}

void Agent::updateState(float a, float delta, float dt) {
    // Kinematic bicycle model equations
    x_ += v_ * cos(theta_) * dt;
    y_ += v_ * sin(theta_) * dt;
    theta_ += (v_ / length_) * tan(delta) * dt;
    v_ += a * dt;
}