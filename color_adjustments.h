#pragma once

#include <array>
#include <glm/vec3.hpp>

constexpr size_t kCurveLutSize = 256;

struct ColorAdjustments
{
    float exposure = 0.0f;
    float contrast = 1.0f;
    float saturation = 1.0f;
    glm::vec3 shadows{1.0f};
    glm::vec3 midtones{1.0f};
    glm::vec3 highlights{1.0f};
    std::array<float, kCurveLutSize> curveLut{};
    bool curveEnabled = false;
};
