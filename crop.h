#pragma once
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

// Compute push constants structure shared across all compute pipelines
struct CropPushConstants
{
    glm::vec2 outputSize;
    glm::vec2 videoSize;
    glm::vec2 targetOrigin;
    glm::vec2 targetSize;
    glm::vec2 cropOrigin;
    glm::vec2 cropSize;
};


struct CropRegion {
    float x = 0.0f;  // normalized 0-1
    float y = 0.0f;  // normalized 0-1
    float width = 1.0f;  // normalized 0-1
    float height = 1.0f; // normalized 0-1
};


class Crop
{
public:
    Crop();
    ~Crop();

    CropPushConstants cropPushConstants{};
    void run();
};
