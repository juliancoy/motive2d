#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "image_resource.h"
#include "engine2d.h"
#include "display2d.h"


class FpsOverlay
{
public:
    ImageResource image;
    float lastFpsValue = -1.0f;
    uint32_t lastRefWidth = 0;
    uint32_t lastRefHeight = 0;
    VkSampler sampler = VK_NULL_HANDLE;
    Engine2D* engine = nullptr;
    FpsOverlay(Engine2D* engine);
    ~FpsOverlay();
    void updateFpsOverlay(
        VkSampler overlaySampler,
        VkSampler fallbackSampler,
        float fpsValue,
        uint32_t fbWidth,
        uint32_t fbHeight);
    std::string formatFpsText(float fps);

};
