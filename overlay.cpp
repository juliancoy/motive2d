#include "overlay.hpp"

#include <array>
#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "engine2d.h"
#include "glyph.h"
#include "utils.h"

namespace overlay
{

void updateFpsOverlay(Engine2D* engine,
                      FpsOverlayResources& fpsOverlay,
                      VkSampler overlaySampler,
                      VkSampler fallbackSampler,
                      float fpsValue,
                      uint32_t fbWidth,
                      uint32_t fbHeight)
{
    fpsOverlay.lastFpsValue = fpsValue;
    fpsOverlay.lastRefWidth = fbWidth;
    fpsOverlay.lastRefHeight = fbHeight;

    glyph::OverlayBitmap bitmap = glyph::buildFrameRateOverlay(fbWidth, fbHeight, fpsValue);
    if (bitmap.width == 0 || bitmap.height == 0 || bitmap.pixels.empty())
    {
        fpsOverlay.info.enabled = false;
        return;
    }

    if (!uploadImageData(engine,
                         fpsOverlay.image,
                         bitmap.pixels.data(),
                         bitmap.pixels.size(),
                         bitmap.width,
                         bitmap.height,
                         VK_FORMAT_R8G8B8A8_UNORM))
    {
        std::cerr << "[Video2D] Failed to upload FPS overlay image." << std::endl;
        fpsOverlay.info.enabled = false;
        return;
    }

    VkSampler sampler = (overlaySampler != VK_NULL_HANDLE) ? overlaySampler : fallbackSampler;
    fpsOverlay.info.overlay.view = fpsOverlay.image.view;
    fpsOverlay.info.overlay.sampler = sampler;
    fpsOverlay.info.extent = {bitmap.width, bitmap.height};
    fpsOverlay.info.offset = {static_cast<int32_t>(bitmap.offsetX), static_cast<int32_t>(bitmap.offsetY)};
    fpsOverlay.info.enabled = true;
}

} // namespace overlay
