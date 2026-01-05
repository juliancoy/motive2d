#include "fps.h"

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "composite_bitmap.h"
#include "engine2d.h"
#include "text.h"
#include "utils.h"
#include "engine2d.h"
#include "widgets.hpp"

widgets::WidgetRenderer& fpsWidgetRenderer()
{
    static widgets::WidgetRenderer renderer{};
    return renderer;
}

FpsOverlay::FpsOverlay(Engine2D* engine) : engine(engine)
{
}

CompositeBitmapCompute& fpsBitmapCompute()
{
    static CompositeBitmapCompute compute{};
    return compute;
}

std::string FpsOverlay::formatFpsText(float fps)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.1f FPS", fps);
    return std::string(buffer);
}


FpsOverlay::~FpsOverlay()
{
    if (sampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(engine->logicalDevice, sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }
}

void FpsOverlay::updateFpsOverlay(
                      VkSampler overlaySampler,
                      VkSampler fallbackSampler,
                      float fpsValue,
                      uint32_t fbWidth,
                      uint32_t fbHeight)
{
    lastFpsValue = fpsValue;
    lastRefWidth = fbWidth;
    lastRefHeight = fbHeight;

    if (!ensureFpsWidgetRenderer(engine) || !ensureFpsCompositeCompute(engine))
    {
        info.enabled = false;
        return;
    }

    const std::string text = formatFpsText(fpsValue);
    fonts::FontBitmap bitmap = fonts::renderText(text, 26u);
    if (bitmap.width == 0 || bitmap.height == 0 || bitmap.pixels.empty())
    {
        info.enabled = false;
        return;
    }

    constexpr uint32_t kPadding = 14;
    const uint32_t overlayWidth = std::max<uint32_t>(1, bitmap.width + kPadding * 2);
    const uint32_t overlayHeight = std::max<uint32_t>(1, bitmap.height + kPadding * 2);

    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    bool recreated = false;
    if (!ensureImageResource(engine,
                             image,
                             overlayWidth,
                             overlayHeight,
                             VK_FORMAT_R8G8B8A8_UNORM,
                             recreated,
                             usage))
    {
        info.enabled = false;
        return;
    }
    (void)recreated;

    std::vector<widgets::DrawCommand> commands;
    commands.reserve(2);

    widgets::DrawCommand background{};
    background.type = widgets::CMD_RECT;
    background.color = glm::vec4(0.05f, 0.05f, 0.05f, 0.85f);
    background.params = glm::vec4(static_cast<float>(overlayWidth) * 0.5f,
                                  static_cast<float>(overlayHeight) * 0.5f,
                                  static_cast<float>(overlayWidth),
                                  static_cast<float>(overlayHeight));
    commands.push_back(background);

    if (overlayWidth > 8 && overlayHeight > 8)
    {
        widgets::DrawCommand border{};
        border.type = widgets::CMD_RECT;
        border.color = glm::vec4(0.8f, 0.8f, 0.8f, 0.15f);
        const float innerWidth = std::max(static_cast<float>(overlayWidth) - 8.0f, 1.0f);
        const float innerHeight = std::max(static_cast<float>(overlayHeight) - 8.0f, 1.0f);
        border.params = glm::vec4(static_cast<float>(overlayWidth) * 0.5f,
                                  static_cast<float>(overlayHeight) * 0.5f,
                                  innerWidth,
                                  innerHeight);
        commands.push_back(border);
    }

    if (!widgets::runWidgetRenderer(engine,
                                    fpsWidgetRenderer(),
                                    image,
                                    overlayWidth,
                                    overlayHeight,
                                    commands,
                                    true))
    {
        info.enabled = false;
        return;
    }

    if (!compositeBitmap(engine,
                                  fpsBitmapCompute(),
                                  image,
                                  overlayWidth,
                                  overlayHeight,
                                  bitmap.pixels.data(),
                                  bitmap.pixels.size(),
                                  bitmap.width,
                                  bitmap.height,
                                  glm::vec2(static_cast<float>(kPadding), static_cast<float>(kPadding)),
                                  glm::vec2(1.0f, 1.0f),
                                  0.0f,
                                  glm::vec2(0.0f, 0.0f),
                                  1.0f))
    {
        info.enabled = false;
        return;
    }

    VkSampler sampler = (overlaySampler != VK_NULL_HANDLE) ? overlaySampler : fallbackSampler;
    info.overlay.view = image.view;
    info.overlay.sampler = sampler;
    info.extent = {overlayWidth, overlayHeight};
    info.offset = {16, 16};
    info.enabled = true;
    lastRefWidth = overlayWidth;
    lastRefHeight = overlayHeight;
}
