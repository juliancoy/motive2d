#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <vulkan/vulkan.h>

#include "display2d.h"
#include "overlay.hpp"

class Engine2D;

class SubtitleOverlay
{
public:
    struct Line
    {
        double start = 0.0;
        double end = 0.0;
        std::string text;
    };

    bool load(const std::filesystem::path& path);
    bool hasData() const { return !lines_.empty(); }
    std::vector<const Line*> activeLines(double currentTime, size_t maxLines = 2) const;

private:
    std::vector<Line> lines_;
    mutable size_t lastIndex_ = 0;
};

struct SubtitleOverlayResources
{
    overlay::ImageResource image;
    OverlayImageInfo info{};
};

bool updateSubtitleOverlay(Engine2D* engine,
                           SubtitleOverlayResources& resources,
                           const SubtitleOverlay& overlay,
                           double currentTime,
                           uint32_t fbWidth,
                           uint32_t fbHeight,
                           glm::vec2 overlayCenter,
                           glm::vec2 overlaySize,
                           VkSampler overlaySampler,
                           VkSampler fallbackSampler,
                           size_t maxLines = 2);

void destroySubtitleOverlayResources(Engine2D* engine, SubtitleOverlayResources& resources);
