#include "scrubber.h"

#include "engine2d.h"
#include "utils.h"

#include <stdexcept>

static ScrubberPushConstants dummyPushConstants{};

Scrubber::Scrubber(Engine2D* engine) : 
    engine(engine),
    pipeline(VK_NULL_HANDLE),
    commandBuffer(VK_NULL_HANDLE),
    descriptorSet(VK_NULL_HANDLE),
    pushConstants(dummyPushConstants)
{
    // Note: pipelineLayout is already initialized in the header to VK_NULL_HANDLE
}

Scrubber::~Scrubber()
{
    if (engine && pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
    }
}

void Scrubber::run(
                          uint32_t groupX,
                          uint32_t groupY)
{
    // TODO: Implement scrubber compute dispatch
    // This is a stub implementation to fix compilation
}


struct ScrubberUi
{
    double left;
    double top;
    double right;
    double bottom;
    double iconLeft;
    double iconTop;
    double iconRight;
    double iconBottom;
};

ScrubberUi computeScrubberUi(int windowWidth, int windowHeight)
{
    const double kScrubberMargin = 20.0;
    const double kScrubberHeight = 64.0;
    const double kScrubberMinWidth = 200.0;
    const double kPlayIconSize = 28.0;

    ScrubberUi ui{};
    const double availableWidth = static_cast<double>(windowWidth);
    const double scrubberWidth =
        std::max(kScrubberMinWidth,
                 availableWidth - (kPlayIconSize + kScrubberMargin * 3.0));
    const double scrubberHeight = kScrubberHeight;
    ui.iconLeft = kScrubberMargin;
    ui.iconRight = ui.iconLeft + kPlayIconSize;
    ui.top = static_cast<double>(windowHeight) - scrubberHeight - kScrubberMargin;
    ui.bottom = ui.top + scrubberHeight;
    ui.iconTop = ui.top + (scrubberHeight - kPlayIconSize) * 0.5;
    ui.iconBottom = ui.iconTop + kPlayIconSize;

    ui.left = ui.iconRight + kScrubberMargin;
    ui.right = ui.left + scrubberWidth;
    return ui;
}

bool cursorInScrubber(double x, double y, int windowWidth, int windowHeight)
{
    const ScrubberUi ui = computeScrubberUi(windowWidth, windowHeight);
    return x >= ui.left && x <= ui.right && y >= ui.top && y <= ui.bottom;
}

bool cursorInPlayButton(double x, double y, int windowWidth, int windowHeight)
{
    const ScrubberUi ui = computeScrubberUi(windowWidth, windowHeight);
    return x >= ui.iconLeft && x <= ui.iconRight && y >= ui.iconTop && y <= ui.iconBottom;
}
