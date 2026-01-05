#include "scrubber.h"

#include "engine2d.h"
#include "utils.h"

#include <stdexcept>

Scrubber::Scrubber(Engine2D* engine, VkPipelineLayout pipelineLayout)
{
    if (!engine || pipelineLayout == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Invalid parameters while creating scrubber pipeline");
    }

    auto scrubCode = readSPIRVFile("shaders/scrubber.comp.spv");
    VkShaderModule scrubModule = engine->createShaderModule(scrubCode);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = scrubModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(engine->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        vkDestroyShaderModule(engine->logicalDevice, scrubModule, nullptr);
        throw std::runtime_error("Failed to create scrubber compute pipeline");
    }

    vkDestroyShaderModule(engine->logicalDevice, scrubModule, nullptr);
}

Scrubber::~Scrubber()
{
    if (engine && pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
    }
}

void Run(
                          uint32_t groupX,
                          uint32_t groupY)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScrubberPushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, groupX, groupY, 1);
    
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
