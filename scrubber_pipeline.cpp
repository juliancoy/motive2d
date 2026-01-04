#include "scrubber_pipeline.h"

#include "engine2d.h"
#include "utils.h"

#include <stdexcept>

VkPipeline createScrubberPipeline(Engine2D* engine, VkPipelineLayout pipelineLayout)
{
    if (!engine || pipelineLayout == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Invalid parameters while creating scrubber pipeline");
    }

    auto scrubCode = readSPIRVFile("shaders/scrubber_blit.comp.spv");
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
    return pipeline;
}

void destroyScrubberPipeline(Engine2D* engine, VkPipeline pipeline)
{
    if (engine && pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
    }
}
