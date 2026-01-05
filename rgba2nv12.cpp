#include "rgba2nv12.h"

#include "engine2d.h"
#include "utils.h"

#include <stdexcept>

VkPipeline createRgbaToNv12Pipeline(Engine2D* engine, VkPipelineLayout pipelineLayout)
{
    if (!engine || pipelineLayout == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Invalid parameters while creating RGBA-to-NV12 pipeline");
    }

    auto shaderCode = readSPIRVFile("shaders/rgba2nv12.comp.spv");
    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(engine->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
        throw std::runtime_error("Failed to create RGBA-to-NV12 compute pipeline");
    }

    vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
    return pipeline;
}

void destroyRgbaToNv12Pipeline(Engine2D* engine, VkPipeline pipeline)
{
    if (engine && pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
    }
}

void dispatchRgbaToNv12(VkCommandBuffer commandBuffer,
                        VkPipeline pipeline,
                        VkPipelineLayout pipelineLayout,
                        VkDescriptorSet descriptorSet,
                        const RgbaToNv12PushConstants& pushConstants,
                        uint32_t groupX,
                        uint32_t groupY)
{
    if (commandBuffer == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE)
    {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RgbaToNv12PushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, groupX, groupY, 1);
}
