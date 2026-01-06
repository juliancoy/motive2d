#include "nv12toBGR.h"

#include "engine2d.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

nv12toBGR::nv12toBGR(Engine2D* engine,
                       uint32_t groupX,
                       uint32_t groupY)
{
    this->engine = engine;
    if (!engine || pipelineLayout == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Invalid parameters while creating NV12-to-BGR pipeline");
    }

    auto shaderCode = readSPIRVFile("shaders/nv12toBGR.spv");
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
        throw std::runtime_error("Failed to create NV12-to-BGR compute pipeline");
    }

    vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
}

nv12toBGR::~nv12toBGR()
{
    if (engine && pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
    }
}

void nv12toBGR::run()
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
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(nv12toBGRPushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, groupX, groupY, 1);
}

bool convertNv12ToBgr(const uint8_t* nv12,
                      size_t yBytes,
                      size_t uvBytes,
                      int width,
                      int height,
                      std::vector<uint8_t>& bgr)
{
    if (!nv12 || width <= 0 || height <= 0 || yBytes == 0 || uvBytes == 0)
    {
        return false;
    }

    const size_t framePixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (bgr.size() != framePixels * 3)
    {
        bgr.resize(framePixels * 3);
    }

    const uint8_t* yPlane = nv12;
    const uint8_t* uvPlane = nv12 + yBytes;

    for (int row = 0; row < height; ++row)
    {
        const uint8_t* yRow = yPlane + static_cast<size_t>(row) * static_cast<size_t>(width);
        const uint8_t* uvRow = uvPlane + static_cast<size_t>(row / 2) * static_cast<size_t>(width);
        for (int col = 0; col < width; ++col)
        {
            const float Y = static_cast<float>(yRow[col]);
            const float U = static_cast<float>(uvRow[(col / 2) * 2]) - 128.0f;
            const float V = static_cast<float>(uvRow[(col / 2) * 2 + 1]) - 128.0f;

            const float y = std::max(0.0f, (Y - 16.0f)) * 1.164383f;
            const float r = y + 1.596027f * V;
            const float g = y - 0.391762f * U - 0.812968f * V;
            const float b = y + 2.017232f * U;

            const size_t dstIndex = (static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)) * 3;
            bgr[dstIndex + 0] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
            bgr[dstIndex + 1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
            bgr[dstIndex + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
        }
    }

    return true;
}
