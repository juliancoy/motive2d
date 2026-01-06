#include "nv12toBGR.h"

#include "engine2d.h"
#include "utils.h"
#include "debug_logging.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

nv12toBGR::nv12toBGR(Engine2D* engine,
                       uint32_t groupX,
                       uint32_t groupY)
    : engine(engine), groupX(groupX), groupY(groupY)
{
    // Pipeline creation deferred until pipelineLayout is set
    // TODO: create pipeline layout and pipeline properly
}

void nv12toBGR::createPipeline()
{
    if (!engine)
    {
        throw std::runtime_error("Invalid engine while creating nv12toBGR pipeline");
    }

    // Define descriptor set layout bindings as per shader (bindings 0,1,2)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    // Binding 0: yPlane storage image (r8)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // Binding 1: uvPlane storage image (rg8)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // Binding 2: bgrOutput storage image (rgba8)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Create descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(engine->logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor set layout for nv12toBGR");
    }

    // Pipeline layout with push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(nv12toBGRPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(engine->logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        vkDestroyDescriptorSetLayout(engine->logicalDevice, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to create pipeline layout for nv12toBGR");
    }

    // Create compute pipeline
    auto shaderCode = readSPIRVFile("shaders/nv12toBGR.spv");
    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(engine->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
        vkDestroyPipelineLayout(engine->logicalDevice, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(engine->logicalDevice, descriptorSetLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        descriptorSetLayout = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to create nv12toBGR compute pipeline");
    }

    vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);

    // Debug log
    if (renderDebugEnabled())
    {
        std::cout << "[nv12toBGR] Pipeline created successfully" << std::endl;
    }
}

nv12toBGR::~nv12toBGR()
{
    if (engine)
    {
        if (pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
        }
        if (pipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(engine->logicalDevice, pipelineLayout, nullptr);
        }
        if (descriptorSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(engine->logicalDevice, descriptorSetLayout, nullptr);
        }
    }
}

void nv12toBGR::run()
{
    if (renderDebugEnabled()) {
        std::cout << "[nv12toBGR] run() called, checking conditions..." << std::endl;
        std::cout << "[nv12toBGR] commandBuffer=" << commandBuffer 
                  << ", pipeline=" << pipeline 
                  << ", pipelineLayout=" << pipelineLayout 
                  << ", descriptorSet=" << descriptorSet << std::endl;
    }
    if (commandBuffer == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE)
    {
        if (renderDebugEnabled()) {
            std::cout << "[nv12toBGR] Skipping dispatch due to null handle" << std::endl;
        }
        return;
    }

    if (renderDebugEnabled())
    {
        std::cout << "[nv12toBGR] run: groupX=" << groupX << ", groupY=" << groupY << std::endl;
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
