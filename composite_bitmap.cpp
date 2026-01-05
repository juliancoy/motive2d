#include "composite_bitmap.hpp"

#include "debug_logging.h"
#include "engine2d.h"
#include "utils.h"

#include <glm/glm.hpp>

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace overlay
{
namespace
{
bool ensureBitmapBuffer(Engine2D* engine,
                        CompositeBitmapCompute& comp,
                        VkDeviceSize requiredSize)
{
    if (!engine || requiredSize == 0)
    {
        return false;
    }
    if (comp.bitmapBufferSize >= requiredSize && comp.bitmapBuffer != VK_NULL_HANDLE &&
        comp.bitmapBufferMemory != VK_NULL_HANDLE)
    {
        return true;
    }

    if (comp.bitmapBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(comp.device, comp.bitmapBuffer, nullptr);
        comp.bitmapBuffer = VK_NULL_HANDLE;
    }
    if (comp.bitmapBufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(comp.device, comp.bitmapBufferMemory, nullptr);
        comp.bitmapBufferMemory = VK_NULL_HANDLE;
    }

    engine->createBuffer(requiredSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         comp.bitmapBuffer,
                         comp.bitmapBufferMemory);
    comp.bitmapBufferSize = requiredSize;
    return true;
}
} // namespace

bool initializeCompositeBitmapCompute(Engine2D* engine, CompositeBitmapCompute& comp)
{
    if (!engine)
    {
        return false;
    }

    comp.device = engine->logicalDevice;
    comp.queue = engine->graphicsQueue;

    VkDescriptorSetLayoutBinding bufferBinding{};
    bufferBinding.binding = 0;
    bufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bufferBinding.descriptorCount = 1;
    bufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding imageBinding{};
    imageBinding.binding = 1;
    imageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageBinding.descriptorCount = 1;
    imageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{bufferBinding, imageBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(comp.device, &layoutInfo, nullptr, &comp.descriptorSetLayout) != VK_SUCCESS)
    {
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(CompositeBitmapPush);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &comp.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(comp.device, &pipelineLayoutInfo, nullptr, &comp.pipelineLayout) != VK_SUCCESS)
    {
        destroyCompositeBitmapCompute(comp);
        return false;
    }

    auto shaderCode = readSPIRVFile("shaders/composite_bitmap.comp.spv");
    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = comp.pipelineLayout;

    if (vkCreateComputePipelines(comp.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &comp.pipeline) != VK_SUCCESS)
    {
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
        destroyCompositeBitmapCompute(comp);
        return false;
    }

    vkDestroyShaderModule(comp.device, shaderModule, nullptr);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(comp.device, &poolInfo, nullptr, &comp.descriptorPool) != VK_SUCCESS)
    {
        destroyCompositeBitmapCompute(comp);
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = comp.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &comp.descriptorSetLayout;

    if (vkAllocateDescriptorSets(comp.device, &allocInfo, &comp.descriptorSet) != VK_SUCCESS)
    {
        destroyCompositeBitmapCompute(comp);
        return false;
    }

    VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCreateInfo.queueFamilyIndex = engine->graphicsQueueFamilyIndex;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(comp.device, &poolCreateInfo, nullptr, &comp.commandPool) != VK_SUCCESS)
    {
        destroyCompositeBitmapCompute(comp);
        return false;
    }

    VkCommandBufferAllocateInfo allocCmd{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocCmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCmd.commandPool = comp.commandPool;
    allocCmd.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(comp.device, &allocCmd, &comp.commandBuffer) != VK_SUCCESS)
    {
        destroyCompositeBitmapCompute(comp);
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = 0;
    if (vkCreateFence(comp.device, &fenceInfo, nullptr, &comp.fence) != VK_SUCCESS)
    {
        destroyCompositeBitmapCompute(comp);
        return false;
    }

    return true;
}

void destroyCompositeBitmapCompute(CompositeBitmapCompute& comp)
{
    if (comp.fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(comp.device, comp.fence, nullptr);
        comp.fence = VK_NULL_HANDLE;
    }
    if (comp.commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(comp.device, comp.commandPool, nullptr);
        comp.commandPool = VK_NULL_HANDLE;
    }
    if (comp.descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(comp.device, comp.descriptorPool, nullptr);
        comp.descriptorPool = VK_NULL_HANDLE;
    }
    if (comp.pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(comp.device, comp.pipeline, nullptr);
        comp.pipeline = VK_NULL_HANDLE;
    }
    if (comp.pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        comp.pipelineLayout = VK_NULL_HANDLE;
    }
    if (comp.descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        comp.descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (comp.bitmapBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(comp.device, comp.bitmapBuffer, nullptr);
        comp.bitmapBuffer = VK_NULL_HANDLE;
    }
    if (comp.bitmapBufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(comp.device, comp.bitmapBufferMemory, nullptr);
        comp.bitmapBufferMemory = VK_NULL_HANDLE;
    }
    comp.bitmapBufferSize = 0;
    comp.device = VK_NULL_HANDLE;
    comp.queue = VK_NULL_HANDLE;
    comp.descriptorSet = VK_NULL_HANDLE;
}

bool compositeBitmap(Engine2D* engine,
                      CompositeBitmapCompute& comp,
                      ImageResource& target,
                      uint32_t targetWidth,
                      uint32_t targetHeight,
                     const void* bitmapPixels,
                     size_t bitmapSize,
                     uint32_t bitmapWidth,
                     uint32_t bitmapHeight,
                     const glm::vec2& position,
                     const glm::vec2& scale,
                     float rotationDegrees,
                     const glm::vec2& pivot,
                     float opacity)
{
    if (!engine || comp.pipeline == VK_NULL_HANDLE || target.image == VK_NULL_HANDLE || target.view == VK_NULL_HANDLE)
    {
        return false;
    }
    if (!bitmapPixels || bitmapWidth == 0 || bitmapHeight == 0 || targetWidth == 0 || targetHeight == 0)
    {
        return false;
    }
    const size_t expectedSize = static_cast<size_t>(bitmapWidth) * bitmapHeight * 4;
    if (bitmapSize < expectedSize)
    {
        return false;
    }

    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(bitmapWidth) * bitmapHeight * sizeof(glm::vec4);
    if (!ensureBitmapBuffer(engine, comp, bufferSize))
    {
        return false;
    }

    void* mapped = nullptr;
    vkMapMemory(comp.device, comp.bitmapBufferMemory, 0, bufferSize, 0, &mapped);
    if (!mapped)
    {
        return false;
    }
    const uint8_t* srcPixels = static_cast<const uint8_t*>(bitmapPixels);
    float* dst = static_cast<float*>(mapped);
    const float invMax = 1.0f / 255.0f;
    size_t pixelCount = static_cast<size_t>(bitmapWidth) * bitmapHeight;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        size_t srcIndex = i * 4;
        size_t dstIndex = i * 4;
        dst[dstIndex + 0] = static_cast<float>(srcPixels[srcIndex + 0]) * invMax;
        dst[dstIndex + 1] = static_cast<float>(srcPixels[srcIndex + 1]) * invMax;
        dst[dstIndex + 2] = static_cast<float>(srcPixels[srcIndex + 2]) * invMax;
        dst[dstIndex + 3] = static_cast<float>(srcPixels[srcIndex + 3]) * invMax;
    }
    vkUnmapMemory(comp.device, comp.bitmapBufferMemory);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = comp.bitmapBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = bufferSize;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = target.view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = comp.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &bufferInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = comp.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(comp.device,
                           static_cast<uint32_t>(writes.size()),
                           writes.data(),
                           0,
                           nullptr);

    vkResetCommandBuffer(comp.commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(comp.commandBuffer, &beginInfo);

    VkImageLayout initialLayout = target.layout;
    if (initialLayout == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkAccessFlags srcAccess = 0;
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (initialLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        srcAccess = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = initialLayout;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = target.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(comp.commandBuffer,
                         srcStage,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);

    glm::vec2 safeScale = scale;
    safeScale.x = safeScale.x == 0.0f ? 1.0f : safeScale.x;
    safeScale.y = safeScale.y == 0.0f ? 1.0f : safeScale.y;

    CompositeBitmapPush push{};
    push.targetSize = glm::vec2(static_cast<float>(targetWidth), static_cast<float>(targetHeight));
    push.bitmapSize = glm::vec2(static_cast<float>(bitmapWidth), static_cast<float>(bitmapHeight));
    push.position = position + glm::vec2(0.5f);
    push.scale = safeScale;
    push.pivot = pivot;
    push.rotation = glm::radians(rotationDegrees);
    push.opacity = opacity;

    vkCmdBindPipeline(comp.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, comp.pipeline);
    vkCmdBindDescriptorSets(comp.commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            comp.pipelineLayout,
                            0,
                            1,
                            &comp.descriptorSet,
                            0,
                            nullptr);
    vkCmdPushConstants(comp.commandBuffer,
                       comp.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(CompositeBitmapPush),
                       &push);

    const uint32_t groupX = (targetWidth + 15) / 16;
    const uint32_t groupY = (targetHeight + 15) / 16;
    vkCmdDispatch(comp.commandBuffer, groupX, groupY, 1);

    vkEndCommandBuffer(comp.commandBuffer);

    vkResetFences(comp.device, 1, &comp.fence);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &comp.commandBuffer;
    vkQueueSubmit(comp.queue, 1, &submitInfo, comp.fence);
    vkWaitForFences(comp.device, 1, &comp.fence, VK_TRUE, UINT64_MAX);

    target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

} // namespace overlay
