#include "overlay.hpp"

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
namespace
{
void copyBufferToImage(Engine2D* engine,
                       VkBuffer stagingBuffer,
                       VkImage targetImage,
                       VkImageLayout currentLayout,
                       uint32_t width,
                       uint32_t height)
{
    VkCommandBuffer cmd = engine->beginSingleTimeCommands();

    VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toTransfer.oldLayout = currentLayout;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = targetImage;
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.baseMipLevel = 0;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.baseArrayLayer = 0;
    toTransfer.subresourceRange.layerCount = 1;
    toTransfer.srcAccessMask = (currentLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : VK_ACCESS_SHADER_READ_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkPipelineStageFlags srcStage = (currentLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                                        ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(cmd,
                         srcStage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toTransfer);

    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(cmd, stagingBuffer, targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier toShader{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.image = targetImage;
    toShader.subresourceRange = toTransfer.subresourceRange;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toShader);

    engine->endSingleTimeCommands(cmd);
}
} // namespace

void destroyOverlayCompute(OverlayCompute& comp)
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
}

void destroyImageResource(Engine2D* engine, ImageResource& res)
{
    if (!engine)
    {
        return;
    }
    if (res.view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(engine->logicalDevice, res.view, nullptr);
        res.view = VK_NULL_HANDLE;
    }
    if (res.image != VK_NULL_HANDLE)
    {
        vkDestroyImage(engine->logicalDevice, res.image, nullptr);
        res.image = VK_NULL_HANDLE;
    }
    if (res.memory != VK_NULL_HANDLE)
    {
        vkFreeMemory(engine->logicalDevice, res.memory, nullptr);
        res.memory = VK_NULL_HANDLE;
    }
    res.format = VK_FORMAT_UNDEFINED;
    res.width = 0;
    res.height = 0;
}

bool ensureImageResource(Engine2D* engine,
                         ImageResource& res,
                         uint32_t width,
                         uint32_t height,
                         VkFormat format,
                         bool& recreated,
                         VkImageUsageFlags usage)
{
    recreated = false;
    if (res.image != VK_NULL_HANDLE && res.width == width && res.height == height && res.format == format)
    {
        return true;
    }

    destroyImageResource(engine, res);
    recreated = true;

    if (width == 0 || height == 0 || format == VK_FORMAT_UNDEFINED)
    {
        return false;
    }

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(engine->logicalDevice, &imageInfo, nullptr, &res.image) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create image." << std::endl;
        return false;
    }

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(engine->logicalDevice, res.image, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = engine->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(engine->logicalDevice, &allocInfo, nullptr, &res.memory) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to allocate image memory." << std::endl;
        vkDestroyImage(engine->logicalDevice, res.image, nullptr);
        res.image = VK_NULL_HANDLE;
        return false;
    }

    vkBindImageMemory(engine->logicalDevice, res.image, res.memory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = res.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(engine->logicalDevice, &viewInfo, nullptr, &res.view) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create image view." << std::endl;
        destroyImageResource(engine, res);
        return false;
    }

    res.format = format;
    res.width = width;
    res.height = height;
    return true;
}

bool uploadImageData(Engine2D* engine,
                     ImageResource& res,
                     const void* data,
                     size_t dataSize,
                     uint32_t width,
                     uint32_t height,
                     VkFormat format)
{
    if (!data || dataSize == 0 || width == 0 || height == 0)
    {
        return false;
    }

    bool recreated = false;
    if (!ensureImageResource(engine, res, width, height, format, recreated))
    {
        return false;
    }

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    engine->createBuffer(static_cast<VkDeviceSize>(dataSize),
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         staging,
                         stagingMemory);

    void* mapped = nullptr;
    vkMapMemory(engine->logicalDevice, stagingMemory, 0, static_cast<VkDeviceSize>(dataSize), 0, &mapped);
    std::memcpy(mapped, data, dataSize);
    vkUnmapMemory(engine->logicalDevice, stagingMemory);

    VkImageLayout oldLayout = (!recreated && res.view != VK_NULL_HANDLE)
                                  ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                  : VK_IMAGE_LAYOUT_UNDEFINED;
    copyBufferToImage(engine, staging, res.image, oldLayout, width, height);

    vkDestroyBuffer(engine->logicalDevice, staging, nullptr);
    vkFreeMemory(engine->logicalDevice, stagingMemory, nullptr);
    return true;
}

void runOverlayCompute(Engine2D* engine,
                       OverlayCompute& comp,
                       ImageResource& target,
                       uint32_t width,
                       uint32_t height,
                       const glm::vec2& rectCenter,
                       const glm::vec2& rectSize,
                       float outerThickness,
                       float innerThickness,
                       float detectionEnabled,
                       const glm::vec2& detectionBoxCenter,
                       const glm::vec2& detectionBoxSize)
{
    bool recreated = false;
    if (!ensureImageResource(engine, target, width, height, VK_FORMAT_R8G8B8A8_UNORM, recreated,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
    {
        return;
    }

    comp.width = width;
    comp.height = height;

    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageInfo.imageView = target.view;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = comp.descriptorSet;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &storageInfo;
    vkUpdateDescriptorSets(comp.device, 1, &write, 0, nullptr);

    vkResetCommandBuffer(comp.commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(comp.commandBuffer, &beginInfo);

    struct OverlayPush {
        glm::vec2 outputSize;
        glm::vec2 rectCenter;
        glm::vec2 rectSize;
        float outerThickness;
        float innerThickness;
        float detectionEnabled;
        glm::vec2 detectionBoxCenter;
        glm::vec2 detectionBoxSize;
    } push{glm::vec2(static_cast<float>(width), static_cast<float>(height)), rectCenter, rectSize, outerThickness, innerThickness,
           detectionEnabled, detectionBoxCenter, detectionBoxSize};

    const VkImageLayout initialLayout = recreated ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageMemoryBarrier toGeneralBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toGeneralBarrier.oldLayout = initialLayout;
    toGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.image = target.image;
    toGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneralBarrier.subresourceRange.baseMipLevel = 0;
    toGeneralBarrier.subresourceRange.levelCount = 1;
    toGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    toGeneralBarrier.subresourceRange.layerCount = 1;
    toGeneralBarrier.srcAccessMask = (initialLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                         ? VK_ACCESS_SHADER_READ_BIT
                                         : 0;
    toGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    VkPipelineStageFlags srcStage = (initialLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                        ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                        : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    vkCmdPipelineBarrier(comp.commandBuffer,
                         srcStage,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toGeneralBarrier);

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
                       sizeof(OverlayPush),
                       &push);

    const uint32_t groupX = (width + 15) / 16;
    const uint32_t groupY = (height + 15) / 16;
    vkCmdDispatch(comp.commandBuffer, groupX, groupY, 1);

    VkImageMemoryBarrier toReadBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toReadBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadBarrier.image = target.image;
    toReadBarrier.subresourceRange = toGeneralBarrier.subresourceRange;
    toReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(comp.commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toReadBarrier);

    vkEndCommandBuffer(comp.commandBuffer);

    vkResetFences(comp.device, 1, &comp.fence);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &comp.commandBuffer;
    vkQueueSubmit(engine->graphicsQueue, 1, &submitInfo, comp.fence);
    vkWaitForFences(comp.device, 1, &comp.fence, VK_TRUE, UINT64_MAX);
}

bool initializeOverlayCompute(Engine2D* engine, OverlayCompute& comp)
{
    comp.device = engine->logicalDevice;
    comp.queue = engine->graphicsQueue;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(comp.device, &layoutInfo, nullptr, &comp.descriptorSetLayout) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create overlay descriptor set layout" << std::endl;
        return false;
    }

    std::vector<char> shaderCode;
    try
    {
        shaderCode = readSPIRVFile("shaders/overlay_rect.comp.spv");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Video2D] " << ex.what() << std::endl;
        destroyOverlayCompute(comp);
        return false;
    }

    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::vec2) * 5 + sizeof(float) * 3; // outputSize, rectCenter, rectSize, detectionBoxCenter, detectionBoxSize (5 vec2) + outerThickness, innerThickness, detectionEnabled (3 float)

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &comp.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(comp.device, &pipelineLayoutInfo, nullptr, &comp.pipelineLayout) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create overlay pipeline layout" << std::endl;
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
        destroyOverlayCompute(comp);
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = comp.pipelineLayout;

    if (vkCreateComputePipelines(comp.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &comp.pipeline) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create overlay compute pipeline" << std::endl;
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
        destroyOverlayCompute(comp);
        return false;
    }

    vkDestroyShaderModule(comp.device, shaderModule, nullptr);

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(comp.device, &poolInfo, nullptr, &comp.descriptorPool) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create overlay descriptor pool" << std::endl;
        destroyOverlayCompute(comp);
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = comp.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &comp.descriptorSetLayout;

    if (vkAllocateDescriptorSets(comp.device, &allocInfo, &comp.descriptorSet) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to allocate overlay descriptor set" << std::endl;
        destroyOverlayCompute(comp);
        return false;
    }

    VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCreateInfo.queueFamilyIndex = engine->graphicsQueueFamilyIndex;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(comp.device, &poolCreateInfo, nullptr, &comp.commandPool) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create overlay command pool" << std::endl;
        destroyOverlayCompute(comp);
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = comp.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(comp.device, &cmdAllocInfo, &comp.commandBuffer) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to allocate overlay command buffer" << std::endl;
        destroyOverlayCompute(comp);
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(comp.device, &fenceInfo, nullptr, &comp.fence) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create overlay fence" << std::endl;
        destroyOverlayCompute(comp);
        return false;
    }

    return true;
}

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
