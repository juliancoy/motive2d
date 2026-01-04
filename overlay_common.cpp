#include "overlay.hpp"

#include <array>
#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "engine2d.h"
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
    res.layout = VK_IMAGE_LAYOUT_UNDEFINED;
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
    res.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

bool uploadImageData(Engine2D* engine,
                     ImageResource& res,
                     const void* data,
                     size_t dataSize,
                     uint32_t width,
                     uint32_t height,
                     VkFormat format,
                     VkImageUsageFlags usage)
{
    if (!data || dataSize == 0 || width == 0 || height == 0)
    {
        return false;
    }

    bool recreated = false;
    if (!ensureImageResource(engine, res, width, height, format, recreated, usage))
    {
        return false;
    }

    // Use a staging buffer for optimal performance
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;
    engine->createBuffer(imageSize,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stagingBuffer,
                         stagingBufferMemory);

    void* mapped = nullptr;
    vkMapMemory(engine->logicalDevice, stagingBufferMemory, 0, imageSize, 0, &mapped);
    if (mapped)
    {
        std::memcpy(mapped, data, dataSize);
        vkUnmapMemory(engine->logicalDevice, stagingBufferMemory);
    }
    else
    {
        vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);
        std::cerr << "[Video2D] Failed to map staging buffer memory." << std::endl;
        return false;
    }

    copyBufferToImage(engine, stagingBuffer, res.image, VK_IMAGE_LAYOUT_UNDEFINED, width, height);

    vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);

    res.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

} // namespace overlay
