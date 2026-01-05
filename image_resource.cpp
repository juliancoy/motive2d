#include "image_resource.h"
#include "engine2d.h"

#include <iostream>

ImageResource::~ImageResource()
{
    if (!engine)
    {
        return;
    }
    if (view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(engine->logicalDevice, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    if (image != VK_NULL_HANDLE)
    {
        vkDestroyImage(engine->logicalDevice, image, nullptr);
        image = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE)
    {
        vkFreeMemory(engine->logicalDevice, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
    format = VK_FORMAT_UNDEFINED;
    width = 0;
    height = 0;
    layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

ImageResource::ImageResource(Engine2D* engine,
                         ImageResource& res,
                         uint32_t width,
                         uint32_t height,
                         VkFormat format,
                         bool& recreated,
                         VkImageUsageFlags usage)
{
    recreated = false;
    if (image != VK_NULL_HANDLE && width == width && height == height && format == format)
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

    if (vkCreateImage(engine->logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create image." << std::endl;
        return false;
    }

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(engine->logicalDevice, image, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = engine->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(engine->logicalDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to allocate image memory." << std::endl;
        vkDestroyImage(engine->logicalDevice, image, nullptr);
        image = VK_NULL_HANDLE;
        return false;
    }

    vkBindImageMemory(engine->logicalDevice, image, memory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(engine->logicalDevice, &viewInfo, nullptr, &view) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create image view." << std::endl;
        destroyImageResource(engine, res);
        return false;
    }

    format = format;
    width = width;
    height = height;
    layout = VK_IMAGE_LAYOUT_UNDEFINED;
    
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

    copyBufferToImage(engine, stagingBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, width, height);

    vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);

    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}
