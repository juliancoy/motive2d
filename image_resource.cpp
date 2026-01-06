#include "image_resource.h"
#include "engine2d.h"

#include <iostream>
#include <cstring>

ImageResource::~ImageResource()
{
    
    if (!engine)
        return;
    
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
    // This constructor is weird: it takes a reference to another ImageResource (res)
    // but doesn't use it? Possibly it's meant to be a factory that ensures the resource exists.
    // We'll implement it as a simple initialization that delegates to ensureImageResource.
    // However ensureImageResource is not defined yet. For now, we'll just set members.
    this->engine = engine;
    this->image = VK_NULL_HANDLE;
    this->memory = VK_NULL_HANDLE;
    this->view = VK_NULL_HANDLE;
    this->format = format;
    this->width = width;
    this->height = height;
    this->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    recreated = false;
    
    // If width/height/format are zero/undefined, leave as null.
    if (width == 0 || height == 0 || format == VK_FORMAT_UNDEFINED)
    {
        return;
    }
    
    // Actually, we should create the image. But the existing code is broken.
    // We'll implement a simple creation path.
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
        return;
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
        return;
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
        vkDestroyImage(engine->logicalDevice, image, nullptr);
        vkFreeMemory(engine->logicalDevice, memory, nullptr);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return;
    }
    
    this->format = format;
    this->width = width;
    this->height = height;
    this->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    recreated = true;
}

bool ImageResource::ensure(uint32_t width,
                           uint32_t height,
                           VkFormat format,
                           bool& recreated,
                           VkImageUsageFlags usage)
{
    // If the resource already matches the requested parameters, nothing to do.
    if (image != VK_NULL_HANDLE && this->width == width && this->height == height && this->format == format)
    {
        recreated = false;
        return true;
    }
    
    // If width/height/format are zero/undefined, leave as null.
    if (width == 0 || height == 0 || format == VK_FORMAT_UNDEFINED)
    {
        recreated = false;
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
        vkDestroyImage(engine->logicalDevice, image, nullptr);
        vkFreeMemory(engine->logicalDevice, memory, nullptr);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return false;
    }
    
    this->format = format;
    this->width = width;
    this->height = height;
    this->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    recreated = true;
    return true;
}

bool ImageResource::uploadImageData(
                     const void* data,
                     size_t dataSize,
                     uint32_t width,
                     uint32_t height,
                     VkFormat format,
                     VkImageUsageFlags usage)
{
    // Create a staging buffer.
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4; // assuming RGBA8
    engine->createBuffer(imageSize,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stagingBuffer,
                         stagingBufferMemory);
    
    // Map and copy data.
    void* mapped = nullptr;
    vkMapMemory(engine->logicalDevice, stagingBufferMemory, 0, imageSize, 0, &mapped);
    if (!mapped)
    {
        std::cerr << "[Video2D] Failed to map staging buffer memory." << std::endl;
        vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);
        return false;
    }
    std::memcpy(mapped, data, dataSize);
    vkUnmapMemory(engine->logicalDevice, stagingBufferMemory);
    
    // Copy buffer to image.
    // Need copyBufferToImage function (defined in engine2d.cpp).
    // We'll declare it externally.
    extern void copyBufferToImage(Engine2D*, VkBuffer, VkImage, VkImageLayout, uint32_t, uint32_t);
    
    // First ensure the image resource exists
    bool recreated = false;
    if (!ensure(width, height, format, recreated, usage)) {
        vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);
        return false;
    }
    
    copyBufferToImage(engine, stagingBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, width, height);
    
    // Cleanup staging buffer.
    vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);
    
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}
