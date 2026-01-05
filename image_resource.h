#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

// Represents an image owned by the rendering engine.
class ImageResource
{
    ImageResource(Engine2D *engine,
                  ImageResource &res,
                  uint32_t width,
                  uint32_t height,
                  VkFormat format,
                  bool &recreated,
                  VkImageUsageFlags usage);
    ~ImageResource();
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};
