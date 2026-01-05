#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

class Engine2D;

// Represents an image owned by the rendering engine.
class ImageResource
{
public:
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
    Engine2D *engine = nullptr;
};
