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

    bool uploadImageData(
        const void *data,
        size_t dataSize,
        uint32_t width,
        uint32_t height,
        VkFormat format,
        VkImageUsageFlags usage);

    // Ensure the image resource matches the given parameters.
    // If the existing image matches width/height/format, does nothing.
    // Otherwise destroys the old resource and creates a new one.
    // Returns true on success, false on failure.
    bool ensure(uint32_t width,
                uint32_t height,
                VkFormat format,
                bool &recreated,
                VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    Engine2D *engine = nullptr;
};
