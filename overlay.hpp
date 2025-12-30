#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "display2d.h"

class Engine2D;

namespace overlay
{

struct ImageResource
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct OverlayResources
{
    OverlayImageInfo info;
    ImageResource image;
    VkSampler sampler = VK_NULL_HANDLE;
};

struct FpsOverlayResources
{
    OverlayImageInfo info;
    ImageResource image;
    float lastFpsValue = -1.0f;
    uint32_t lastRefWidth = 0;
    uint32_t lastRefHeight = 0;
};

struct DetectionEntry
{
    glm::vec4 bbox;      // normalized x, y, width, height
    glm::vec4 color;
    float confidence;
    int classId;
    int padding[2];
};

struct OverlayCompute
{
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkBuffer detectionBuffer = VK_NULL_HANDLE;
    VkDeviceMemory detectionBufferMemory = VK_NULL_HANDLE;
    VkDeviceSize detectionBufferSize = 0;
};

bool ensureImageResource(Engine2D* engine,
                         ImageResource& res,
                         uint32_t width,
                         uint32_t height,
                         VkFormat format,
                         bool& recreated,
                         VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

void destroyImageResource(Engine2D* engine, ImageResource& res);

bool uploadImageData(Engine2D* engine,
                     ImageResource& res,
                     const void* data,
                     size_t dataSize,
                     uint32_t width,
                     uint32_t height,
                     VkFormat format);

bool initializeOverlayCompute(Engine2D* engine, OverlayCompute& comp);
void destroyOverlayCompute(OverlayCompute& comp);

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
                       const DetectionEntry* detections,
                       uint32_t detectionCount);

void updateFpsOverlay(Engine2D* engine,
                      FpsOverlayResources& fpsOverlay,
                      VkSampler overlaySampler,
                      VkSampler fallbackSampler,
                      float fpsValue,
                      uint32_t fbWidth,
                      uint32_t fbHeight);

} // namespace overlay
