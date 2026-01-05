#pragma once

#include <vulkan/vulkan.h>
#include <glm/vec2.hpp>
#include "engine2d.h"

struct CompositeBitmapCompute
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
    VkBuffer bitmapBuffer = VK_NULL_HANDLE;
    VkDeviceMemory bitmapBufferMemory = VK_NULL_HANDLE;
    VkDeviceSize bitmapBufferSize = 0;
};

struct CompositeBitmapPush
{
    glm::vec2 targetSize;
    glm::vec2 bitmapSize;
    glm::vec2 position;
    glm::vec2 scale;
    glm::vec2 pivot;
    float rotation;
    float opacity;
    float _pad = 0.0f;
};


bool initializeCompositeBitmapCompute(Engine2D* engine, CompositeBitmapCompute& comp);
void destroyCompositeBitmapCompute(CompositeBitmapCompute& comp);
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
                     float opacity = 1.0f);
