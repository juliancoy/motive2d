#pragma once

#include <vulkan/vulkan.h>
#include <glm/vec2.hpp>

class Engine2D;

struct RgbaToNv12PushConstants
{
    glm::ivec2 rgbaSize;
    glm::ivec2 uvSize;
    int colorSpace;
    int colorRange;
};

VkPipeline createRgbaToNv12Pipeline(Engine2D* engine, VkPipelineLayout pipelineLayout);
void destroyRgbaToNv12Pipeline(Engine2D* engine, VkPipeline pipeline);
void dispatchRgbaToNv12(VkCommandBuffer commandBuffer,
                         VkPipeline pipeline,
                         VkPipelineLayout pipelineLayout,
                         VkDescriptorSet descriptorSet,
                         const RgbaToNv12PushConstants& pushConstants,
                         uint32_t groupX,
                         uint32_t groupY);
