#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

class Engine2D;

struct ScrubberPushConstants
{
    glm::vec2 resolution;
    float progress;
    float isPlaying;
    float _pad = 0.0f;
};

VkPipeline createScrubberPipeline(Engine2D* engine, VkPipelineLayout pipelineLayout);
void destroyScrubberPipeline(Engine2D* engine, VkPipeline pipeline);

void dispatchScrubberPass(VkCommandBuffer commandBuffer,
                          VkPipeline pipeline,
                          VkPipelineLayout pipelineLayout,
                          VkDescriptorSet descriptorSet,
                          const ScrubberPushConstants& pushConstants,
                          uint32_t groupX,
                          uint32_t groupY);
