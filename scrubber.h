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

class Scrubber
{
public:
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    Scrubber(Engine2D *engine);
    ~Scrubber();
    Engine2D *engine;
    VkPipeline pipeline;
    VkCommandBuffer commandBuffer;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSet descriptorSet;
    const ScrubberPushConstants &pushConstants;
    void run(
        uint32_t groupX,
        uint32_t groupY);
};