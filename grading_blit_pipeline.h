#pragma once

#include <vulkan/vulkan.h>

class Engine2D;

VkPipeline createGradingBlitPipeline(Engine2D* engine, VkPipelineLayout pipelineLayout);
void destroyGradingBlitPipeline(Engine2D* engine, VkPipeline pipeline);
