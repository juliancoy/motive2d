#pragma once

#include <vulkan/vulkan.h>

class Engine2D;

VkPipeline createScrubberPipeline(Engine2D* engine, VkPipelineLayout pipelineLayout);
void destroyScrubberPipeline(Engine2D* engine, VkPipeline pipeline);
