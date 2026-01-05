#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "fps.h"

class Engine2D;

namespace widgets
{
enum CommandType : uint32_t
{
    CMD_RECT = 0u,
    CMD_CIRCLE = 1u,
    CMD_LINE = 2u,
    CMD_GRID = 3u,
};

struct DrawCommand
{
    uint32_t type = 0;
    uint32_t padding[3] = {0, 0, 0};
    glm::vec4 color = glm::vec4(0.0f);
    glm::vec4 params = glm::vec4(0.0f);
    glm::vec4 params2 = glm::vec4(0.0f);
};

struct ButtonDescriptor
{
    glm::vec2 center{0.0f, 0.0f};
    glm::vec2 size{0.0f, 0.0f};
    glm::vec4 backgroundColor{0.2f, 0.2f, 0.2f, 1.0f};
    glm::vec4 borderColor{0.05f, 0.05f, 0.05f, 1.0f};
    float borderThickness = 2.0f;
};

struct SliderDescriptor
{
    glm::vec2 start{0.0f, 0.0f};
    glm::vec2 end{0.0f, 0.0f};
    float trackThickness = 6.0f;
    glm::vec4 trackColor{0.3f, 0.3f, 0.3f, 1.0f};
    glm::vec4 progressColor{0.2f, 0.6f, 0.95f, 1.0f};
    float value = 0.5f;
    float handleRadius = 10.0f;
    glm::vec4 handleColor{0.96f, 0.96f, 0.96f, 1.0f};
    glm::vec4 handleBorderColor{0.0f, 0.0f, 0.0f, 0.85f};
};

struct WidgetPushConstants
{
    glm::vec2 outputSize;
    uint32_t clearFirst = 0;
    uint32_t padding = 0;
};
static_assert(sizeof(WidgetPushConstants) == 16, "Push constant size must match shader");

struct WidgetRenderer
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
    VkBuffer commandBufferStorage = VK_NULL_HANDLE;
    VkDeviceMemory commandBufferMemory = VK_NULL_HANDLE;
    VkDeviceSize commandBufferSize = 0;
};

bool initializeWidgetRenderer(Engine2D* engine, WidgetRenderer& renderer);
void destroyWidgetRenderer(WidgetRenderer& renderer);
bool ensureWidgetCommandStorage(Engine2D* engine, WidgetRenderer& renderer, VkDeviceSize size);

void appendButtonCommands(std::vector<DrawCommand>& commands, const ButtonDescriptor& descriptor);
void appendSliderCommands(std::vector<DrawCommand>& commands, const SliderDescriptor& descriptor);

bool runWidgetRenderer(Engine2D* engine,
                       WidgetRenderer& renderer,
                       ImageResource& target,
                       uint32_t width,
                       uint32_t height,
                       const std::vector<DrawCommand>& commands,
                       bool clearFirst);
} // namespace widgets
