#include "widgets.hpp"

#include "engine2d.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace widgets
{
namespace
{
struct DrawCommandHeader
{
    uint32_t commandCount = 0;
    uint32_t padding[3] = {0, 0, 0};
};

static_assert(sizeof(DrawCommandHeader) == 16, "Draw command header must match shader layout");

inline DrawCommand makeRectCommand(const glm::vec2& center,
                                   const glm::vec2& size,
                                   const glm::vec4& color)
{
    DrawCommand cmd{};
    cmd.type = CMD_RECT;
    cmd.color = color;
    cmd.params = glm::vec4(center.x, center.y, size.x, size.y);
    return cmd;
}

inline DrawCommand makeCircleCommand(const glm::vec2& center,
                                     float radius,
                                     const glm::vec4& color)
{
    DrawCommand cmd{};
    cmd.type = CMD_CIRCLE;
    cmd.color = color;
    cmd.params = glm::vec4(center.x, center.y, radius, 0.0f);
    return cmd;
}

inline DrawCommand makeLineCommand(const glm::vec2& start,
                                   const glm::vec2& end,
                                   float thickness,
                                   const glm::vec4& color)
{
    DrawCommand cmd{};
    cmd.type = CMD_LINE;
    cmd.color = color;
    cmd.params = glm::vec4(start.x, start.y, end.x, end.y);
    cmd.params2.x = thickness;
    return cmd;
}
} // namespace

bool initializeWidgetRenderer(Engine2D* engine, WidgetRenderer& renderer)
{
    if (!engine)
    {
        return false;
    }

    renderer.device = engine->logicalDevice;
    renderer.queue = engine->graphicsQueue;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(renderer.device, &layoutInfo, nullptr, &renderer.descriptorSetLayout) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to create descriptor set layout" << std::endl;
        destroyWidgetRenderer(renderer);
        return false;
    }

    std::vector<char> shaderCode;
    try
    {
        shaderCode = readSPIRVFile("shaders/widgets.comp.spv");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Widgets] " << ex.what() << std::endl;
        destroyWidgetRenderer(renderer);
        return false;
    }

    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(WidgetPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &renderer.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(renderer.device, &pipelineLayoutInfo, nullptr, &renderer.pipelineLayout) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to create pipeline layout" << std::endl;
        vkDestroyShaderModule(renderer.device, shaderModule, nullptr);
        destroyWidgetRenderer(renderer);
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = renderer.pipelineLayout;

    if (vkCreateComputePipelines(renderer.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &renderer.pipeline) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to create compute pipeline" << std::endl;
        vkDestroyShaderModule(renderer.device, shaderModule, nullptr);
        destroyWidgetRenderer(renderer);
        return false;
    }

    vkDestroyShaderModule(renderer.device, shaderModule, nullptr);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(renderer.device, &poolInfo, nullptr, &renderer.descriptorPool) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to create descriptor pool" << std::endl;
        destroyWidgetRenderer(renderer);
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = renderer.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &renderer.descriptorSetLayout;

    if (vkAllocateDescriptorSets(renderer.device, &allocInfo, &renderer.descriptorSet) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to allocate descriptor set" << std::endl;
        destroyWidgetRenderer(renderer);
        return false;
    }

    VkCommandPoolCreateInfo commandPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commandPoolInfo.queueFamilyIndex = engine->graphicsQueueFamilyIndex;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(renderer.device, &commandPoolInfo, nullptr, &renderer.commandPool) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to create command pool" << std::endl;
        destroyWidgetRenderer(renderer);
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = renderer.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(renderer.device, &cmdAllocInfo, &renderer.commandBuffer) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to allocate command buffer" << std::endl;
        destroyWidgetRenderer(renderer);
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(renderer.device, &fenceInfo, nullptr, &renderer.fence) != VK_SUCCESS)
    {
        std::cerr << "[Widgets] Failed to create fence" << std::endl;
        destroyWidgetRenderer(renderer);
        return false;
    }

    return true;
}

void destroyWidgetRenderer(WidgetRenderer& renderer)
{
    if (renderer.fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(renderer.device, renderer.fence, nullptr);
        renderer.fence = VK_NULL_HANDLE;
    }
    if (renderer.commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(renderer.device, renderer.commandPool, nullptr);
        renderer.commandPool = VK_NULL_HANDLE;
    }
    if (renderer.descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(renderer.device, renderer.descriptorPool, nullptr);
        renderer.descriptorPool = VK_NULL_HANDLE;
    }
    if (renderer.pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(renderer.device, renderer.pipeline, nullptr);
        renderer.pipeline = VK_NULL_HANDLE;
    }
    if (renderer.pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(renderer.device, renderer.pipelineLayout, nullptr);
        renderer.pipelineLayout = VK_NULL_HANDLE;
    }
    if (renderer.descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(renderer.device, renderer.descriptorSetLayout, nullptr);
        renderer.descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (renderer.commandBufferStorage != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(renderer.device, renderer.commandBufferStorage, nullptr);
        renderer.commandBufferStorage = VK_NULL_HANDLE;
    }
    if (renderer.commandBufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(renderer.device, renderer.commandBufferMemory, nullptr);
        renderer.commandBufferMemory = VK_NULL_HANDLE;
    }
    renderer.commandBufferSize = 0;
    renderer.queue = VK_NULL_HANDLE;
    renderer.device = VK_NULL_HANDLE;
}

bool ensureWidgetCommandStorage(Engine2D* engine, WidgetRenderer& renderer, VkDeviceSize size)
{
    if (!engine)
    {
        return false;
    }

    if (size == 0)
    {
        return false;
    }

    if (renderer.commandBufferStorage != VK_NULL_HANDLE && renderer.commandBufferSize >= size)
    {
        return true;
    }

    if (renderer.commandBufferStorage != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(renderer.device, renderer.commandBufferStorage, nullptr);
        renderer.commandBufferStorage = VK_NULL_HANDLE;
    }
    if (renderer.commandBufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(renderer.device, renderer.commandBufferMemory, nullptr);
        renderer.commandBufferMemory = VK_NULL_HANDLE;
    }

    engine->createBuffer(size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         renderer.commandBufferStorage,
                         renderer.commandBufferMemory);
    if (renderer.commandBufferStorage == VK_NULL_HANDLE || renderer.commandBufferMemory == VK_NULL_HANDLE)
    {
        return false;
    }

    renderer.commandBufferSize = size;
    return true;
}

void appendButtonCommands(std::vector<DrawCommand>& commands, const ButtonDescriptor& descriptor)
{
    const glm::vec2 padding(descriptor.borderThickness * 2.0f);
    const glm::vec2 outerSize = descriptor.size + padding;
    commands.push_back(makeRectCommand(descriptor.center, outerSize, descriptor.borderColor));

    glm::vec2 innerSize = descriptor.size - padding;
    innerSize.x = std::max(innerSize.x, 0.0f);
    innerSize.y = std::max(innerSize.y, 0.0f);
    if (innerSize.x > 0.0f && innerSize.y > 0.0f)
    {
        commands.push_back(makeRectCommand(descriptor.center, innerSize, descriptor.backgroundColor));
    }
}

void appendSliderCommands(std::vector<DrawCommand>& commands, const SliderDescriptor& descriptor)
{
    const float thickness = std::max(descriptor.trackThickness, 1.0f);
    commands.push_back(makeLineCommand(descriptor.start, descriptor.end, thickness, descriptor.trackColor));

    const float clampedValue = std::clamp(descriptor.value, 0.0f, 1.0f);
    const glm::vec2 direction = descriptor.end - descriptor.start;
    const glm::vec2 handlePos = descriptor.start + direction * clampedValue;
    if (clampedValue > 0.0f)
    {
        commands.push_back(makeLineCommand(descriptor.start, handlePos, thickness, descriptor.progressColor));
    }

    commands.push_back(makeCircleCommand(handlePos, descriptor.handleRadius, descriptor.handleColor));
    if (descriptor.handleBorderColor.a > 0.0f)
    {
        commands.push_back(makeCircleCommand(handlePos, descriptor.handleRadius + thickness * 0.25f, descriptor.handleBorderColor));
    }
}

bool runWidgetRenderer(Engine2D* engine,
                       WidgetRenderer& renderer,
                       ImageResource& target,
                       uint32_t width,
                       uint32_t height,
                       const std::vector<DrawCommand>& commands,
                       bool clearFirst)
{
    if (!engine || width == 0 || height == 0 || target.view == VK_NULL_HANDLE || target.image == VK_NULL_HANDLE)
    {
        return false;
    }

    const size_t count = commands.size();
    if (count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    constexpr VkDeviceSize headerSize = sizeof(DrawCommandHeader);
    const VkDeviceSize dataSize = static_cast<VkDeviceSize>(count) * sizeof(DrawCommand);
    VkDeviceSize requiredSize = headerSize + dataSize;
    if (requiredSize == 0)
    {
        requiredSize = headerSize;
    }

    if (!ensureWidgetCommandStorage(engine, renderer, requiredSize))
    {
        return false;
    }

    DrawCommandHeader header{};
    header.commandCount = static_cast<uint32_t>(count);

    void* mapped = nullptr;
    if (vkMapMemory(renderer.device, renderer.commandBufferMemory, 0, requiredSize, 0, &mapped) != VK_SUCCESS || !mapped)
    {
        return false;
    }

    std::memcpy(mapped, &header, sizeof(header));
    if (dataSize > 0)
    {
        std::memcpy(static_cast<uint8_t*>(mapped) + sizeof(header), commands.data(), dataSize);
    }

    vkUnmapMemory(renderer.device, renderer.commandBufferMemory);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView = target.view;

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = renderer.commandBufferStorage;
    bufferInfo.offset = 0;
    bufferInfo.range = requiredSize;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = renderer.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = renderer.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(renderer.device, 2, writes, 0, nullptr);

    vkResetCommandBuffer(renderer.commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(renderer.commandBuffer, &beginInfo);

    VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    switch (target.layout)
    {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        toGeneral.srcAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        toGeneral.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        toGeneral.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;
    default:
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
    }

    toGeneral.oldLayout = target.layout;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = target.image;
    toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneral.subresourceRange.baseMipLevel = 0;
    toGeneral.subresourceRange.levelCount = 1;
    toGeneral.subresourceRange.baseArrayLayer = 0;
    toGeneral.subresourceRange.layerCount = 1;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(renderer.commandBuffer,
                         srcStage,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &toGeneral);

    vkCmdBindPipeline(renderer.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, renderer.pipeline);
    vkCmdBindDescriptorSets(renderer.commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            renderer.pipelineLayout,
                            0,
                            1,
                            &renderer.descriptorSet,
                            0,
                            nullptr);

    WidgetPushConstants push{};
    push.outputSize = glm::vec2(static_cast<float>(width), static_cast<float>(height));
    push.clearFirst = clearFirst ? 1u : 0u;
    vkCmdPushConstants(renderer.commandBuffer,
                       renderer.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(WidgetPushConstants),
                       &push);

    const uint32_t groupX = (width + 15) / 16;
    const uint32_t groupY = (height + 15) / 16;
    vkCmdDispatch(renderer.commandBuffer, groupX, groupY, 1);

    VkImageMemoryBarrier toRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.image = target.image;
    toRead.subresourceRange = toGeneral.subresourceRange;
    toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(renderer.commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &toRead);

    vkEndCommandBuffer(renderer.commandBuffer);

    vkResetFences(renderer.device, 1, &renderer.fence);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &renderer.commandBuffer;

    vkQueueSubmit(renderer.queue, 1, &submitInfo, renderer.fence);
    vkWaitForFences(renderer.device, 1, &renderer.fence, VK_TRUE, UINT64_MAX);

    target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}
} // namespace widgets
