#include "engine2d.h"

#include <algorithm>
#include <iostream>
#include <thread>
#include <stdexcept>

Engine2D::Engine2D()
    : renderDevice()
    , instance(renderDevice.getInstance())
    , logicalDevice(renderDevice.getLogicalDevice())
    , physicalDevice(renderDevice.getPhysicalDevice())
    , graphicsQueue(renderDevice.getGraphicsQueue())
    , graphicsQueueFamilyIndex(renderDevice.getGraphicsQueueFamilyIndex())
    , videoQueue(renderDevice.getVideoQueue())
    , videoQueueFamilyIndex(renderDevice.getVideoQueueFamilyIndex())
{
    fpsLastSample = std::chrono::steady_clock::now();
    createComputeResources();

    if (!glfwInit())
    {
        const char* errMsg = nullptr;
        int errCode = glfwGetError(&errMsg);
        std::cerr << "[Engine2D] Failed to initialize GLFW ("
                    << errCode << "): " << (errMsg ? errMsg : "unknown") << "\n";
    }
    glfwInitialized = true;

    if (!glfwVulkanSupported())
    {
        std::cerr << "[Engine2D] GLFW reports Vulkan support is unavailable.\n";
        glfwTerminate();
        glfwInitialized = false;
    }

    try
    {
        renderDevice.initialize(true);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Engine2D] Failed to initialize Vulkan: " << ex.what() << "\n";
        if (glfwInitialized)
        {
            glfwTerminate();
            glfwInitialized = false;
        }
    }

    fpsLastSample = std::chrono::steady_clock::now();
    initialized = true;
    std::cout << "[Engine2D] Engine2D initialized successfully.\n";
}


Engine2D::~Engine2D()
{
    std::cout << "[Engine2D] Shutting down...\n";

    std::cout << "[Engine2D] Shutdown complete.\n";
}

VkCommandBuffer Engine2D::beginSingleTimeCommands()
{
    return renderDevice.beginSingleTimeCommands();
}

void Engine2D::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    renderDevice.endSingleTimeCommands(commandBuffer);
}

void Engine2D::createBuffer(VkDeviceSize size,
                           VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags properties,
                           VkBuffer& buffer,
                           VkDeviceMemory& bufferMemory)
{
    renderDevice.createBuffer(size, usage, properties, buffer, bufferMemory);
}

void Engine2D::updateFpsOverlay()
{
    // TODO: Implement FPS overlay update
}

void copyBufferToImage(Engine2D* engine,
                       VkBuffer stagingBuffer,
                       VkImage targetImage,
                       VkImageLayout currentLayout,
                       uint32_t width,
                       uint32_t height)
{
    VkCommandBuffer cmd = engine->beginSingleTimeCommands();

    VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toTransfer.oldLayout = currentLayout;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = targetImage;
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.baseMipLevel = 0;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.baseArrayLayer = 0;
    toTransfer.subresourceRange.layerCount = 1;
    toTransfer.srcAccessMask = (currentLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : VK_ACCESS_SHADER_READ_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkPipelineStageFlags srcStage = (currentLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                                        ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(cmd,
                         srcStage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toTransfer);

    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(cmd, stagingBuffer, targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier toShader{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.image = targetImage;
    toShader.subresourceRange = toTransfer.subresourceRange;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toShader);

    engine->endSingleTimeCommands(cmd);
}

bool Engine2D::initialize(bool requireWindow) {
    // Already initialized in constructor
    return initialized;
}

void Engine2D::createComputeResources() {
    // Stub implementation
    // TODO: Implement compute resource creation
}

uint32_t Engine2D::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    return renderDevice.findMemoryType(typeFilter, properties);
}

VkShaderModule Engine2D::createShaderModule(const std::vector<char>& code) {
    return renderDevice.createShaderModule(code);
}

VkQueue& Engine2D::getGraphicsQueue() {
    return graphicsQueue;
}

uint32_t Engine2D::getGraphicsQueueFamilyIndex() {
    return graphicsQueueFamilyIndex;
}

VkQueue& Engine2D::getVideoQueue() {
    return videoQueue;
}

uint32_t Engine2D::getVideoQueueFamilyIndex() {
    return videoQueueFamilyIndex;
}

VkPhysicalDeviceProperties& Engine2D::getDeviceProperties() {
    return renderDevice.getDeviceProperties();
}


// Helper: Create a Vulkan image view
VkImageView Engine2D::createImageView(VkImage image, VkFormat format)
{
    if (image == VK_NULL_HANDLE)
    {
        throw std::runtime_error("[Video] createImageView: invalid VkImage handle");
    }
    if (format == VK_FORMAT_UNDEFINED)
    {
        throw std::runtime_error("[Video] createImageView: undefined VkFormat for image");
    }
    
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &view) != VK_SUCCESS)
    {
        throw std::runtime_error("[Video] createImageView: vkCreateImageView failed");
    }
    return view;
}
