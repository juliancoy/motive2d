#include "display2d.h"
#include "engine2d.h"
#include "scrubber.h"
#include "color_grading_pass.h"
#include "utils.h"
#include "debug_logging.h"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <iostream>
#include <glm/glm.hpp>

namespace
{
struct LayoutInfo
{
    VkAccessFlags accessMask;
    VkPipelineStageFlags stage;
};

LayoutInfo layoutInfoFor(VkImageLayout layout)
{
    switch (layout)
    {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        return {0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    case VK_IMAGE_LAYOUT_GENERAL:
        return {VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return {VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return {VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return {0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT};
    default:
        return {0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    }
}

VkExtent2D chooseSwapExtentImpl(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)};

    actualExtent.width = std::max(capabilities.minImageExtent.width,
                                  std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(capabilities.minImageExtent.height,
                                   std::min(capabilities.maxImageExtent.height, actualExtent.height));
    return actualExtent;
}
} // namespace

Display2D::Display2D(Engine2D* engine, int width, int height, const char* title)
    : engine(engine), width(width), height(height), colorGrading(this)
{
    if (!engine)
    {
        throw std::runtime_error("Display2D requires a valid engine");
    }
    graphicsQueue = engine->graphicsQueue;
    createWindow(title);
    createSurface();
    createSwapchain();
    createCommandResources();
}

Display2D::~Display2D()
{
    if (shutdownPerformed)
    {
        return;
    }

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    // Wait for any in-flight frame work to complete before tearing down swapchain-dependent resources.
    if (!inFlightFences.empty() && engine && engine->logicalDevice != VK_NULL_HANDLE)
    {
        vkWaitForFences(engine->logicalDevice,
                        static_cast<uint32_t>(inFlightFences.size()),
                        inFlightFences.data(),
                        VK_TRUE,
                        UINT64_MAX);
    }
    if (graphicsQueue != VK_NULL_HANDLE)
    {
        vkQueueWaitIdle(graphicsQueue);
    }
    if (engine && engine->logicalDevice != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(engine->logicalDevice);
    }
    auto tWait = clock::now();

    cleanupSwapchain();
    auto tSwapchain = clock::now();

    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(engine->logicalDevice, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    scrubPipeline = VK_NULL_HANDLE;
    colorGrading.destroyPipeline();
    if (descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(engine->logicalDevice, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(engine->logicalDevice, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    auto tGpuObjects = clock::now();

    if (surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(engine->instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }
    if (window)
    {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    auto tEnd = clock::now();

    auto waitMs = std::chrono::duration_cast<std::chrono::milliseconds>(tWait - t0).count();
    auto swapchainMs = std::chrono::duration_cast<std::chrono::milliseconds>(tSwapchain - tWait).count();
    auto gpuObjectsMs = std::chrono::duration_cast<std::chrono::milliseconds>(tGpuObjects - tSwapchain).count();
    auto windowMs = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tGpuObjects).count();
    if (renderDebugEnabled())
    {
        std::cout << "[Display2D] teardown timing: waitAll=" << waitMs
                  << " ms, swapchain=" << swapchainMs
                  << " ms, gpuObjects=" << gpuObjectsMs
                  << " ms, window=" << windowMs << " ms" << std::endl;
    }

    shutdownPerformed = true;
}

void Display2D::createWindow(const char* title)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window)
    {
        throw std::runtime_error("Failed to create GLFW window for Display2D");
    }
}

void Display2D::createSurface()
{
    if (glfwCreateWindowSurface(engine->instance, window, nullptr, &surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface for Display2D");
    }
}

void Display2D::createSwapchain()
{
    VkSurfaceCapabilitiesKHR capabilities{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(engine->physicalDevice, surface, &capabilities) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to query surface capabilities for Display2D");
    }

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(engine->physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(engine->physicalDevice, surface, &formatCount, formats.data());

    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (const auto& fmt : formats)
    {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            surfaceFormat = fmt;
            break;
        }
    }

    swapchainFormat = surfaceFormat.format;
    swapchainExtent = chooseSwapExtentImpl(window, capabilities);

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    {
        imageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = swapchainFormat;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = swapchainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // Single queue family path
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(engine->logicalDevice, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create swapchain for Display2D");
    }

    vkGetSwapchainImagesKHR(engine->logicalDevice, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(engine->logicalDevice, swapchain, &imageCount, swapchainImages.data());

    swapchainImageViews.resize(imageCount);
    for (size_t i = 0; i < swapchainImages.size(); ++i)
    {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = swapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(engine->logicalDevice, &viewInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create swapchain image view for Display2D");
    }
}

    swapchainImageLayouts.assign(swapchainImages.size(), VK_IMAGE_LAYOUT_UNDEFINED);
    colorGrading.createGradingImages();

    imageAvailableSemaphores.resize(kMaxFramesInFlight);
    renderFinishedSemaphores.resize(kMaxFramesInFlight);
    inFlightFences.resize(kMaxFramesInFlight);

    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
    if (vkCreateSemaphore(engine->logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(engine->logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(engine->logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create sync objects for Display2D");
        }
    }
}

void Display2D::cleanupSwapchain()
{
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    for (auto view : swapchainImageViews)
    {
        vkDestroyImageView(engine->logicalDevice, view, nullptr);
    }
    auto tViews = clock::now();
    swapchainImageViews.clear();
    swapchainImages.clear();
    swapchainImageLayouts.clear();
    colorGrading.destroyGradingImages();

    for (auto fence : inFlightFences)
    {
        vkDestroyFence(engine->logicalDevice, fence, nullptr);
    }
    auto tFences = clock::now();
    for (auto sem : imageAvailableSemaphores)
    {
        vkDestroySemaphore(engine->logicalDevice, sem, nullptr);
    }
    auto tImageSems = clock::now();
    for (auto sem : renderFinishedSemaphores)
    {
        vkDestroySemaphore(engine->logicalDevice, sem, nullptr);
    }
    auto tRenderSems = clock::now();

    imageAvailableSemaphores.clear();
    renderFinishedSemaphores.clear();
    inFlightFences.clear();

    if (swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(engine->logicalDevice, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
    auto tSwapchain = clock::now();

    auto viewsMs = std::chrono::duration_cast<std::chrono::milliseconds>(tViews - t0).count();
    auto fencesMs = std::chrono::duration_cast<std::chrono::milliseconds>(tFences - tViews).count();
    auto semAvailMs = std::chrono::duration_cast<std::chrono::milliseconds>(tImageSems - tFences).count();
    auto semRenderMs = std::chrono::duration_cast<std::chrono::milliseconds>(tRenderSems - tImageSems).count();
    auto swapchainMs = std::chrono::duration_cast<std::chrono::milliseconds>(tSwapchain - tRenderSems).count();
    if (renderDebugEnabled())
    {
        std::cout << "[Display2D] cleanupSwapchain timing: views=" << viewsMs
                  << " ms, fences=" << fencesMs
                  << " ms, sem(ap)=" << semAvailMs
                  << " ms, sem(render)=" << semRenderMs
                  << " ms, swapchain=" << swapchainMs << " ms" << std::endl;
    }
}


void Display2D::createCommandResources()
{
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = engine->graphicsQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(engine->logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create command pool for Display2D");
    }

    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(engine->logicalDevice, &allocInfo, &commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate command buffer for Display2D");
    }
}


void Display2D::recreateSwapchain()
{
    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    while (fbWidth == 0 || fbHeight == 0)
    {
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(engine->logicalDevice);

    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(engine->logicalDevice, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(engine->logicalDevice, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    cleanupSwapchain();
    createSwapchain();
    createCommandResources();
    createComputeResources();
}

bool Display2D::shouldClose() const
{
    return glfwWindowShouldClose(window);
}

void Display2D::pollEvents() const
{
    glfwPollEvents();
}

