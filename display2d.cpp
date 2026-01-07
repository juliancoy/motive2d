// display2d.cpp
#include "display2d.h"
#include "engine2d.h"

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace {

// Track per-swapchain image layouts without adding members to Display2D.
// Keying by VkSwapchainKHR is fine because we erase on destroy/recreate.
static std::unordered_map<VkSwapchainKHR, std::vector<VkImageLayout>> g_swapchainLayouts;

static VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats)
{
    // Prefer SRGB if available; otherwise first.
    for (const auto& f : formats)
    {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return f;
        }
    }
    return formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
                           : formats[0];
}

static VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes)
{
    // FIFO is guaranteed; MAILBOX is nice if available.
    for (auto m : modes)
    {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& caps)
{
    if (caps.currentExtent.width != UINT32_MAX)
    {
        return caps.currentExtent;
    }

    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);

    VkExtent2D e{};
    e.width  = static_cast<uint32_t>(std::max(0, fbW));
    e.height = static_cast<uint32_t>(std::max(0, fbH));

    e.width  = std::max(caps.minImageExtent.width,  std::min(caps.maxImageExtent.width,  e.width));
    e.height = std::max(caps.minImageExtent.height, std::min(caps.maxImageExtent.height, e.height));
    return e;
}

static void framebufferResizeCallback(GLFWwindow* wnd, int /*w*/, int /*h*/)
{
    auto* self = reinterpret_cast<Display2D*>(glfwGetWindowUserPointer(wnd));
    if (self)
    {
        // We can’t access private members here; this is a friendless callback.
        // So we just set a window user pointer flag by storing it in GLFW itself:
        // We’ll read framebuffer size each frame and handle OUT_OF_DATE/SUBOPTIMAL.
        // (Display2D uses framebufferResized_ internally; we flip it via a small hack below.)
    }
}

static void setFramebufferResized(Display2D* d, bool v)
{
    // Minimal “friendless” way: store a boolean in GLFW’s user pointer alongside Display2D*
    // is messy. Instead we rely primarily on OUT_OF_DATE/SUBOPTIMAL and the zero-size loop.
    // But we *do* still want the hint; easiest is to use a GLFW window attribute slot:
    // Not available. So we’ll just recreate on OUT_OF_DATE/SUBOPTIMAL.
    (void)d; (void)v;
}

} // namespace

Display2D::Display2D(Engine2D* engine, int width, int height, const char* title)
    : engine(engine), width_(width), height_(height)
{
    if (!engine)
        throw std::runtime_error("Display2D requires a valid engine");

    graphicsQueue_ = engine->graphicsQueue;

    createWindow_(title);
    createSurface_();
    createSwapchain_();
    createFrameResources_();

    // Optional: you can still register a callback (OUT_OF_DATE handling is the real trigger).
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
}

Display2D::~Display2D()
{
    shutdown();
}

void Display2D::shutdown()
{
    if (shutdownPerformed_)
        return;

    if (engine && engine->logicalDevice != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(engine->logicalDevice);
    }

    destroyFrameResources_();
    destroySwapchain_();

    if (surface_ != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(engine->instance, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }

    if (window_)
    {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    shutdownPerformed_ = true;
}

bool Display2D::shouldClose() const
{
    return window_ ? glfwWindowShouldClose(window_) : true;
}

void Display2D::pollEvents() const
{
    glfwPollEvents();
}

SwapchainInfo Display2D::swapchainInfo() const
{
    SwapchainInfo sc{};
    sc.format = swapchainFormat_;
    sc.extent = swapchainExtent_;
    sc.generation = swapchainGeneration_;
    sc.images = &swapchainImages_;
    sc.views  = &swapchainImageViews_;
    return sc;
}

void Display2D::createWindow_(const char* title)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(width_, height_, title, nullptr, nullptr);
    if (!window_)
        throw std::runtime_error("Failed to create GLFW window for Display2D");

    glfwSetWindowUserPointer(window_, this);
}

void Display2D::createSurface_()
{
    if (glfwCreateWindowSurface(engine->instance, window_, nullptr, &surface_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface for Display2D");
}

void Display2D::createSwapchain_()
{
    // Handle minimized window (0x0) gracefully by waiting.
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window_, &fbW, &fbH);
    while (fbW == 0 || fbH == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window_, &fbW, &fbH);
    }

    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(engine->physicalDevice, surface_, &caps) != VK_SUCCESS)
        throw std::runtime_error("Failed to query surface capabilities");

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(engine->physicalDevice, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    if (formatCount)
        vkGetPhysicalDeviceSurfaceFormatsKHR(engine->physicalDevice, surface_, &formatCount, formats.data());

    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(engine->physicalDevice, surface_, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(pmCount);
    if (pmCount)
        vkGetPhysicalDeviceSurfacePresentModesKHR(engine->physicalDevice, surface_, &pmCount, presentModes.data());

    const VkSurfaceFormatKHR chosenFormat = chooseSurfaceFormat(formats);
    const VkPresentModeKHR chosenPresentMode = choosePresentMode(presentModes);
    const VkExtent2D extent = chooseSwapExtent(window_, caps);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosenFormat.format;
    ci.imageColorSpace = chosenFormat.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;

    // We want transfer dst for a generic presenter that copies/blits into the swapchain.
    // If you later add a compute presenter that writes directly, it can transition to GENERAL.
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = chosenPresentMode;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(engine->logicalDevice, &ci, nullptr, &swapchain_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create swapchain");

    swapchainFormat_ = chosenFormat.format;
    swapchainExtent_ = extent;

    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(engine->logicalDevice, swapchain_, &actualCount, nullptr);
    swapchainImages_.resize(actualCount);
    vkGetSwapchainImagesKHR(engine->logicalDevice, swapchain_, &actualCount, swapchainImages_.data());

    swapchainImageViews_.resize(actualCount);
    for (size_t i = 0; i < swapchainImages_.size(); ++i)
    {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = swapchainImages_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = swapchainFormat_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel = 0;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount = 1;

        if (vkCreateImageView(engine->logicalDevice, &vi, nullptr, &swapchainImageViews_[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swapchain image view");
    }

    g_swapchainLayouts[swapchain_] = std::vector<VkImageLayout>(swapchainImages_.size(), VK_IMAGE_LAYOUT_UNDEFINED);

    ++swapchainGeneration_;

    if (presenter_)
        presenter_->onSwapchainChanged(swapchainInfo());
}

void Display2D::destroySwapchain_()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    if (swapchain_ != VK_NULL_HANDLE)
    {
        g_swapchainLayouts.erase(swapchain_);
    }

    for (auto v : swapchainImageViews_)
    {
        if (v != VK_NULL_HANDLE)
            vkDestroyImageView(engine->logicalDevice, v, nullptr);
    }
    swapchainImageViews_.clear();
    swapchainImages_.clear();

    if (swapchain_ != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(engine->logicalDevice, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }

    swapchainFormat_ = VK_FORMAT_UNDEFINED;
    swapchainExtent_ = {0, 0};
}

void Display2D::createFrameResources_()
{
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = engine->graphicsQueueFamilyIndex;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(engine->logicalDevice, &pi, nullptr, &commandPool_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");

    // Allocate one cmd buffer per frame-in-flight.
    std::vector<VkCommandBuffer> bufs(kMaxFramesInFlight);

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kMaxFramesInFlight;

    if (vkAllocateCommandBuffers(engine->logicalDevice, &ai, bufs.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        frames_[i].cmd = bufs[i];
    }

    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        if (vkCreateSemaphore(engine->logicalDevice, &si, nullptr, &frames_[i].imageAvailable) != VK_SUCCESS ||
            vkCreateSemaphore(engine->logicalDevice, &si, nullptr, &frames_[i].renderFinished) != VK_SUCCESS ||
            vkCreateFence(engine->logicalDevice, &fi, nullptr, &frames_[i].inFlight) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-frame sync objects");
        }
    }
}

void Display2D::destroyFrameResources_()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        if (frames_[i].inFlight)
            vkDestroyFence(engine->logicalDevice, frames_[i].inFlight, nullptr);
        if (frames_[i].imageAvailable)
            vkDestroySemaphore(engine->logicalDevice, frames_[i].imageAvailable, nullptr);
        if (frames_[i].renderFinished)
            vkDestroySemaphore(engine->logicalDevice, frames_[i].renderFinished, nullptr);

        frames_[i] = {};
    }

    if (commandPool_ != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(engine->logicalDevice, commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
    }
}

void Display2D::recreateSwapchain_()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    // Avoid recreating while minimized.
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window_, &fbW, &fbH);
    while (fbW == 0 || fbH == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window_, &fbW, &fbH);
    }

    vkDeviceWaitIdle(engine->logicalDevice);

    destroySwapchain_();
    createSwapchain_();
}

void Display2D::beginCmd_(VkCommandBuffer cmd)
{
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
}

void Display2D::endCmd_(VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
}

void Display2D::renderFrame(VkSemaphore externalWaitSemaphore,
                            VkPipelineStageFlags externalWaitStage)
{
    if (!engine || swapchain_ == VK_NULL_HANDLE)
        return;

    Frame& fr = frames_[currentFrame_];

    vkWaitForFences(engine->logicalDevice, 1, &fr.inFlight, VK_TRUE, UINT64_MAX);
    vkResetFences(engine->logicalDevice, 1, &fr.inFlight);

    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(engine->logicalDevice,
                                         swapchain_,
                                         UINT64_MAX,
                                         fr.imageAvailable,
                                         VK_NULL_HANDLE,
                                         &imageIndex);

    if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR)
    {
        recreateSwapchain_();
        return;
    }
    if (acq != VK_SUCCESS)
        throw std::runtime_error("vkAcquireNextImageKHR failed");

    vkResetCommandBuffer(fr.cmd, 0);
    beginCmd_(fr.cmd);

    // --- swapchain layout -> TRANSFER_DST_OPTIMAL (same as your code) ---
    auto it = g_swapchainLayouts.find(swapchain_);
    if (it == g_swapchainLayouts.end() || imageIndex >= it->second.size())
    {
        endCmd_(fr.cmd);
        recreateSwapchain_();
        return;
    }

    VkImage swapImg = swapchainImages_[imageIndex];
    VkImageLayout oldLayout = it->second[imageIndex];

    VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toDst.oldLayout = oldLayout;
    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image = swapImg;
    toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toDst.subresourceRange.baseMipLevel = 0;
    toDst.subresourceRange.levelCount = 1;
    toDst.subresourceRange.baseArrayLayer = 0;
    toDst.subresourceRange.layerCount = 1;
    toDst.srcAccessMask = 0;
    toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(fr.cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toDst);

    it->second[imageIndex] = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // presenter work
    if (presenter_ && presentInput_.image != VK_NULL_HANDLE)
    {
        presenter_->record(fr.cmd,
                           swapImg,
                           swapchainImageViews_[imageIndex],
                           swapchainExtent_,
                           presentInput_,
                           overrides_);
    }
    else
    {
        VkClearColorValue black{};
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.levelCount = 1;
        range.layerCount = 1;
        vkCmdClearColorImage(fr.cmd, swapImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }

    // --- TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR (same as your code) ---
    VkImageMemoryBarrier toPresent{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.image = swapImg;
    toPresent.subresourceRange = toDst.subresourceRange;
    toPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toPresent.dstAccessMask = 0;

    vkCmdPipelineBarrier(fr.cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toPresent);

    it->second[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    endCmd_(fr.cmd);

    // ---- submit with 1 or 2 wait semaphores ----
    VkSemaphore waitSems[2] = { fr.imageAvailable, externalWaitSemaphore };
    VkPipelineStageFlags waitStages[2] = { VK_PIPELINE_STAGE_TRANSFER_BIT, externalWaitStage };

    uint32_t waitCount = 1;
    if (externalWaitSemaphore != VK_NULL_HANDLE)
        waitCount = 2;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = waitCount;
    si.pWaitSemaphores = waitSems;
    si.pWaitDstStageMask = waitStages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &fr.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &fr.renderFinished;

    if (vkQueueSubmit(graphicsQueue_, 1, &si, fr.inFlight) != VK_SUCCESS)
        throw std::runtime_error("vkQueueSubmit failed");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &fr.renderFinished;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &imageIndex;

    VkResult pr = vkQueuePresentKHR(graphicsQueue_, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR)
        recreateSwapchain_();
    else if (pr != VK_SUCCESS)
        throw std::runtime_error("vkQueuePresentKHR failed");

    currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
}
