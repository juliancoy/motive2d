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
}

Engine2D::~Engine2D()
{
    shutdown();
}

bool Engine2D::initialize(bool requireWindow)
{
    if (initialized)
    {
        return true;
    }

    if (requireWindow)
    {
        if (!glfwInit())
        {
            const char* errMsg = nullptr;
            int errCode = glfwGetError(&errMsg);
            std::cerr << "[Engine2D] Failed to initialize GLFW ("
                      << errCode << "): " << (errMsg ? errMsg : "unknown") << "\n";
            return false;
        }
        glfwInitialized = true;

        if (!glfwVulkanSupported())
        {
            std::cerr << "[Engine2D] GLFW reports Vulkan support is unavailable.\n";
            glfwTerminate();
            glfwInitialized = false;
            return false;
        }
    }

    try
    {
        renderDevice.initialize(requireWindow);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Engine2D] Failed to initialize Vulkan: " << ex.what() << "\n";
        if (glfwInitialized)
        {
            glfwTerminate();
            glfwInitialized = false;
        }
        return false;
    }

    fpsLastSample = std::chrono::steady_clock::now();
    initialized = true;
    std::cout << "[Engine2D] Engine2D initialized successfully.\n";
    return true;
}

Display2D* Engine2D::createWindow(int width, int height, const char* title)
{
    if (!initialized)
    {
        std::cerr << "[Engine2D] Engine2D has not been initialized.\n";
        return nullptr;
    }

    try
    {
        auto window = std::make_unique<Display2D>(this, width, height, title);
        Display2D* ptr = window.get();
        windows.push_back(std::move(window));
        std::cout << "[Engine2D] Created window: " << title
                  << " (" << width << "x" << height << ")\n";
        return ptr;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Engine2D] Failed to create window: " << ex.what() << "\n";
        return nullptr;
    }
}

bool Engine2D::loadVideo(const std::filesystem::path& filePath, std::optional<bool> swapUV)
{
    if (!initialized)
    {
        std::cerr << "[Engine2D] Engine2D not initialized.\n";
        return false;
    }

    if (!std::filesystem::exists(filePath))
    {
        std::cerr << "[Engine2D] Video file not found: " << filePath << "\n";
        return false;
    }

    double durationSeconds = 0.0;
    if (!initializeVideoPlayback(filePath, this, playbackState, durationSeconds, swapUV))
    {
        std::cerr << "[Engine2D] Failed to initialize video playback.\n";
        return false;
    }

    videoLoaded = true;
    duration = static_cast<float>(durationSeconds);
    currentTime = 0.0f;
    playing = true;

    if (!initializeOverlay())
    {
        std::cerr << "[Engine2D] Warning: Failed to initialize overlay compute.\n";
    }

    std::cout << "[Engine2D] Loaded video: " << filePath << "\n";
    std::cout << "  Resolution: " << playbackState.decoder.width << "x" << playbackState.decoder.height << "\n";
    std::cout << "  Framerate: " << playbackState.decoder.fps << " fps\n";
    std::cout << "  Duration: " << duration << " seconds\n";

    return true;
}

void Engine2D::setDecodeDebugEnabled(bool enabled)
{
    decodeDebugEnabled = enabled;
}

bool Engine2D::isDecodeDebugEnabled() const
{
    return decodeDebugEnabled;
}

Engine2D::VideoInfo Engine2D::getVideoInfo() const
{
    VideoInfo info{};
    if (videoLoaded)
    {
        info.width = static_cast<uint32_t>(playbackState.decoder.width);
        info.height = static_cast<uint32_t>(playbackState.decoder.height);
        info.framerate = static_cast<float>(playbackState.decoder.fps);
        info.duration = duration;
    }
    return info;
}

void Engine2D::setCurrentTime(float timeSeconds)
{
    currentTime = timeSeconds;
}

void Engine2D::setGrading(const GradingSettings& settings)
{
    gradingSettings = settings;
}

void Engine2D::setCrop(const CropRegion& region)
{
    cropRegion = region;
}

void Engine2D::clearGrading()
{
    gradingSettings.reset();
}

void Engine2D::clearCrop()
{
    cropRegion.reset();
}

bool Engine2D::initializeOverlay()
{
    if (overlayInitialized)
    {
        return true;
    }

    if (!initializeRectOverlayCompute(this, rectOverlayCompute))
    {
        return false;
    }

    if (!initializePoseOverlayCompute(this, poseOverlayCompute))
    {
        destroyRectOverlayCompute(rectOverlayCompute);
        return false;
    }

    overlayInitialized = true;
    return true;
}

void Engine2D::updateFpsOverlay()
{
    if (!videoLoaded)
    {
        return;
    }

    fpsFrameCounter++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsLastSample).count();

    if (elapsed >= 500)
    {
        currentFps = static_cast<float>(fpsFrameCounter) * 1000.0f / static_cast<float>(elapsed);
        fpsFrameCounter = 0;
        fpsLastSample = now;

        updateFpsOverlay(this,
                                  playbackState.fpsOverlay,
                                  playbackState.overlay.sampler,
                                  playbackState.video.sampler,
                                  currentFps,
                                  playbackState.fpsOverlay.lastRefWidth,
                                  playbackState.fpsOverlay.lastRefHeight);
    }
}

void Engine2D::applyGrading(ColorAdjustments& adjustments) const
{
    if (!gradingSettings)
    {
        return;
    }

    adjustments.exposure = gradingSettings->exposure;
    adjustments.contrast = gradingSettings->contrast;
    adjustments.saturation = gradingSettings->saturation;
    adjustments.shadows = gradingSettings->shadows;
    adjustments.midtones = gradingSettings->midtones;
    adjustments.highlights = gradingSettings->highlights;

    if (gradingSettings->curveEnabled)
    {
        adjustments.curveLut = gradingSettings->curveLut;
        adjustments.curveEnabled = true;
    }
}

RenderOverrides Engine2D::buildRenderOverrides() const
{
    RenderOverrides overrides{};

    if (cropRegion)
    {
        overrides.useCrop = true;
        overrides.cropOrigin = glm::vec2(cropRegion->x, cropRegion->y);
        overrides.cropSize = glm::vec2(cropRegion->width, cropRegion->height);
    }

    return overrides;
}

bool Engine2D::renderFrame()
{
    if (!initialized || !videoLoaded || windows.empty())
    {
        return false;
    }

    for (auto& window : windows)
    {
        if (window)
        {
            window->pollEvents();
            if (window->shouldClose())
            {
                return false;
            }
        }
    }

    double playbackSeconds = advancePlayback(playbackState, playing);
    currentTime = static_cast<float>(playbackSeconds);

    updateFpsOverlay();

    ColorAdjustments adjustments{};
    applyGrading(adjustments);
    const ColorAdjustments* activeAdjustments = gradingSettings ? &adjustments : nullptr;

    RenderOverrides overrides = buildRenderOverrides();
    float scrubProgress = duration > 0.0f ? currentTime / duration : 0.0f;
    float scrubPlaying = playing ? 1.0f : 0.0f;

    for (auto& window : windows)
    {
        if (window)
        {
            window->renderFrame(playbackState.video.descriptors,
                                playbackState.overlay.info,
                                playbackState.fpsOverlay.info,
                                playbackState.colorInfo,
                                scrubProgress,
                                scrubPlaying,
                                &overrides,
                                activeAdjustments);
        }
    }

    return true;
}

void Engine2D::run()
{
    if (!initialized || !videoLoaded || windows.empty())
    {
        std::cerr << "[Engine2D] Cannot run: engine not initialized, no video loaded, or no windows.\n";
        return;
    }

    std::cout << "[Engine2D] Starting main render loop...\n";

    while (true)
    {
        if (!renderFrame())
        {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    std::cout << "[Engine2D] Render loop ended.\n";
}

void Engine2D::shutdown()
{
    std::cout << "[Engine2D] Shutting down...\n";

    if (overlayInitialized)
    {
        destroyPoseOverlayCompute(poseOverlayCompute);
        destroyRectOverlayCompute(rectOverlayCompute);
        overlayInitialized = false;
    }

    if (videoLoaded)
    {
        stopAsyncDecoding(playbackState.decoder);

        destroyImageResource(this, playbackState.video.lumaImage);
        destroyImageResource(this, playbackState.video.chromaImage);
        destroyImageResource(this, playbackState.overlay.image);
        destroyImageResource(this, playbackState.fpsOverlay.image);

        if (playbackState.video.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(logicalDevice, playbackState.video.sampler, nullptr);
            playbackState.video.sampler = VK_NULL_HANDLE;
        }
        if (playbackState.overlay.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(logicalDevice, playbackState.overlay.sampler, nullptr);
            playbackState.overlay.sampler = VK_NULL_HANDLE;
        }

        cleanupVideoDecoder(playbackState.decoder);
        videoLoaded = false;
    }

    for (auto& window : windows)
    {
        if (window)
        {
            window->shutdown();
        }
    }
    windows.clear();

    if (glfwInitialized)
    {
        glfwTerminate();
        glfwInitialized = false;
    }

    if (initialized)
    {
        renderDevice.shutdown();
        initialized = false;
    }

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

uint32_t Engine2D::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    return renderDevice.findMemoryType(typeFilter, properties);
}

VkShaderModule Engine2D::createShaderModule(const std::vector<char>& code)
{
    return renderDevice.createShaderModule(code);
}

VkQueue& Engine2D::getGraphicsQueue()
{
    return renderDevice.getGraphicsQueue();
}

uint32_t Engine2D::getGraphicsQueueFamilyIndex()
{
    return renderDevice.getGraphicsQueueFamilyIndex();
}

VkQueue& Engine2D::getVideoQueue()
{
    return renderDevice.getVideoQueue();
}

uint32_t Engine2D::getVideoQueueFamilyIndex()
{
    return renderDevice.getVideoQueueFamilyIndex();
}

VkPhysicalDeviceProperties& Engine2D::getDeviceProperties()
{
    return renderDevice.getDeviceProperties();
}

VideoPlaybackState& Engine2D::getPlaybackState()
{
    return playbackState;
}

RectOverlayCompute& Engine2D::getRectOverlayCompute()
{
    return rectOverlayCompute;
}

PoseOverlayCompute& Engine2D::getPoseOverlayCompute()
{
    return poseOverlayCompute;
}

void Engine2D::refreshFpsOverlay()
{
    updateFpsOverlay();
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
} // namespace

void destroyImageResource(Engine2D* engine, ImageResource& res)
{
    if (!engine)
    {
        return;
    }
    if (res.view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(engine->logicalDevice, res.view, nullptr);
        res.view = VK_NULL_HANDLE;
    }
    if (res.image != VK_NULL_HANDLE)
    {
        vkDestroyImage(engine->logicalDevice, res.image, nullptr);
        res.image = VK_NULL_HANDLE;
    }
    if (res.memory != VK_NULL_HANDLE)
    {
        vkFreeMemory(engine->logicalDevice, res.memory, nullptr);
        res.memory = VK_NULL_HANDLE;
    }
    res.format = VK_FORMAT_UNDEFINED;
    res.width = 0;
    res.height = 0;
    res.layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

bool ensureImageResource(Engine2D* engine,
                         ImageResource& res,
                         uint32_t width,
                         uint32_t height,
                         VkFormat format,
                         bool& recreated,
                         VkImageUsageFlags usage)
{
    recreated = false;
    if (res.image != VK_NULL_HANDLE && res.width == width && res.height == height && res.format == format)
    {
        return true;
    }

    destroyImageResource(engine, res);
    recreated = true;

    if (width == 0 || height == 0 || format == VK_FORMAT_UNDEFINED)
    {
        return false;
    }

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(engine->logicalDevice, &imageInfo, nullptr, &res.image) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create image." << std::endl;
        return false;
    }

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(engine->logicalDevice, res.image, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = engine->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(engine->logicalDevice, &allocInfo, nullptr, &res.memory) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to allocate image memory." << std::endl;
        vkDestroyImage(engine->logicalDevice, res.image, nullptr);
        res.image = VK_NULL_HANDLE;
        return false;
    }

    vkBindImageMemory(engine->logicalDevice, res.image, res.memory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = res.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(engine->logicalDevice, &viewInfo, nullptr, &res.view) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create image view." << std::endl;
        destroyImageResource(engine, res);
        return false;
    }

    res.format = format;
    res.width = width;
    res.height = height;
    res.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

bool uploadImageData(Engine2D* engine,
                     ImageResource& res,
                     const void* data,
                     size_t dataSize,
                     uint32_t width,
                     uint32_t height,
                     VkFormat format,
                     VkImageUsageFlags usage)
{
    if (!data || dataSize == 0 || width == 0 || height == 0)
    {
        return false;
    }

    bool recreated = false;
    if (!ensureImageResource(engine, res, width, height, format, recreated, usage))
    {
        return false;
    }

    // Use a staging buffer for optimal performance
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;
    engine->createBuffer(imageSize,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stagingBuffer,
                         stagingBufferMemory);

    void* mapped = nullptr;
    vkMapMemory(engine->logicalDevice, stagingBufferMemory, 0, imageSize, 0, &mapped);
    if (mapped)
    {
        std::memcpy(mapped, data, dataSize);
        vkUnmapMemory(engine->logicalDevice, stagingBufferMemory);
    }
    else
    {
        vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);
        std::cerr << "[Video2D] Failed to map staging buffer memory." << std::endl;
        return false;
    }

    copyBufferToImage(engine, stagingBuffer, res.image, VK_IMAGE_LAYOUT_UNDEFINED, width, height);

    vkDestroyBuffer(engine->logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(engine->logicalDevice, stagingBufferMemory, nullptr);

    res.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}
