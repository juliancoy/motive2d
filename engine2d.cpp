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

bool Engine2D::initialize()
{
    if (initialized)
    {
        return true;
    }

    if (!glfwInit())
    {
        std::cerr << "[Engine2D] Failed to initialize GLFW.\n";
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

    try
    {
        renderDevice.initialize();
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

void Engine2D::play()
{
    playing = true;
}

void Engine2D::pause()
{
    playing = false;
}

void Engine2D::seek(float timeSeconds)
{
    if (!videoLoaded)
    {
        return;
    }

    timeSeconds = std::clamp(timeSeconds, 0.0f, duration);
    currentTime = timeSeconds;

    video::stopAsyncDecoding(playbackState.decoder);
    if (!video::seekVideoDecoder(playbackState.decoder, timeSeconds))
    {
        std::cerr << "[Engine2D] Failed to seek video decoder.\n";
    }

    vkDeviceWaitIdle(logicalDevice);

    playbackState.pendingFrames.clear();
    playbackState.playbackClockInitialized = false;

    video::startAsyncDecoding(playbackState.decoder, 12);
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

    if (!overlay::initializeOverlayCompute(this, overlayCompute))
    {
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

        overlay::updateFpsOverlay(this,
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
        overlay::destroyOverlayCompute(overlayCompute);
        overlayInitialized = false;
    }

    if (videoLoaded)
    {
        video::stopAsyncDecoding(playbackState.decoder);

        overlay::destroyImageResource(this, playbackState.video.lumaImage);
        overlay::destroyImageResource(this, playbackState.video.chromaImage);
        overlay::destroyImageResource(this, playbackState.overlay.image);
        overlay::destroyImageResource(this, playbackState.fpsOverlay.image);

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

        video::cleanupVideoDecoder(playbackState.decoder);
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

overlay::OverlayCompute& Engine2D::getOverlayCompute()
{
    return overlayCompute;
}

void Engine2D::refreshFpsOverlay()
{
    updateFpsOverlay();
}
