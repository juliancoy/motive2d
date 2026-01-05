#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <optional>
#include <chrono>
#include <array>
#include <vector>
#include <GLFW/glfw3.h>
#include "crop.h"
#include "graphicsdevice.h"
#include "image_resource.h"

struct RenderOptions {
    bool showScrubber = true;
    bool showFps = true;
    bool showOverlay = true;
    std::optional<GradingSettings> grading;
    std::optional<CropRegion> crop;
    float playbackSpeed = 1.0f;
};


class Engine2D {
public:
    Engine2D();
    ~Engine2D();

    RenderDevice renderDevice;
    VkInstance &instance;
    VkDevice &logicalDevice;
    VkPhysicalDevice &physicalDevice;
    VkQueue &graphicsQueue;
    uint32_t &graphicsQueueFamilyIndex;
    VkQueue &videoQueue;
    uint32_t &videoQueueFamilyIndex;
    ColorGrading colorGrading;
    void createComputeResources();



    // Disable copy
    Engine2D(const Engine2D&) = delete;
    Engine2D& operator=(const Engine2D&) = delete;

    // Initialize the engine (Vulkan, GLFW, etc.)
    // `requireWindow` controls whether GLFW initialization is performed.
    bool initialize(bool requireWindow = true);

    // Create a 2D display window
    Display2D* createWindow(int width = 1280, int height = 720, 
                            const char* title = "Motive 2D");

    // Load a video file
    bool loadVideo(const std::filesystem::path& filePath,
                   std::optional<bool> swapUV = std::nullopt);

    // Get video information
    struct VideoInfo {
        uint32_t width = 0;
        uint32_t height = 0;
        float framerate = 0.0f;
        float duration = 0.0f;
        // Codec info can be derived from decoder
    };
    VideoInfo getVideoInfo() const;

    // Playback control
    void play();
    void pause();
    bool isPlaying() const { return playing; }
    void seek(float timeSeconds);
    float getCurrentTime() const { return currentTime; }
    float getDuration() const { return duration; }
    void setCurrentTime(float timeSeconds);

    // Grading and crop
    void setGrading(const GradingSettings& settings);
    void setCrop(const CropRegion& region);
    void clearGrading();
    void clearCrop();

    // Render a frame to all windows
    bool renderFrame();

    void refreshFpsOverlay();

    RectOverlay& getRectOverlay();
    PoseOverlay& getPoseOverlay();

    void setDecodeDebugEnabled(bool enabled);
    bool isDecodeDebugEnabled() const;

    // Main loop (blocks until all windows closed)
    void run();

    // Cleanup
    void shutdown();

    // Vulkan helpers
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    VkQueue& getGraphicsQueue();
    uint32_t getGraphicsQueueFamilyIndex();
    VkQueue& getVideoQueue();
    uint32_t getVideoQueueFamilyIndex();
    VkPhysicalDeviceProperties& getDeviceProperties();

bool ensureImageResource(Engine2D* engine,
                         ImageResource& res,
                         uint32_t width,
                         uint32_t height,
                         VkFormat format,
                         bool& recreated,
                         VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

void destroyImageResource(Engine2D* engine, ImageResource& res);

bool uploadImageData(Engine2D* engine,
                     ImageResource& res,
                     const void* data,
                     size_t dataSize,
                     uint32_t width,
                     uint32_t height,
                     VkFormat format,
                     VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    void createComputeResources();
 private:

    bool initialized = false;
    bool glfwInitialized = false;
    
    // Video state
    bool videoLoaded = false;
    float duration = 0.0f;
    float currentTime = 0.0f;
    bool playing = true;
    bool decodeDebugEnabled = false;
    
    // Grading and crop state
    std::optional<GradingSettings> gradingSettings;
    std::optional<CropRegion> cropRegion;
    
    // Overlay resources
    RectOverlay rectOverlay;
    PoseOverlay poseOverlay;
    bool overlayInitialized = false;
    
    // FPS tracking
    std::chrono::steady_clock::time_point fpsLastSample;
    int fpsFrameCounter = 0;
    float currentFps = 0.0f;
    
    // Internal methods
    void updateFpsOverlay();
    void applyGrading(ColorAdjustments& adjustments) const;
    RenderOverrides buildRenderOverrides() const;
};
