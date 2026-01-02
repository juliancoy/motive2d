#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <optional>
#include <chrono>
#include <array>
#include <vector>
#include <GLFW/glfw3.h>
#include "display2d.h"
#include "video.h"
#include "decode.h"
#include "overlay.hpp"
#include "grading.hpp"
#include "graphicsdevice.h"


struct CropRegion {
    float x = 0.0f;  // normalized 0-1
    float y = 0.0f;  // normalized 0-1
    float width = 1.0f;  // normalized 0-1
    float height = 1.0f; // normalized 0-1
};

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

    VideoPlaybackState& getPlaybackState();
    overlay::RectOverlayCompute& getRectOverlayCompute();
    overlay::PoseOverlayCompute& getPoseOverlayCompute();

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

 private:

    std::vector<std::unique_ptr<Display2D>> windows;
    bool initialized = false;
    bool glfwInitialized = false;
    
    // Video state
    VideoPlaybackState playbackState;
    bool videoLoaded = false;
    float duration = 0.0f;
    float currentTime = 0.0f;
    bool playing = true;
    bool decodeDebugEnabled = false;
    
    // Grading and crop state
    std::optional<GradingSettings> gradingSettings;
    std::optional<CropRegion> cropRegion;
    
    // Overlay resources
    overlay::RectOverlayCompute rectOverlayCompute;
    overlay::PoseOverlayCompute poseOverlayCompute;
    bool overlayInitialized = false;
    
    // FPS tracking
    std::chrono::steady_clock::time_point fpsLastSample;
    int fpsFrameCounter = 0;
    float currentFps = 0.0f;
    
    // Internal methods
    bool initializeOverlay();
    void updateFpsOverlay();
    void applyGrading(ColorAdjustments& adjustments) const;
    RenderOverrides buildRenderOverrides() const;
};
