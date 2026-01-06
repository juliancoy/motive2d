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
    void createComputeResources();



    // Disable copy
    Engine2D(const Engine2D&) = delete;
    Engine2D& operator=(const Engine2D&) = delete;

    // Initialize the engine (Vulkan, GLFW, etc.)
    // `requireWindow` controls whether GLFW initialization is performed.
    bool initialize(bool requireWindow = true);

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

    // Render a frame to all windows
    bool renderFrame();

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

    bool initialized = false;
    bool glfwInitialized = false;
    
    // Video state
    bool videoLoaded = false;
    float duration = 0.0f;
    float currentTime = 0.0f;
    bool playing = true;
    bool decodeDebugEnabled = false;
    
    
    // FPS tracking
    std::chrono::steady_clock::time_point fpsLastSample;
    int fpsFrameCounter = 0;
    float currentFps = 0.0f;
    
    // Internal methods
    void updateFpsOverlay();
};
