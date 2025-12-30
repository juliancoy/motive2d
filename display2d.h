#pragma once
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <chrono>
#include <array>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include "video.h"

class Engine2D;

struct ImageViewSampler
{
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

struct VideoImageSet
{
    ImageViewSampler luma;
    ImageViewSampler chroma;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t chromaDivX = 1;
    uint32_t chromaDivY = 1;
};

struct OverlayImageInfo
{
    ImageViewSampler overlay;
    VkExtent2D extent{0, 0};
    VkOffset2D offset{0, 0};
    bool enabled = false;
};

struct RenderOverrides
{
    // Optional: override where the video is placed in the output framebuffer (in pixels)
    bool useTargetOverride = false;
    glm::vec2 targetOrigin{0.0f, 0.0f};
    glm::vec2 targetSize{0.0f, 0.0f};
    // Optional: crop region in normalized video UVs (0-1)
    bool useCrop = false;
    glm::vec2 cropOrigin{0.0f, 0.0f};
    glm::vec2 cropSize{1.0f, 1.0f};
    // Optional: hide scrubber/play controls
    bool hideScrubber = false;
};

struct ColorAdjustments
{
    float exposure = 0.0f;
    float contrast = 1.0f;
    float saturation = 1.0f;
    glm::vec3 shadows{1.0f};
    glm::vec3 midtones{1.0f};
    glm::vec3 highlights{1.0f};
    std::array<float, 256> curveLut{};
    bool curveEnabled = false;
};

class Display2D
{
public:
    Display2D(Engine2D* engine, int width = 800, int height = 600, const char* title = "Motive 2D");
    ~Display2D();

    void renderFrame(const VideoImageSet& videoImages,
                     const OverlayImageInfo& overlayInfo,
                     const OverlayImageInfo& fpsOverlayInfo,
                     const video::VideoColorInfo& colorInfo,
                     float scrubProgress,
                     float scrubPlaying,
                     const RenderOverrides* overrides = nullptr,
                     const ColorAdjustments* adjustments = nullptr);
    void shutdown();
    bool shouldClose() const;
    void pollEvents() const;

    GLFWwindow* window = nullptr;
    int width = 0;
    int height = 0;

private:
    Engine2D* engine = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_B8G8R8A8_SRGB;
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkBuffer curveUBO = VK_NULL_HANDLE;
    VkDeviceMemory curveUBOMemory = VK_NULL_HANDLE;
    void* curveUBOMapped = nullptr;
    VkDeviceSize curveUBOSize = sizeof(glm::vec4) * 64; // 256 floats packed as vec4
    std::array<float, 256> lastCurveLut{};
    bool lastCurveEnabled = false;
    bool curveUploaded = false;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    size_t currentFrame = 0;

    void createWindow(const char* title);
    void createSurface();
    void createSwapchain();
    void cleanupSwapchain();
    void createCommandResources();
    void createComputeResources();
    void recreateSwapchain();
    bool shutdownPerformed = false;
};
