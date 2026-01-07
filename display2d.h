// display2d.h
#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <vector>
#include <glm/vec2.hpp>

class Engine2D;

// Keep your existing overrides; Display2D just forwards them to the presenter.
struct RenderOverrides
{
    bool useTargetOverride = false;
    glm::vec2 targetOrigin{0.0f, 0.0f};
    glm::vec2 targetSize{0.0f, 0.0f};

    bool useCrop = false;
    glm::vec2 cropOrigin{0.0f, 0.0f};
    glm::vec2 cropSize{1.0f, 1.0f};
};

// What upstream passes (decoder/blit/crop/etc.) hand to the display for presentation.
struct PresentInput
{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;

    // Layout the producer left the image in (e.g. GENERAL, SHADER_READ_ONLY_OPTIMAL).
    // The presenter is responsible for transitioning the input to what it needs (e.g. TRANSFER_SRC_OPTIMAL).
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

    // Pixel dimensions of the input image.
    VkExtent2D extent{0, 0};

    // Format of the input image.
    VkFormat format = VK_FORMAT_UNDEFINED;
};

// Tiny swapchain “view” so presenters can rebuild resources without peeking into internals.
struct SwapchainInfo
{
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{0, 0};
    uint64_t generation = 0;

    // Owned by Display2D. Do not cache across recreates.
    const std::vector<VkImage>* images = nullptr;
    const std::vector<VkImageView>* views = nullptr;
};

// Presenter interface.
class IDisplayPresenter
{
public:
    virtual ~IDisplayPresenter() = default;

    // Called whenever swapchain is created/recreated.
    virtual void onSwapchainChanged(const SwapchainInfo& sc) = 0;

    // Record commands that write the PresentInput into the acquired swapchain image.
    // Presenter handles any transitions of the *input* image it needs.
    virtual void record(VkCommandBuffer cmd,
                        VkImage swapImage,
                        VkImageView swapView,
                        VkExtent2D swapExtent,
                        const PresentInput& input,
                        const RenderOverrides& overrides) = 0;
};

class Display2D
{
public:
    static constexpr uint32_t kMaxFramesInFlight = 2;

    Display2D(Engine2D* engine,
              int width = 800,
              int height = 600,
              const char* title = "Motive 2D");
    ~Display2D();

    bool shouldClose() const;
    void pollEvents() const;

    // Convenience: render without any external synchronization.
    // (Equivalent to passing VK_NULL_HANDLE; stage is ignored in that case.)
    void renderFrame();

    // Render, waiting on an upstream semaphore (e.g. compute completion).
    // Typical for transfer-based presenters: externalWaitStage = VK_PIPELINE_STAGE_TRANSFER_BIT.
    void renderFrame(VkSemaphore externalWaitSemaphore,
                     VkPipelineStageFlags externalWaitStage);

    void setPresentInput(const PresentInput& input) { presentInput_ = input; }
    void clearPresentInput() { presentInput_ = {}; }

    void setOverrides(const RenderOverrides& o) { overrides_ = o; }
    const RenderOverrides& overrides() const { return overrides_; }

    void setPresenter(IDisplayPresenter* presenter) { presenter_ = presenter; }
    IDisplayPresenter* presenter() const { return presenter_; }

    SwapchainInfo swapchainInfo() const;

    GLFWwindow* window() const { return window_; }
    bool valid() const { return swapchain_ != VK_NULL_HANDLE; }

    void shutdown();

    // --- Data ---
    Engine2D* engine = nullptr;

private:
    struct Frame
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
    };

    void createWindow_(const char* title);
    void createSurface_();
    void createSwapchain_();
    void destroySwapchain_();
    void recreateSwapchain_();

    void createFrameResources_();
    void destroyFrameResources_();

    void beginCmd_(VkCommandBuffer cmd);
    void endCmd_(VkCommandBuffer cmd);

private:
    GLFWwindow* window_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    bool framebufferResized_ = false;

    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{0, 0};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    uint64_t swapchainGeneration_ = 0;

    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    Frame frames_[kMaxFramesInFlight]{};
    uint32_t currentFrame_ = 0;

    PresentInput presentInput_{};
    RenderOverrides overrides_{};

    IDisplayPresenter* presenter_ = nullptr;

    bool shutdownPerformed_ = false;
};
