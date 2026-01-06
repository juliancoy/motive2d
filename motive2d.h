
#include <fstream>
#include <optional>
#include <filesystem>
#include <vector>
#include "pose_overlay.h"
#include "rect_overlay.h"
#include "color_grading_pass.h"
#include "color_grading_ui.h"
#include "subtitle.h"
#include "decoder_vulkan.h"
#include "scrubber.h"   
#include "nv12toBGR.h"

const std::filesystem::path kDefaultVideoPath("P1090533_main8_hevc_fast.mkv");

// Maximum number of frames that can be in flight (triple buffering)
constexpr int MAX_FRAMES_IN_FLIGHT = 3;


struct CliOptions
{
    std::filesystem::path videoPath = kDefaultVideoPath;
    std::optional<bool> swapUV;
    bool showInput = true;
    bool showRegion = true;
    bool showGrading = true;
    bool poseEnabled = false;
    bool overlaysEnabled = true;
    bool scrubberEnabled = true;
    bool debugLogging = false;
    std::filesystem::path poseModelBase = "yolov8n_pose";
    bool debugDecode = false;
    bool inputOnly = false;
    bool skipBlit = false;
    bool singleFrame = false;
    std::filesystem::path outputImagePath = "frame.png";
    bool subtitleBackground = true;
    bool pipelineTest = false;
    std::filesystem::path pipelineTestDir = "intermittant";
    bool gpuDecode = false;  // Force Vulkan hardware decoding
};


// Intermediate BGR image for processing (after NV12 conversion)
struct BGRImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Frame synchronization resources
struct FrameResources {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;      // Single buffer for all compute
    VkFence fence = VK_NULL_HANDLE;                      // CPU-GPU sync
    VkSemaphore decodeReadySemaphore = VK_NULL_HANDLE;   // Timeline (optional)
    VkSemaphore computeCompleteSemaphore = VK_NULL_HANDLE; // Binary (for present)
    uint64_t decodeSemaphoreValue = 0;

    // Descriptor sets for compute pipelines
    VkDescriptorSet nv12toBGRDescriptorSet = VK_NULL_HANDLE;
};

class Motive2D {   
public:
    void renderFrame();
    Motive2D(CliOptions cliOptions);
    void run();
    ~Motive2D();
    
    // Pipeline test functionality
    void exportPipelineTestFrame();
    
    std::vector<std::unique_ptr<Display2D>> windows;

    Engine2D *engine; // this is kiond of like the context

    Display2D *inputWindow = nullptr;
    Display2D *regionWindow = nullptr;
    Display2D *gradingWindow = nullptr;

    ColorGrading * colorGrading = nullptr;
    ColorGradingUi * colorGradingUi = nullptr;
    Subtitle * subtitle;
    RectOverlay * rectOverlay;
    PoseOverlay * poseOverlay;
    Crop * crop;
    nv12toBGR * nv12toBGRPipeline = nullptr;
    VkSampler * blackSampler;
    VideoImageSet * blackVideo;
    Scrubber * scrubber;
    FpsOverlay * fpsOverlay;
    DecoderVulkan * decoder;
    
    CliOptions options; // Store command line options

private:
    // Synchronization resources
    std::vector<FrameResources> frames;
    int currentFrame = 0;

    // Intermediate images for processing
    std::vector<BGRImage> bgrImages; // One per frame in flight

    // Descriptor resources
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;

    // Helper functions
    void createSynchronizationObjects();
    void destroySynchronizationObjects();
    void createComputeResources();
    void destroyComputeResources();
    void recordComputeCommands(VkCommandBuffer commandBuffer, int frameIndex);
    void updateDescriptorSets(int frameIndex);
};


static inline float intersection_area(const PoseObject &a, const PoseObject &b)
{
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    const float width = std::max(0.0f, right - left);
    const float height = std::max(0.0f, bottom - top);
    return width * height;
}
