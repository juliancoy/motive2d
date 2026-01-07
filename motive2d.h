#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>

#include "color_grading_pass.h"
#include "color_grading_ui.h"
#include "crop.h"
#include "decoder_vulkan.h"
#include "engine2d.h"
#include "fps.h"
#include "nv12_to_rgba.h"
#include "pose_overlay.h"
#include "rect_overlay.h"
#include "scrubber.h"
#include "subtitle.h"

// Display2D is used by pointer/unique_ptr in the header.
#include "display2d.h"

const std::filesystem::path kDefaultVideoPath("P1090533_main8_hevc_fast.mkv");

// Frames-in-flight is still defined here (Motive2D owns sync slots).
// Passes are initialized with this value at construction time.
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

    bool gpuDecode = true;
};

// Frame synchronization resources (one per in-flight slot).
struct FrameResources
{
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;        // records: decode barriers + compute passes
    VkFence fence = VK_NULL_HANDLE;                        // CPU-GPU sync for this slot
    VkSemaphore computeCompleteSemaphore = VK_NULL_HANDLE; // (optional) binary semaphore if you want to chain submits

    // Optional timeline decode sync (unused unless you wire DecoderVulkan timeline semaphores)
    VkSemaphore decodeReadySemaphore = VK_NULL_HANDLE;
    uint64_t decodeSemaphoreValue = 0;
};

class Motive2D
{
public:
    explicit Motive2D(CliOptions cliOptions);
    ~Motive2D();

    Motive2D(const Motive2D&) = delete;
    Motive2D& operator=(const Motive2D&) = delete;

    void run();

    // Legacy API (keep only if something else calls it)
    void renderFrame() {}

    // Pipeline test functionality (if still used elsewhere)
    void exportPipelineTestFrame();

public:
    std::vector<std::unique_ptr<Display2D>> windows;

    Engine2D* engine = nullptr;

    Display2D* inputWindow = nullptr;
    Display2D* regionWindow = nullptr;
    Display2D* gradingWindow = nullptr;

    // Passes
    // NV12->RGBA pass now owns its RGBA output per frame-in-flight.
    // (This type should exist in your codebase per the updated motive2d.cpp.)
    class Nv12ToRgbaPass* nv12Pass = nullptr;

    // RGBA->RGBA grading pass owns its output per frame-in-flight.
    ColorGrading* colorGrading = nullptr;

    // UI / overlays
    ColorGradingUi* colorGradingUi = nullptr;
    Subtitle* subtitle = nullptr;
    RectOverlay* rectOverlay = nullptr;
    PoseOverlay* poseOverlay = nullptr;
    Crop* crop = nullptr;
    Scrubber* scrubber = nullptr;
    FpsOverlay* fpsOverlay = nullptr;

    // Decode
    DecoderVulkan* decoder = nullptr;

    CliOptions options;

    // Synchronization resources (Motive2D owns the in-flight slots)
    std::vector<FrameResources> frames;
    int currentFrame = 0;

private:
    void createSynchronizationObjects();
    void destroySynchronizationObjects();

    void recordComputeCommands(VkCommandBuffer commandBuffer, int frameIndex, const VulkanSurface& surf);
};

static inline float intersection_area(const PoseObject& a, const PoseObject& b)
{
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    const float width = std::max(0.0f, right - left);
    const float height = std::max(0.0f, bottom - top);
    return width * height;
}
