// decoder_vulkan.h
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

// Forward decl
class Engine2D;

extern "C" {
    struct AVFormatContext;
    struct AVCodecContext;
    struct AVFrame;
    struct AVPacket;
    struct AVBufferRef;
    struct AVCodec;
    struct AVRational;

    // If you prefer, you can include <libavutil/pixfmt.h> here instead.
    enum AVPixelFormat;
}

struct VulkanSurface
{
    bool valid = false;

    uint32_t width = 0;
    uint32_t height = 0;

    // We only support the common case (NV12/NV21 or 2-plane Vulkan decode surfaces).
    // If you later need >2, increase this.
    uint32_t planes = 0;

    std::array<VkImage, 2> images{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageLayout, 2> layouts{VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};

    // FFmpeg Vulkan frames often provide timeline semaphores for readiness.
    std::array<VkSemaphore, 2> semaphores{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<uint64_t, 2> semaphoreValues{0, 0};

    // Queue family that currently “owns” the images (FFmpeg side).
    std::array<uint32_t, 2> queueFamily{VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED};

    // Best-effort VkFormat for each plane (may be VK_FORMAT_UNDEFINED if unknown).
    std::array<VkFormat, 2> planeFormats{VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};

    bool validate() const
    {
        if (!valid) return false;
        if (width == 0 || height == 0) return false;
        if (planes == 0 || planes > 2) return false;
        if (images[0] == VK_NULL_HANDLE) return false;
        if (planes > 1 && images[1] == VK_NULL_HANDLE) return false;
        return true;
    }
};

// Minimal decoded frame wrapper that keeps the AVFrame alive so the Vulkan images stay valid.
struct DecodedFrame
{
    double ptsSeconds = 0.0;

    VulkanSurface vk{};
    AVFrame* avFrame = nullptr; // owns a clone

    ~DecodedFrame();
    DecodedFrame() = default;
    DecodedFrame(const DecodedFrame&) = delete;
    DecodedFrame& operator=(const DecodedFrame&) = delete;
    DecodedFrame(DecodedFrame&& other) noexcept;
    DecodedFrame& operator=(DecodedFrame&& other) noexcept;

    void reset();
};

// Simple bounded queue for decoded frames (producer/consumer).
template <typename T, size_t Capacity>
class BoundedQueue
{
public:
    bool push(T&& item);
    bool pop(T& out);
    bool try_pop(T& out);
    void stop();
    void reset();
    size_t size() const;

private:
    mutable std::mutex m_;
    std::condition_variable cvNotEmpty_;
    std::condition_variable cvNotFull_;

    std::array<T, Capacity> buf_{};
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
    bool stopped_ = false;
};

// What kind of YUV layout the source *conceptually* is.
// (This is not the old PrimitiveYuvFormat; it’s local to this decoder.)
enum class YuvLayout
{
    NV12,      // NV12/NV21 two-plane interleaved chroma
    Planar420, // 4:2:0 planar (3 planes logically; FFmpeg Vulkan decode may still expose 2 images depending on impl)
    Planar422,
    Planar444,
    Unknown
};

class DecoderVulkan
{
public:
    static constexpr size_t kBufferedFrames = 10;

    DecoderVulkan(const std::filesystem::path& videoPath, Engine2D* eng);
    ~DecoderVulkan();

    DecoderVulkan(const DecoderVulkan&) = delete;
    DecoderVulkan& operator=(const DecoderVulkan&) = delete;

    // Status
    bool valid = false;
    const std::string& getHardwareInitFailureReason() const { return hardwareInitFailureReason; }

    // Dimensions / timing
    int getWidth() const { return static_cast<int>(width); }
    int getHeight() const { return static_cast<int>(height); }
    double getFps() const { return fps; }
    double getDurationSeconds() const { return durationSeconds; }

    // Output views (rebuilt per presented frame)
    VkImageView externalLumaView = VK_NULL_HANDLE;
    VkImageView externalChromaView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    // Latched surface metadata for the currently displayed frame.
    bool getCurrentSurface(VulkanSurface& out) const
    {
        std::lock_guard<std::mutex> lk(currentSurfaceMutex_);
        if (!currentSurface_.validate()) return false;
        out = currentSurface_;
        return true;
    }

    // Decode thread control
    bool startAsyncDecoding();
    void stopAsyncDecoding();
    bool isStopRequested() const { return stopRequested.load(); }

    // Playback (consumer side)
    // Returns "seconds displayed since playback start".
    double advancePlayback();
    bool seek(float timeSeconds);
    void resetPlaybackClock();

    // Optional: allow consumer to toggle playback without killing decode thread.
    void setPlaying(bool p) { playing = p; }
    bool isPlaying() const { return playing; }

private:
    // ---- FFmpeg setup / teardown ----
    bool openInputAndCodec(const std::filesystem::path& videoPath);
    bool initFFmpegVulkanDevice();
    void cleanupFFmpeg();

    // Pixel format configuration
    bool configureFormatForPixelFormat(AVPixelFormat pix_fmt);

    // Single-frame decode (used by async thread; also used by decode-only benchmark)
    bool decodeNextFrame(DecodedFrame& out);

    // ---- Async decode loop ----
    void asyncDecodeLoop();

    // ---- Vulkan helpers ----
    VkSampler createLinearClampSampler();
    void destroyExternalVideoViews();
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect);

    bool waitForVulkanFrameReady(const VulkanSurface& s);
    bool createExternalViewsFromSurface(const VulkanSurface& s);

    // ---- Playback timing ----
    void updatePlaybackTimestamps(const DecodedFrame& frame, std::chrono::steady_clock::time_point now);
    bool shouldDisplayNow(const DecodedFrame& frame, std::chrono::steady_clock::time_point now) const;
    void dropLateFrames(std::chrono::steady_clock::time_point now);

private:
    Engine2D* engine = nullptr;

    // FFmpeg core objects
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    AVBufferRef* hwDeviceCtx = nullptr;

    int videoStreamIndex = -1;
    AVRational streamTimeBase{0, 1};

    // Stream state
    uint32_t width = 0;
    uint32_t height = 0;
    double fps = 30.0;
    double durationSeconds = 0.0;

    // Pixel format state
    AVPixelFormat sourcePixelFormat = static_cast<AVPixelFormat>(-1);

    // Chroma subsampling divisors (2^log2_chroma_w / 2^log2_chroma_h)
    uint32_t chromaDivX = 2;
    uint32_t chromaDivY = 2;
    uint32_t chromaWidth = 0;
    uint32_t chromaHeight = 0;

    // Bit depth info (for future 10/12-bit handling)
    int bitDepth = 8;
    int bytesPerComponent = 1;

    // High-level layout info
    YuvLayout yuvLayout = YuvLayout::Unknown;
    bool swapChromaUV = false; // true for NV21-style UV swap

    // Decode queue
    BoundedQueue<DecodedFrame, kBufferedFrames> decodedQ;

    std::atomic<bool> asyncDecoding{false};
    std::thread decodeThread;

    std::atomic<bool> stopRequested{false};
    std::atomic<bool> threadRunning{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> draining{false};

    // Seeking: when set >=0, producer drops frames until pts >= target
    std::atomic<int64_t> seekTargetMicroseconds{-1};

    // Playback
    bool playing = true;
    bool clockInitialized = false;
    double firstPtsSeconds = 0.0;
    double lastFramePtsSeconds = 0.0;
    double lastDisplayedSeconds = 0.0;

    std::chrono::steady_clock::time_point playbackStartWall{};
    std::chrono::steady_clock::time_point lastFrameRenderWall{};

    size_t framesDecoded = 0;
    double fallbackPtsSeconds = 0.0;

    std::optional<DecodedFrame> candidate;

    // Current latched surface metadata for the most recently presented frame
    mutable std::mutex currentSurfaceMutex_;
    VulkanSurface currentSurface_{};

    // External views state
    bool usingExternal = false;

    // Human-readable init failure
    std::string hardwareInitFailureReason;

    // Allow decode-only benchmark to call decodeNextFrame
    friend int runDecodeOnlyBenchmark(const std::filesystem::path& videoPath, double benchmarkSeconds);
};
