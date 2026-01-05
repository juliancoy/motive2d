#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include "video_frame_utils.h"

#include <vulkan/vulkan.h>

extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVBufferRef;
struct AVCodec;
#include <libavutil/pixfmt.h>
#include <libavutil/hwcontext.h>
}

// Forward declarations
class Engine2D;

namespace video {
    struct VulkanInteropContext {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        uint32_t graphicsQueueFamilyIndex = 0;
        VkQueue videoQueue = VK_NULL_HANDLE;
        uint32_t videoQueueFamilyIndex = 0;
    };
}

enum class DecodeImplementation {
    Software = 0,
    Vulkan = 1
};

struct DecoderInitParams {
    DecodeImplementation implementation = DecodeImplementation::Software;
    std::optional<video::VulkanInteropContext> vulkanInterop;
    bool requireGraphicsQueue = true;
    bool debugLogging = false;
};

struct DecodedFrame {
    std::vector<uint8_t> buffer;
    double ptsSeconds = 0.0;
    
    struct VulkanSurface {
        bool valid = false;
        uint32_t planes = 0;
        VkImage images[2]{};
        VkImageLayout layouts[2]{};
        VkSemaphore semaphores[2]{};
        uint64_t semaphoreValues[2]{};
        uint32_t queueFamily[2]{};
        VkFormat planeFormats[2]{};
        uint32_t width = 0;
        uint32_t height = 0;
    } vkSurface;
};

struct VideoResources {
    VkSampler sampler = VK_NULL_HANDLE;
    VkImageView externalLumaView = VK_NULL_HANDLE;
    VkImageView externalChromaView = VK_NULL_HANDLE;
    bool usingExternal = false;
    
    // Add other resources needed for rendering
};

class Decoder
{
public:
    Decoder(const std::filesystem::path& videoPath, const DecoderInitParams& initParams);
    ~Decoder();
    
    // Public interface
    bool initialize();
    bool seek(float timeSeconds);
    bool acquireDecodedFrame(DecodedFrame& outFrame);
    bool startAsyncDecoding(size_t maxBufferedFrames = 12);
    void stopAsyncDecoding();
    
    // Frame upload and playback
    bool uploadDecodedFrame(Engine2D* engine, const DecodedFrame& frame);
    double advancePlayback();
    std::string hardwareInitFailureReason;
    
    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    double getDuration() const { return durationSeconds; }
    double getCurrentTime() const { return currentTimeSeconds; }
    bool isPlaying() const { return playing; }
    bool isStopRequested() const { return stopRequested.load(); }
    
    // Benchmark
    static int runDecodeOnlyBenchmark(const std::filesystem::path& videoPath,
                                      const std::optional<bool>& swapUvOverride,
                                      double benchmarkSeconds = 5.0);
    bool configureFormatForPixelFormat(AVPixelFormat pix_fmt);

    // Private methods
    bool initializeVideoDecoder(const std::filesystem::path& videoPath, const DecoderInitParams& initParams);
    bool seekVideoDecoder(double targetSeconds);
    bool decodeNextFrame(DecodedFrame& decodedFrame, bool copyFrameBuffer = true);
    void pumpDecodedFrames();
    void cleanupVideoDecoder();
    
    // Helper methods
    VkSampler createLinearClampSampler(Engine2D* engine);
    void destroyExternalVideoViews(Engine2D* engine, VideoResources& video);
    bool waitForVulkanFrameReady(Engine2D* engine, const DecodedFrame::VulkanSurface& surface);
    
    // Async decoding
    void asyncDecodeLoop();
    
    // FFmpeg resources
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* swFrame = nullptr;
    AVPacket* packet = nullptr;
    AVBufferRef* hwDeviceCtx = nullptr;
    AVBufferRef* hwFramesCtx = nullptr;
    
    // Video properties
    int videoStreamIndex = -1;
    int width = 0;
    int height = 0;
    double durationSeconds = 0.0;
    double currentTimeSeconds = 0.0;
    double fps = 30.0;
    AVRational streamTimeBase{1, 1};
    
    // Decoding state
    std::atomic<bool> finished{false};
    std::atomic<bool> draining{false};
    std::atomic<bool> stopRequested{false};
    std::atomic<bool> threadRunning{false};
    std::atomic<int64_t> seekTargetMicroseconds{-1};
    uint64_t framesDecoded = 0;
    double fallbackPtsSeconds = 0.0;
    
    // Format info
    DecodeImplementation implementation = DecodeImplementation::Software;
    AVHWDeviceType hwDeviceType = AV_HWDEVICE_TYPE_NONE;
    AVPixelFormat hwPixelFormat = AV_PIX_FMT_NONE;
    AVPixelFormat sourcePixelFormat = AV_PIX_FMT_NONE;
    AVPixelFormat requestedSwPixelFormat = AV_PIX_FMT_NONE;
    std::string implementationName = "Software";
    PrimitiveYuvFormat outputFormat = PrimitiveYuvFormat::NV12;
    
    // YUV format details
    bool planarYuv = false;
    bool swapChromaUV = false;
    bool chromaInterleaved = false;
    uint32_t chromaDivX = 2;
    uint32_t chromaDivY = 2;
    uint32_t chromaWidth = 0;
    uint32_t chromaHeight = 0;
    size_t bytesPerComponent = 1;
    uint32_t bitDepth = 8;
    size_t yPlaneBytes = 0;
    size_t uvPlaneBytes = 0;
    int bufferSize = 0;
    
    // Color info
    AVColorSpace colorSpace = AVCOL_SPC_UNSPECIFIED;
    AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED;
    
    // Async decoding
    std::thread decodeThread;
    std::mutex frameMutex;
    std::condition_variable frameCond;
    std::deque<DecodedFrame> frameQueue;
    size_t maxBufferedFrames = 12;
    bool asyncDecoding = false;
    
    // Playback state
    bool playing = false;
    std::deque<DecodedFrame> pendingFrames;
    bool playbackClockInitialized = false;
    double basePtsSeconds = 0.0;
    double lastFramePtsSeconds = 0.0;
    double lastDisplayedSeconds = 0.0;
    std::chrono::steady_clock::time_point lastFrameRenderTime;
    
    // Resources
    VideoResources videoResources;
    void copyDecodedFrameToBuffer(const AVFrame *frame, std::vector<uint8_t> &buffer);
    bool configureDecodeImplementation(
                                        const AVCodec *codec,
                                        DecodeImplementation decodeImplementation,
                                        const std::optional<video::VulkanInteropContext> &vulkanInterop,
                                        bool requireGraphicsQueue,
                                        bool debugLogging);
private:
    // Engine reference
    Engine2D* engine = nullptr;
};
