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

extern "C"
{
    struct AVFormatContext;
    struct AVCodecContext;
    struct AVFrame;
    struct AVPacket;
    struct AVBufferRef;
    struct AVCodec;
#include <libavutil/pixfmt.h>
#include <libavutil/hwcontext.h>
}

int runDecodeOnlyBenchmark(const std::filesystem::path &videoPath,
                                double benchmarkSeconds);

struct DecodedFrame
{
    std::vector<uint8_t> buffer;
    double ptsSeconds = 0.0;

};

// Forward declarations
class Engine2D;

class DecoderVulkan
{
public:
    DecoderVulkan(const std::filesystem::path &videoPath, 
                  Engine2D* engine);
    ~DecoderVulkan();

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
    
    bool requireGraphicsQueue = true;
    bool debugLogging = false;
    VkSampler sampler = VK_NULL_HANDLE;
    VkImageView externalLumaView = VK_NULL_HANDLE;
    VkImageView externalChromaView = VK_NULL_HANDLE;
    bool usingExternal = false;
    
    Engine2D* engine = nullptr;

    // Public interface
    bool initialize();
    bool seek(float timeSeconds);
    bool acquireDecodedFrame(DecodedFrame &outFrame);
    bool startAsyncDecoding(size_t maxBufferedFrames = 12);
    void stopAsyncDecoding();

    // Frame upload and playback
    bool uploadDecodedFrame(Engine2D *engine, const DecodedFrame &frame);
    double advancePlayback();

    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    double getDuration() const { return durationSeconds; }
    double getCurrentTime() const { return currentTimeSeconds; }
    bool isPlaying() const { return playing; }
    bool isStopRequested() const { return stopRequested.load(); }
    const std::string& getHardwareInitFailureReason() const { return hardwareInitFailureReason; }

    // Private methods
    bool seekVideoDecoder(double targetSeconds);
    bool decodeNextFrame(DecodedFrame &decodedFrame);
    void pumpDecodedFrames();
    void cleanupVideoDecoder();
    void asyncDecodeLoop();

    // Helper methods
    VkSampler createLinearClampSampler();
    void destroyExternalVideoViews();
    bool waitForVulkanFrameReady();
    
    // Configuration
    bool configureDecodeImplementation(
        const AVCodec *codec,
        bool requireGraphicsQueue,
        bool debugLogging);
    bool configureFormatForPixelFormat(AVPixelFormat pix_fmt);

    // FFmpeg resources
    AVFormatContext *formatCtx = nullptr;
    AVCodecContext *codecCtx = nullptr;
    AVFrame *frame = nullptr;
    AVPacket *packet = nullptr;
    AVBufferRef *hwDeviceCtx = nullptr;

    // Video properties
    int videoStreamIndex = -1;
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
    AVHWDeviceType hwDeviceType = AV_HWDEVICE_TYPE_VULKAN;
    AVPixelFormat hwPixelFormat = AV_PIX_FMT_VULKAN;
    AVPixelFormat sourcePixelFormat = AV_PIX_FMT_NONE;
    std::string hardwareInitFailureReason;
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


private:
    // Playback state
    bool playing = false;
    std::deque<DecodedFrame> pendingFrames;
    bool playbackClockInitialized = false;
    double basePtsSeconds = 0.0;
    double lastFramePtsSeconds = 0.0;
    double lastDisplayedSeconds = 0.0;
    std::chrono::steady_clock::time_point lastFrameRenderTime;

};