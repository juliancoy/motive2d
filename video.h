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

#include <vulkan/vulkan.h>


enum class PrimitiveYuvFormat : uint32_t {
    None = 0,
    NV12 = 1,
    Planar420 = 2,
    Planar422 = 3,
    Planar444 = 4
};


namespace glyph {
struct OverlayBitmap;
}

extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVBufferRef;
#include <libavutil/pixfmt.h>
#include <libavutil/hwcontext.h>
}

namespace video {

enum class DecodeImplementation {
    Software = 0,
    Vulkan = 1
};

struct VulkanInteropContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamilyIndex = 0;
    VkQueue videoQueue = VK_NULL_HANDLE;
    uint32_t videoQueueFamilyIndex = 0;
};

struct DecodedFrame;

struct DecoderInitParams {
    DecodeImplementation implementation = DecodeImplementation::Software;
    std::optional<VulkanInteropContext> vulkanInterop;
    bool requireGraphicsQueue = true;
};

struct VideoDecoder {
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* swFrame = nullptr;
    AVPacket* packet = nullptr;
    AVBufferRef* hwDeviceCtx = nullptr;
    AVBufferRef* hwFramesCtx = nullptr;
    int videoStreamIndex = -1;
    int width = 0;
    int height = 0;
    int bufferSize = 0;
    int yPlaneSize = 0;
    int uvPlaneSize = 0;
    AVColorSpace colorSpace = AVCOL_SPC_UNSPECIFIED;
    AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED;
    double fps = 30.0;
    bool draining = false;
    std::atomic<bool> finished{false};
    DecodeImplementation implementation = DecodeImplementation::Software;
    AVHWDeviceType hwDeviceType = AV_HWDEVICE_TYPE_NONE;
    AVPixelFormat hwPixelFormat = AV_PIX_FMT_NONE;
    AVPixelFormat sourcePixelFormat = AV_PIX_FMT_NONE;
    AVPixelFormat requestedSwPixelFormat = AV_PIX_FMT_NONE;
    std::string implementationName = "Software";
    PrimitiveYuvFormat outputFormat = PrimitiveYuvFormat::NV12;
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
    // Async decoding
    std::thread decodeThread;
    std::mutex frameMutex;
    std::condition_variable frameCond;
    AVRational streamTimeBase{1, 1};
    double fallbackPtsSeconds = 0.0;
    uint64_t framesDecoded = 0;
    std::deque<DecodedFrame> frameQueue;
    size_t maxBufferedFrames = 12;
    bool asyncDecoding = false;
    std::atomic<bool> stopRequested{false};
    std::atomic<bool> threadRunning{false};
    std::atomic<int64_t> seekTargetMicroseconds{-1};
};

std::optional<std::filesystem::path> locateVideoFile(const std::string& filename);
bool initializeVideoDecoder(const std::filesystem::path& videoPath,
                            VideoDecoder& decoder,
                            const DecoderInitParams& initParams = DecoderInitParams{});
bool seekVideoDecoder(VideoDecoder& decoder, double targetSeconds);
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

bool decodeNextFrame(VideoDecoder& decoder,
                     DecodedFrame& decodedFrame,
                     bool copyFrameBuffer = true);
bool startAsyncDecoding(VideoDecoder& decoder, size_t maxBufferedFrames = 12);
bool acquireDecodedFrame(VideoDecoder& decoder, DecodedFrame& outFrame);
void stopAsyncDecoding(VideoDecoder& decoder);
void cleanupVideoDecoder(VideoDecoder& decoder);
std::vector<std::string> listAvailableHardwareDevices();
enum class VideoColorSpace : uint32_t {
    BT601 = 0,
    BT709 = 1,
    BT2020 = 2
};

enum class VideoColorRange : uint32_t {
    Limited = 0,
    Full = 1
};

struct VideoColorInfo {
    VideoColorSpace colorSpace = VideoColorSpace::BT709;
    VideoColorRange colorRange = VideoColorRange::Limited;
};

struct Nv12Overlay {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t offsetX = 0;
    uint32_t offsetY = 0;
    uint32_t uvWidth = 0;
    uint32_t uvHeight = 0;
    std::vector<uint8_t> yPlane;
    std::vector<uint8_t> yAlpha;
    std::vector<uint8_t> uvPlane;
    std::vector<uint8_t> uvAlpha;

    bool isValid() const
    {
        return width > 0 && height > 0 && !yPlane.empty() && !uvPlane.empty();
    }
};

VideoColorInfo deriveVideoColorInfo(const VideoDecoder& decoder);
Nv12Overlay convertOverlayToNv12(const glyph::OverlayBitmap& bitmap, const VideoColorInfo& colorInfo);
void applyNv12Overlay(std::vector<uint8_t>& nv12Buffer,
                      uint32_t frameWidth,
                      uint32_t frameHeight,
                      const Nv12Overlay& overlay);
void applyOverlayToDecodedFrame(std::vector<uint8_t>& buffer,
                                const VideoDecoder& decoder,
                                const Nv12Overlay& overlay);

} // namespace video
