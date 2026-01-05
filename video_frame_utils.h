#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <vulkan/vulkan.h>

// Forward declarations
enum class PrimitiveYuvFormat : uint32_t;
enum class DecodeImplementation;

class Decoder;  // Forward declaration

extern "C" {
struct AVCodec;
struct AVStream;
struct AVCodecContext;
struct AVFrame;
#include <libavutil/pixfmt.h>
#include <libavutil/hwcontext.h>
}

enum class PrimitiveYuvFormat : uint32_t {
    None = 0,
    NV12 = 1,
    Planar420 = 2,
    Planar422 = 3,
    Planar444 = 4
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

// Format configuration
bool configureFormatForPixelFormat(Decoder& decoder, AVPixelFormat pixFmt);

// Frame copying
void copyDecodedFrameToBuffer(const Decoder& decoder,
                              AVFrame* frame,
                              std::vector<uint8_t>& buffer);

// Debug utilities
bool debugLoggingEnabled();
std::string pixelFormatDescription(AVPixelFormat fmt);
std::string colorSpaceName(AVColorSpace cs);
std::string colorRangeName(AVColorRange cr);
std::string ffmpegErrorString(int err);

// Stream analysis
int determineStreamBitDepth(AVStream* stream, AVCodecContext* codecCtx);
AVPixelFormat pickPreferredSwPixelFormat(int bitDepth, AVPixelFormat sourceFormat);

// Hardware decoding configuration
bool configureDecodeImplementation(
    Decoder& decoder,
    const AVCodec* codec,
    DecodeImplementation implementation,
    const std::optional<VulkanInteropContext>& vulkanInterop,
    bool requireGraphicsQueue,
    bool debugLogging);
