#include "decoder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <array>
#include <optional>
#include <string>
#include <sstream>
#include <memory>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec2.hpp>

#include <vulkan/vulkan.h>

#include "display2d.h"
#include "engine2d.h"
#include "fps.h"
#include "color_grading_ui.h"
#include "utils.h"
#include "video_frame_utils.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

// Look for the sample video in the current directory (files were moved up).
const std::filesystem::path kDefaultVideoPath = std::filesystem::path("P1090533_main8_hevc_fast.mkv");

// Interrupt callback for FFmpeg
static int interrupt_callback(void *opaque)
{
    Decoder *decoder = static_cast<Decoder *>(opaque);
    if (decoder && decoder->isStopRequested())
        return 1;
    return 0;
}

// Determine stream bit depth
int determineStreamBitDepth(AVStream *stream, AVCodecContext *codecCtx)
{
    if (stream && stream->codecpar)
    {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(stream->codecpar->format));
        if (desc)
            return desc->comp[0].depth;
    }
    return 8;
}

// Pick preferred software pixel format
AVPixelFormat pickPreferredSwPixelFormat(int bitDepth, AVPixelFormat sourceFormat)
{
    if (bitDepth > 8)
    {
        // Prefer packed 16-bit formats for high bit depth
        if (sourceFormat == AV_PIX_FMT_P010LE || sourceFormat == AV_PIX_FMT_P016LE)
            return AV_PIX_FMT_P016LE;
        return AV_PIX_FMT_YUV420P16LE;
    }
    // Default to NV12 for 8-bit
    return AV_PIX_FMT_NV12;
}

// Configure decode implementation
bool Decoder::configureDecodeImplementation(
    const AVCodec *codec,
    DecodeImplementation decodeImplementation,
    const std::optional<video::VulkanInteropContext> &vulkanInterop,
    bool requireGraphicsQueue,
    bool debugLogging)
{
    implementation = decodeImplementation;

    if (implementation == DecodeImplementation::Software)
    {
        implementationName = "Software";
        hwDeviceType = AV_HWDEVICE_TYPE_NONE;
        hwPixelFormat = AV_PIX_FMT_NONE;
        return true;
    }

    if (implementation == DecodeImplementation::Vulkan)
    {
        if (!vulkanInterop.has_value())
        {
            hardwareInitFailureReason = "Vulkan interop context not provided";
            return false;
        }

        const auto &interop = vulkanInterop.value();
        if (interop.device == VK_NULL_HANDLE || interop.physicalDevice == VK_NULL_HANDLE)
        {
            hardwareInitFailureReason = "Invalid Vulkan handles";
            return false;
        }

        hwDeviceType = AV_HWDEVICE_TYPE_VULKAN;
        hwPixelFormat = AV_PIX_FMT_VULKAN;
        implementationName = "Vulkan";

        // Configure Vulkan device context
        hwDeviceCtx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
        if (!hwDeviceCtx)
        {
            hardwareInitFailureReason = "Failed to allocate Vulkan device context";
            return false;
        }

        AVHWDeviceContext *deviceCtx = reinterpret_cast<AVHWDeviceContext *>(hwDeviceCtx->data);
        AVVulkanDeviceContext *vulkanCtx = static_cast<AVVulkanDeviceContext *>(deviceCtx->hwctx);

        vulkanCtx->inst = interop.instance;
        vulkanCtx->phys_dev = interop.physicalDevice;
        vulkanCtx->act_dev = interop.device;
        vulkanCtx->queue_family_index = interop.videoQueueFamilyIndex;
        vulkanCtx->nb_graphics_queues = 1;
        vulkanCtx->queue_family_tx_index = interop.graphicsQueueFamilyIndex;
        vulkanCtx->nb_tx_queues = 1;

        // Set device features
        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        vkGetPhysicalDeviceFeatures(interop.physicalDevice, &features2.features);
        vulkanCtx->device_features = features2;

        int ret = av_hwdevice_ctx_init(hwDeviceCtx);
        if (ret < 0)
        {
            hardwareInitFailureReason = "Failed to initialize Vulkan device context: " + ffmpegErrorString(ret);
            av_buffer_unref(&hwDeviceCtx);
            return false;
        }

        codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx);
        return true;
    }

    hardwareInitFailureReason = "Unsupported implementation";
    return false;
}

// Configure format for pixel format
bool Decoder::configureFormatForPixelFormat(AVPixelFormat pix_fmt)
{
    sourcePixelFormat = pix_fmt;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);
    if (!desc)
        return false;

    planarYuv = !(desc->flags & AV_PIX_FMT_FLAG_PLANAR) ? false : true;
    chromaInterleaved = (desc->flags & AV_PIX_FMT_FLAG_PLANAR) ? false : true;

    // Set chroma subsampling
    chromaDivX = 1 << desc->log2_chroma_w;
    chromaDivY = 1 << desc->log2_chroma_h;
    chromaWidth = std::max<uint32_t>(1u, static_cast<uint32_t>((width + chromaDivX - 1) / chromaDivX));
    chromaHeight = std::max<uint32_t>(1u, static_cast<uint32_t>((height + chromaDivY - 1) / chromaDivY));

    // Determine format
    if (pix_fmt == AV_PIX_FMT_NV12 || pix_fmt == AV_PIX_FMT_NV21)
    {
        outputFormat = PrimitiveYuvFormat::NV12;
        swapChromaUV = (pix_fmt == AV_PIX_FMT_NV21);
    }
    else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1)
    {
        outputFormat = PrimitiveYuvFormat::Planar420;
    }
    else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0)
    {
        outputFormat = PrimitiveYuvFormat::Planar422;
    }
    else if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0)
    {
        outputFormat = PrimitiveYuvFormat::Planar444;
    }
    else
    {
        outputFormat = PrimitiveYuvFormat::None;
    }

    // Set bit depth and component size
    bitDepth = desc->comp[0].depth;
    bytesPerComponent = (bitDepth > 8) ? 2 : 1;

    // Calculate plane sizes
    if (outputFormat == PrimitiveYuvFormat::NV12)
    {
        yPlaneBytes = width * height * bytesPerComponent;
        uvPlaneBytes = chromaWidth * chromaHeight * 2 * bytesPerComponent;
    }
    else
    {
        yPlaneBytes = width * height * bytesPerComponent;
        uvPlaneBytes = chromaWidth * chromaHeight * 2 * bytesPerComponent;
    }

    bufferSize = static_cast<int>(yPlaneBytes + uvPlaneBytes);

    return outputFormat != PrimitiveYuvFormat::None;
}

// Copy decoded frame to buffer
void Decoder::copyDecodedFrameToBuffer(const AVFrame *frame, std::vector<uint8_t> &buffer)
{
    buffer.resize(bufferSize);

    if (outputFormat == PrimitiveYuvFormat::NV12)
    {
        // Copy Y plane
        const uint8_t *ySrc = frame->data[0];
        uint8_t *yDst = buffer.data();
        for (int i = 0; i < height; ++i)
        {
            std::memcpy(yDst, ySrc, width * bytesPerComponent);
            ySrc += frame->linesize[0];
            yDst += width * bytesPerComponent;
        }

        // Copy UV plane (interleaved)
        const uint8_t *uvSrc = frame->data[1];
        uint8_t *uvDst = buffer.data() + yPlaneBytes;
        int uvHeight = chromaHeight;
        int uvWidth = chromaWidth * 2;

        for (int i = 0; i < uvHeight; ++i)
        {
            std::memcpy(uvDst, uvSrc, uvWidth * bytesPerComponent);
            uvSrc += frame->linesize[1];
            uvDst += uvWidth * bytesPerComponent;
        }
    }
    else
    {
        // Planar YUV formats
        // Copy Y plane
        const uint8_t *ySrc = frame->data[0];
        uint8_t *yDst = buffer.data();
        for (int i = 0; i < height; ++i)
        {
            std::memcpy(yDst, ySrc, width * bytesPerComponent);
            ySrc += frame->linesize[0];
            yDst += width * bytesPerComponent;
        }

        // Copy U and V planes
        uint8_t *uvDst = buffer.data() + yPlaneBytes;
        if (planarYuv)
        {
            // Separate U and V planes
            const uint8_t *uSrc = frame->data[1];
            const uint8_t *vSrc = frame->data[2];

            size_t uvPlaneSize = chromaWidth * chromaHeight * bytesPerComponent;

            for (int i = 0; i < chromaHeight; ++i)
            {
                std::memcpy(uvDst, uSrc, chromaWidth * bytesPerComponent);
                uSrc += frame->linesize[1];
                uvDst += chromaWidth * bytesPerComponent;
            }

            for (int i = 0; i < chromaHeight; ++i)
            {
                std::memcpy(uvDst, vSrc, chromaWidth * bytesPerComponent);
                vSrc += frame->linesize[2];
                uvDst += chromaWidth * bytesPerComponent;
            }
        }
        else
        {
            // Interleaved UV plane
            const uint8_t *uvSrc = frame->data[1];
            for (int i = 0; i < chromaHeight; ++i)
            {
                std::memcpy(uvDst, uvSrc, chromaWidth * 2 * bytesPerComponent);
                uvSrc += frame->linesize[1];
                uvDst += chromaWidth * 2 * bytesPerComponent;
            }
        }
    }
}

// Constructor
Decoder::Decoder(const std::filesystem::path &videoPath, const DecoderInitParams &initParams)
    : engine(nullptr), playing(false)
{
    if (!initializeVideoDecoder(videoPath, initParams))
    {
        throw std::runtime_error("Failed to initialize video decoder");
    }
}

// Destructor
Decoder::~Decoder()
{
    stopAsyncDecoding();
    cleanupVideoDecoder();
}

bool Decoder::initialize()
{
    // Already initialized in constructor
    return formatCtx != nullptr && codecCtx != nullptr;
}

bool Decoder::initializeVideoDecoder(const std::filesystem::path &videoPath, const DecoderInitParams &initParams)
{
    if (avformat_open_input(&formatCtx, videoPath.string().c_str(), nullptr, nullptr) < 0)
    {
        std::cerr << "[Video] Failed to open file: " << videoPath << std::endl;
        return false;
    }

    formatCtx->interrupt_callback.callback = interrupt_callback;
    formatCtx->interrupt_callback.opaque = this;

    if (avformat_find_stream_info(formatCtx, nullptr) < 0)
    {
        std::cerr << "[Video] Unable to read stream info for: " << videoPath << std::endl;
        return false;
    }

    videoStreamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoStreamIndex < 0)
    {
        std::cerr << "[Video] No video stream found in file: " << videoPath << std::endl;
        return false;
    }

    AVStream *videoStream = formatCtx->streams[videoStreamIndex];
    const AVCodec *codec = avcodec_find_decoder(videoStream->codecpar->codec_id);
    if (!codec)
    {
        std::cerr << "[Video] Decoder not found for stream." << std::endl;
        return false;
    }

    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx)
    {
        std::cerr << "[Video] Failed to allocate codec context." << std::endl;
        return false;
    }

    if (avcodec_parameters_to_context(codecCtx, videoStream->codecpar) < 0)
    {
        std::cerr << "[Video] Unable to copy codec parameters." << std::endl;
        return false;
    }

    if (!configureDecodeImplementation(
                                       codec,
                                       initParams.implementation,
                                       initParams.vulkanInterop,
                                       initParams.requireGraphicsQueue,
                                       initParams.debugLogging))
    {
        return false;
    }

    const int streamBitDepth = determineStreamBitDepth(videoStream, codecCtx);
    if (implementation != DecodeImplementation::Software)
    {
        AVPixelFormat sourceFormat = static_cast<AVPixelFormat>(videoStream->codecpar->format);
        requestedSwPixelFormat = pickPreferredSwPixelFormat(streamBitDepth, sourceFormat);
        codecCtx->sw_pix_fmt = requestedSwPixelFormat;

        if (initParams.debugLogging)
        {
            std::cout << "[Video] Requesting " << pixelFormatDescription(requestedSwPixelFormat)
                      << " software frames from " << implementationName
                      << " decoder (bit depth " << streamBitDepth << ")" << std::endl;
        }
    }
    else
    {
        requestedSwPixelFormat = codecCtx->pix_fmt;
    }

    // Allow FFmpeg to spin multiple worker threads for decode if possible.
    unsigned int hwThreads = std::thread::hardware_concurrency();
    codecCtx->thread_count = hwThreads > 0 ? static_cast<int>(hwThreads) : 0;
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    if (avcodec_open2(codecCtx, codec, nullptr) < 0)
    {
        std::cerr << "[Video] Failed to open codec." << std::endl;
        return false;
    }

    width = codecCtx->width;
    height = codecCtx->height;
    frame = av_frame_alloc();
    if (implementation != DecodeImplementation::Software)
    {
        swFrame = av_frame_alloc();
    }
    packet = av_packet_alloc();
    if (!frame || !packet ||
        (implementation != DecodeImplementation::Software && !swFrame))
    {
        std::cerr << "[Video] Failed to allocate FFmpeg structures." << std::endl;
        return false;
    }

    streamTimeBase = videoStream->time_base;
    fallbackPtsSeconds = 0.0;
    framesDecoded = 0;
    colorSpace = codecCtx->colorspace;
    colorRange = codecCtx->color_range;

    if (!configureFormatForPixelFormat(codecCtx->pix_fmt))
    {
        std::cerr << "[Video] Unsupported pixel format for decoder: "
                  << pixelFormatDescription(codecCtx->pix_fmt) << std::endl;
        return false;
    }

    if (initParams.debugLogging)
    {
        std::cout << "[Video] Stream pixel format: " << pixelFormatDescription(sourcePixelFormat)
                  << " -> outputFormat=" << static_cast<int>(outputFormat)
                  << " chromaDiv=" << chromaDivX << "x" << chromaDivY
                  << " bytesPerComponent=" << bytesPerComponent
                  << " bitDepth=" << bitDepth
                  << " swapUV=" << (swapChromaUV ? "yes" : "no")
                  << " colorSpace=" << colorSpaceName(colorSpace)
                  << " colorRange=" << colorRangeName(colorRange)
                  << " implementation=" << implementationName
                  << std::endl;
    }

    AVRational frameRate = av_guess_frame_rate(formatCtx, videoStream, nullptr);
    fps = (frameRate.num != 0 && frameRate.den != 0) ? av_q2d(frameRate) : 30.0;
    fps = fps > 0.0 ? fps : 30.0;

    if (initParams.debugLogging)
    {
        std::cout << "[Video] Using " << implementationName;
        if (implementation != DecodeImplementation::Software)
        {
            std::cout << " (hw " << pixelFormatDescription(hwPixelFormat) << ")";
            if (requestedSwPixelFormat != AV_PIX_FMT_NONE)
            {
                std::cout << " -> sw " << pixelFormatDescription(requestedSwPixelFormat);
            }
            std::cout << ")";
        }
        std::cout << " decoder" << std::endl;
    }

    // Calculate duration
    durationSeconds = 0.0;
    if (formatCtx && formatCtx->duration > 0)
    {
        durationSeconds = static_cast<double>(formatCtx->duration) / static_cast<double>(AV_TIME_BASE);
    }

    // Initialize video resources
    videoResources.sampler = VK_NULL_HANDLE;
    videoResources.externalLumaView = VK_NULL_HANDLE;
    videoResources.externalChromaView = VK_NULL_HANDLE;
    videoResources.usingExternal = false;

    return true;
}

VkSampler Decoder::createLinearClampSampler(Engine2D *engine)
{
    if (!engine)
        return VK_NULL_HANDLE;

    VkSampler sampler = VK_NULL_HANDLE;
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 1.0f; // Simplified
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(engine->logicalDevice, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create sampler.");
    }
    return sampler;
}

void Decoder::destroyExternalVideoViews(Engine2D *engine, VideoResources &video)
{
    if (!engine)
    {
        return;
    }
    if (video.externalLumaView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(engine->logicalDevice, video.externalLumaView, nullptr);
        video.externalLumaView = VK_NULL_HANDLE;
    }
    if (video.externalChromaView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(engine->logicalDevice, video.externalChromaView, nullptr);
        video.externalChromaView = VK_NULL_HANDLE;
    }
    video.usingExternal = false;
}

bool Decoder::seekVideoDecoder(double targetSeconds)
{
    if (!formatCtx || !codecCtx)
    {
        return false;
    }

    // Clear any stale frames before performing the seek.
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        frameQueue.clear();
    }

    // For async decoders, we need to stop the background thread, seek, then restart.
    bool wasAsync = asyncDecoding;
    if (wasAsync)
    {
        stopAsyncDecoding();
    }

    std::cout << "[Video] Seeking to " << targetSeconds << "s..." << std::endl;
    seekTargetMicroseconds.store(static_cast<int64_t>(targetSeconds * 1000000.0));

    // Convert target seconds to the stream's timebase
    AVStream *videoStream = formatCtx->streams[videoStreamIndex];
    const int64_t targetTimestamp =
        av_rescale_q(static_cast<int64_t>(targetSeconds * AV_TIME_BASE),
                     AV_TIME_BASE_Q,
                     videoStream->time_base);

    // Seek to the nearest keyframe before the target timestamp
    int ret = avformat_seek_file(formatCtx,
                                 videoStreamIndex,
                                 std::numeric_limits<int64_t>::min(),
                                 targetTimestamp,
                                 targetTimestamp,
                                 AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        std::cerr << "[Video] Failed to seek: " << ffmpegErrorString(ret) << std::endl;
        seekTargetMicroseconds.store(-1);
        // Even if seek fails, we might need to restart the async thread
        if (wasAsync)
        {
            startAsyncDecoding(maxBufferedFrames);
        }
        return false;
    }

    // Flush the decoder buffers
    avcodec_flush_buffers(codecCtx);
    avformat_flush(formatCtx);

    // Reset decoder state
    finished.store(false);
    draining = false;
    fallbackPtsSeconds = targetSeconds;
    framesDecoded = 0;

    // Restart async decoding if it was active
    if (wasAsync)
    {
        if (!startAsyncDecoding(maxBufferedFrames))
        {
            std::cerr << "[Video] Failed to restart async decoding after seek." << std::endl;
            return false;
        }
    }

    return true;
}

bool Decoder::seek(float timeSeconds)
{
    if (!formatCtx)
    {
        return false;
    }

    timeSeconds = std::clamp(timeSeconds, 0.0f, static_cast<float>(durationSeconds));
    currentTimeSeconds = timeSeconds;

    stopAsyncDecoding();
    if (!seekVideoDecoder(timeSeconds))
    {
        std::cerr << "[Decoder] Failed to seek video decoder.\n";
        return false;
    }

    if (engine)
    {
        vkDeviceWaitIdle(engine->logicalDevice);
    }

    pendingFrames.clear();
    playbackClockInitialized = false;

    return startAsyncDecoding(12);
}

bool Decoder::decodeNextFrame(DecodedFrame &decodedFrame, bool copyFrameBuffer)
{
    if (finished.load())
    {
        return false;
    }

    static int callCount = 0;
    callCount++;
    auto decodeStart = std::chrono::steady_clock::now();

    while (true)
    {
        if (!draining)
        {
            auto readStart = std::chrono::steady_clock::now();
            int readResult = av_read_frame(formatCtx, packet);
            auto readEnd = std::chrono::steady_clock::now();

            if (readResult >= 0)
            {
                if (packet->stream_index == videoStreamIndex)
                {
                    if (avcodec_send_packet(codecCtx, packet) < 0)
                    {
                        std::cerr << "[Video] Failed to send packet to decoder" << std::endl;
                    }
                }
                av_packet_unref(packet);
            }
            else
            {
                av_packet_unref(packet);
                draining = true;
                avcodec_send_packet(codecCtx, nullptr);
            }
        }

        int receiveResult = avcodec_receive_frame(codecCtx, frame);

        if (receiveResult == 0)
        {
            decodedFrame.vkSurface = {};
            const bool isHardwareVulkan =
                implementation == DecodeImplementation::Vulkan &&
                hwDeviceType == AV_HWDEVICE_TYPE_VULKAN &&
                frame->format == AV_PIX_FMT_VULKAN;

            const AVPixelFormat frameFormat = isHardwareVulkan
                                                  ? codecCtx->sw_pix_fmt
                                                  : static_cast<AVPixelFormat>(frame->format);

            const AVFrame *ptsFrame = frame;

            // Update decoder dimensions/pixel format even if we skip buffer copy
            if (width != frame->width || height != frame->height ||
                frameFormat != sourcePixelFormat)
            {
                width = frame->width;
                height = frame->height;
                if (!configureFormatForPixelFormat(frameFormat))
                {
                    std::cerr << "[Video] Unsupported pixel format during decode: "
                              << pixelFormatDescription(frameFormat) << std::endl;
                    return false;
                }

                std::cout << "[Video] Decoder output pixel format changed to "
                          << pixelFormatDescription(frameFormat) << std::endl;
            }

            // If Vulkan hardware frames are unavailable, force a CPU copy so playback still works.
            bool doCopy = copyFrameBuffer;
            if (!doCopy)
            {
                if (frame->format != AV_PIX_FMT_VULKAN ||
                    hwDeviceType != AV_HWDEVICE_TYPE_VULKAN)
                {
                    doCopy = true;
                }
            }

            if (doCopy)
            {
                AVFrame *workingFrame = frame;
                if (implementation != DecodeImplementation::Software &&
                    frame->format == hwPixelFormat)
                {
                    if (!swFrame)
                    {
                        swFrame = av_frame_alloc();
                        if (!swFrame)
                        {
                            std::cerr << "[Video] Failed to allocate sw transfer frame." << std::endl;
                            return false;
                        }
                    }
                    av_frame_unref(swFrame);
                    int transferResult = av_hwframe_transfer_data(swFrame, frame, 0);
                    if (transferResult < 0)
                    {
                        std::cerr << "[Video] Failed to transfer hardware frame: "
                                  << ffmpegErrorString(transferResult) << std::endl;
                        return false;
                    }
                    workingFrame = swFrame;
                }

                copyDecodedFrameToBuffer(workingFrame, decodedFrame.buffer);
                ptsFrame = workingFrame;
            }
            else
            {
                decodedFrame.buffer.clear();

                // Populate Vulkan surface info when we stay on the GPU path
                if (implementation == DecodeImplementation::Vulkan &&
                    hwDeviceType == AV_HWDEVICE_TYPE_VULKAN &&
                    frame->format == AV_PIX_FMT_VULKAN)
                {
                    const AVVkFrame *vkf = reinterpret_cast<AVVkFrame *>(frame->data[0]);
                    if (vkf)
                    {
                        decodedFrame.vkSurface.valid = true;
                        decodedFrame.vkSurface.width = static_cast<uint32_t>(frame->width);
                        decodedFrame.vkSurface.height = static_cast<uint32_t>(frame->height);
                        const VkFormat *vkFormats = av_vkfmt_from_pixfmt(codecCtx->sw_pix_fmt);
                        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(codecCtx->sw_pix_fmt);
                        uint32_t planes = desc ? desc->nb_components : 0;
                        planes = std::min<uint32_t>(planes, 2u);
                        decodedFrame.vkSurface.planes = planes;
                        for (uint32_t i = 0; i < planes; ++i)
                        {
                            decodedFrame.vkSurface.images[i] = vkf->img[i];
                            decodedFrame.vkSurface.layouts[i] = vkf->layout[i];
                            decodedFrame.vkSurface.semaphores[i] = vkf->sem[i];
                            decodedFrame.vkSurface.semaphoreValues[i] = vkf->sem_value[i];
                            decodedFrame.vkSurface.queueFamily[i] = vkf->queue_family[i];
                            decodedFrame.vkSurface.planeFormats[i] = vkFormats ? vkFormats[i] : VK_FORMAT_UNDEFINED;
                        }
                    }
                }
            }

            double ptsSeconds = fallbackPtsSeconds;
            const int64_t bestTimestamp = ptsFrame->best_effort_timestamp;
            if (bestTimestamp != AV_NOPTS_VALUE)
            {
                const double timeBase = streamTimeBase.den != 0
                                            ? static_cast<double>(streamTimeBase.num) /
                                                  static_cast<double>(streamTimeBase.den)
                                            : 0.0;
                ptsSeconds = timeBase * static_cast<double>(bestTimestamp);
            }
            else
            {
                const double frameDuration = fps > 0.0 ? (1.0 / fps) : (1.0 / 30.0);
                ptsSeconds = framesDecoded > 0
                                 ? fallbackPtsSeconds + frameDuration
                                 : 0.0;
            }
            fallbackPtsSeconds = ptsSeconds;
            framesDecoded++;
            decodedFrame.ptsSeconds = ptsSeconds;

            av_frame_unref(frame);

            return true;
        }
        else if (receiveResult == AVERROR(EAGAIN))
        {
            continue;
        }
        else if (receiveResult == AVERROR_EOF)
        {
            finished.store(true);
            return false;
        }
        else
        {
            std::cerr << "[Video] Decoder error: " << ffmpegErrorString(receiveResult) << std::endl;
            return false;
        }
    }
}

void Decoder::asyncDecodeLoop()
{
    std::cout << "[Video] asyncDecodeLoop started" << std::endl;

    DecodedFrame localFrame;
    localFrame.buffer.reserve(bufferSize);
    int frameCount = 0;
    bool preferZeroCopy =
        implementation == DecodeImplementation::Vulkan &&
        hwDeviceType == AV_HWDEVICE_TYPE_VULKAN;

    while (!stopRequested.load())
    {
        frameCount++;
        bool decodeSuccess = decodeNextFrame(localFrame, /*copyFrameBuffer=*/!preferZeroCopy);

        if (!decodeSuccess)
        {
            std::cout << "[Video] asyncDecodeLoop: decodeNextFrame returned false at frame "
                      << frameCount << ", finished=" << finished.load() << std::endl;
            break;
        }

        int64_t currentSeekTarget = seekTargetMicroseconds.load();
        if (currentSeekTarget >= 0)
        {
            const int64_t frameMicros = static_cast<int64_t>(localFrame.ptsSeconds * 1000000.0);
            if (frameMicros < currentSeekTarget)
            {
                // Drop frames prior to the seek target
                continue;
            }
            if (frameMicros >= currentSeekTarget)
            {
                seekTargetMicroseconds.store(-1);
                std::cout << "[Video] Seek target reached." << std::endl;
            }
        }

        if (frameCount % 30 == 0)
        {
            std::cout << "[Video] asyncDecodeLoop: decoded frame " << frameCount
                      << ", pts=" << localFrame.ptsSeconds << "s" << std::endl;
        }

        std::unique_lock<std::mutex> lock(frameMutex);
        frameCond.wait(lock, [this]()
                       { return stopRequested.load() || frameQueue.size() < maxBufferedFrames; });

        if (stopRequested.load())
        {
            std::cout << "[Video] asyncDecodeLoop: stopRequested detected, breaking loop at frame "
                      << frameCount << std::endl;
            break;
        }

        // If zero-copy was requested but the frame did not carry a Vulkan surface, disable it.
        if (preferZeroCopy && !localFrame.vkSurface.valid)
        {
            preferZeroCopy = false;
        }

        frameQueue.emplace_back(std::move(localFrame));
        lock.unlock();
        frameCond.notify_all();
        localFrame = DecodedFrame{};
        localFrame.buffer.reserve(bufferSize);
    }

    std::cout << "[Video] asyncDecodeLoop exiting after " << frameCount << " frames" << std::endl;

    threadRunning.store(false);
    frameCond.notify_all();
}

bool Decoder::startAsyncDecoding(size_t maxBufferedFrames)
{
    if (asyncDecoding)
    {
        return true;
    }

    this->maxBufferedFrames = std::max<size_t>(1, maxBufferedFrames);
    stopRequested.store(false);
    threadRunning.store(true);
    asyncDecoding = true;
    frameQueue.clear();

    try
    {
        decodeThread = std::thread(&Decoder::asyncDecodeLoop, this);
    }
    catch (const std::system_error &err)
    {
        std::cerr << "[Video] Failed to start decode thread: " << err.what() << std::endl;
        threadRunning.store(false);
        asyncDecoding = false;
        return false;
    }

    return true;
}

bool Decoder::acquireDecodedFrame(DecodedFrame &outFrame)
{
    std::unique_lock<std::mutex> lock(frameMutex);
    if (frameQueue.empty())
    {
        return false;
    }

    outFrame = std::move(frameQueue.front());
    frameQueue.pop_front();
    lock.unlock();
    frameCond.notify_all();
    return true;
}

void Decoder::stopAsyncDecoding()
{
    std::cout << "[Video] stopAsyncDecoding called" << std::endl;

    if (!asyncDecoding)
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(frameMutex);
        stopRequested.store(true);
        std::cout << "[Video] stopAsyncDecoding: set stopRequested=true, frameQueue size="
                  << frameQueue.size() << std::endl;
    }
    frameCond.notify_all();

    if (decodeThread.joinable())
    {
        std::cout << "[Video] stopAsyncDecoding: joining decode thread..." << std::endl;
        decodeThread.join();
        std::cout << "[Video] stopAsyncDecoding: decode thread joined" << std::endl;
    }

    frameQueue.clear();
    asyncDecoding = false;
    threadRunning.store(false);
    stopRequested.store(false);
}

bool Decoder::waitForVulkanFrameReady(Engine2D *engine, const DecodedFrame::VulkanSurface &surface)
{
    if (!engine || !surface.valid)
    {
        return false;
    }

    std::vector<VkSemaphore> semaphores;
    std::vector<uint64_t> values;
    for (uint32_t i = 0; i < surface.planes; ++i)
    {
        if (surface.semaphores[i] != VK_NULL_HANDLE)
        {
            semaphores.push_back(surface.semaphores[i]);
            values.push_back(surface.semaphoreValues[i]);
        }
    }

    if (semaphores.empty())
    {
        return true;
    }

    VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.semaphoreCount = static_cast<uint32_t>(semaphores.size());
    waitInfo.pSemaphores = semaphores.data();
    waitInfo.pValues = values.data();
    return vkWaitSemaphores(engine->logicalDevice, &waitInfo, UINT64_MAX) == VK_SUCCESS;
}

bool Decoder::uploadDecodedFrame(Engine2D *engine, const DecodedFrame &frame)
{
    if (!engine)
    {
        return frame.vkSurface.valid; // allow zero-copy path
    }

    // Zero-copy Vulkan path: bind decoded images directly
    if (frame.vkSurface.valid)
    {
        // Block until decode completion for this frame
        if (!waitForVulkanFrameReady(engine, frame.vkSurface))
        {
            std::cerr << "[Video2D] Failed waiting for Vulkan decode semaphores." << std::endl;
            return false;
        }

        // Destroy previous external views after ensuring GPU is idle to avoid use-after-free
        vkDeviceWaitIdle(engine->logicalDevice);
        destroyExternalVideoViews(engine, videoResources);

        auto createView = [&](VkImage image, VkFormat format) -> VkImageView
        {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY};
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            VkImageView view = VK_NULL_HANDLE;
            if (vkCreateImageView(engine->logicalDevice, &viewInfo, nullptr, &view) != VK_SUCCESS)
            {
                return VK_NULL_HANDLE;
            }
            return view;
        };

        VkImageView lumaView = VK_NULL_HANDLE;
        VkImageView chromaView = VK_NULL_HANDLE;

        if (frame.vkSurface.planes > 0 && frame.vkSurface.images[0] != VK_NULL_HANDLE)
        {
            lumaView = createView(frame.vkSurface.images[0], frame.vkSurface.planeFormats[0]);
        }
        if (frame.vkSurface.planes > 1 && frame.vkSurface.images[1] != VK_NULL_HANDLE)
        {
            chromaView = createView(frame.vkSurface.images[1], frame.vkSurface.planeFormats[1]);
        }

        if (lumaView == VK_NULL_HANDLE)
        {
            std::cerr << "[Video2D] Failed to create image view for Vulkan-decoded frame." << std::endl;
            destroyExternalVideoViews(engine, videoResources);
            return false;
        }

        videoResources.externalLumaView = lumaView;
        videoResources.externalChromaView = chromaView != VK_NULL_HANDLE ? chromaView : lumaView;
        videoResources.usingExternal = true;

        std::cout << "[Video2D] Uploaded frame " << frame.vkSurface.width << "x" << frame.vkSurface.height
                  << " external=" << (videoResources.usingExternal ? "yes" : "no") << std::endl;
        return true;
    }

    // CPU upload path
    if (videoResources.usingExternal)
    {
        vkDeviceWaitIdle(engine->logicalDevice);
        destroyExternalVideoViews(engine, videoResources);
    }

    if (frame.buffer.empty())
    {
        return false;
    }

    // Note: For the CPU upload path, you would need to implement actual Vulkan image creation
    // and data transfer here. This is simplified.
    std::cout << "[Video2D] CPU frame upload not fully implemented, size: "
              << frame.buffer.size() << " bytes" << std::endl;

    return true;
}

void Decoder::pumpDecodedFrames()
{
    constexpr size_t kMaxPendingFrames = 6;
    DecodedFrame stagingFrame;
    stagingFrame.buffer.reserve(bufferSize);

    while (pendingFrames.size() < kMaxPendingFrames &&
           acquireDecodedFrame(stagingFrame))
    {
        pendingFrames.emplace_back(std::move(stagingFrame));
        stagingFrame = DecodedFrame{};
        stagingFrame.buffer.reserve(bufferSize);
    }
}

double Decoder::advancePlayback()
{
    if (videoResources.sampler == VK_NULL_HANDLE)
    {
        return 0.0;
    }

    pumpDecodedFrames();

    if (!playing)
    {
        return lastDisplayedSeconds;
    }

    if (pendingFrames.empty())
    {
        if (finished.load() && !threadRunning.load())
        {
            // End of video reached
        }
        return lastDisplayedSeconds;
    }

    auto currentTime = std::chrono::steady_clock::now();
    auto &nextFrame = pendingFrames.front();

    if (!playbackClockInitialized)
    {
        playbackClockInitialized = true;
        // Keep the visual scrub position continuous across seeks by anchoring the base
        // to the requested playback time (lastDisplayedSeconds) rather than resetting to zero.
        basePtsSeconds = nextFrame.ptsSeconds - lastDisplayedSeconds;
        lastFramePtsSeconds = nextFrame.ptsSeconds;
        lastFrameRenderTime = currentTime;
    }

    double frameDelta = nextFrame.ptsSeconds - lastFramePtsSeconds;
    if (frameDelta < 1e-6)
    {
        frameDelta = 1.0 / std::max(30.0, fps);
    }

    auto targetTime = lastFrameRenderTime + std::chrono::duration<double>(frameDelta);
    if (currentTime + std::chrono::milliseconds(1) < targetTime)
    {
        return lastDisplayedSeconds;
    }

    auto frame = std::move(nextFrame);
    pendingFrames.pop_front();

    if (!uploadDecodedFrame(engine, frame))
    {
        std::cerr << "[Video2D] Failed to upload decoded frame." << std::endl;
    }

    lastFramePtsSeconds = frame.ptsSeconds;
    lastFrameRenderTime = currentTime;
    lastDisplayedSeconds = std::max(0.0, lastFramePtsSeconds - basePtsSeconds);
    return lastDisplayedSeconds;
}

void Decoder::cleanupVideoDecoder()
{
    if (packet)
    {
        av_packet_free(&packet);
    }
    if (frame)
    {
        av_frame_free(&frame);
    }
    if (swFrame)
    {
        av_frame_free(&swFrame);
    }
    if (hwFramesCtx)
    {
        av_buffer_unref(&hwFramesCtx);
        hwFramesCtx = nullptr;
    }
    if (hwDeviceCtx)
    {
        av_buffer_unref(&hwDeviceCtx);
        hwDeviceCtx = nullptr;
    }
    if (codecCtx)
    {
        avcodec_free_context(&codecCtx);
    }
    if (formatCtx)
    {
        avformat_close_input(&formatCtx);
    }

    // Cleanup Vulkan resources
    if (engine)
    {
        destroyExternalVideoViews(engine, videoResources);
        if (videoResources.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, videoResources.sampler, nullptr);
            videoResources.sampler = VK_NULL_HANDLE;
        }
    }
}

int Decoder::runDecodeOnlyBenchmark(const std::filesystem::path &videoPath,
                                    const std::optional<bool> &swapUvOverride,
                                    double benchmarkSeconds)
{
    const double kBenchmarkSeconds = benchmarkSeconds > 0.0 ? benchmarkSeconds : 5.0;
    if (!std::filesystem::exists(videoPath))
    {
        std::cerr << "[DecodeOnly] Missing video file: " << videoPath << std::endl;
        return 1;
    }

    DecoderInitParams params{};
    params.implementation = DecodeImplementation::Vulkan;
    params.requireGraphicsQueue = false;

    try
    {
        Decoder decoder(videoPath, params);

        DecodedFrame frame;
        frame.buffer.reserve(decoder.bufferSize);

        auto start = std::chrono::steady_clock::now();
        size_t framesDecoded = 0;
        double firstPts = -1.0;
        double decodedSeconds = 0.0;
        while (decoder.decodeNextFrame(frame, /*copyFrameBuffer=*/false))
        {
            framesDecoded++;
            if (firstPts < 0.0)
            {
                firstPts = frame.ptsSeconds;
            }

            // Stop after decoding the first kBenchmarkSeconds worth of video.
            double elapsedPts = frame.ptsSeconds - (firstPts < 0.0 ? 0.0 : firstPts);
            decodedSeconds = elapsedPts;
            if (elapsedPts >= kBenchmarkSeconds)
            {
                break;
            }

            frame.buffer.clear();
        }
        auto end = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        double fps = seconds > 0.0 ? static_cast<double>(framesDecoded) / seconds : 0.0;

        std::cout << "[DecodeOnly] Decoded "
                  << framesDecoded << " frames (~" << std::min(decodedSeconds, kBenchmarkSeconds)
                  << "s of source) in " << seconds << "s -> " << fps << " fps using "
                  << decoder.implementationName << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[DecodeOnly] Failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


static uint32_t getFrameIndex(double seconds, double fps)
{
    if (fps <= 0.0)
    {
        return 0;
    }
    double value = seconds * fps;
    if (value <= 0.0)
    {
        return 0;
    }
    double rounded = std::floor(value + 0.5);
    if (rounded >= static_cast<double>(std::numeric_limits<uint32_t>::max()))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(rounded);
}

bool saveRawFrameData(const std::filesystem::path &path, const uint8_t *data, size_t size)
{
    if (!data || size == 0)
    {
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "[motive2d] Failed to open file for writing: " << path << "\n";
        return false;
    }

    file.write(reinterpret_cast<const char *>(data), size);
    if (!file)
    {
        std::cerr << "[motive2d] Failed to write frame data to: " << path << "\n";
        return false;
    }

    std::cout << "[motive2d] Saved raw frame data (" << size << " bytes) to: " << path << "\n";
    return true;
}
