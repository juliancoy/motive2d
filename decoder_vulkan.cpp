#include "decoder_vulkan.h"

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

// Constructor
DecoderVulkan::DecoderVulkan(const std::filesystem::path &videoPath, Engine2D* engine)
    : engine(engine), playing(false)
{
    std::cout << "[Video] Loading video file: " << videoPath << std::endl;
    
    // Open the input file
    if (avformat_open_input(&formatCtx, videoPath.string().c_str(), nullptr, nullptr) < 0)
    {
        throw std::runtime_error("[Video] Failed to open file: " + videoPath.string());
    }

    formatCtx->interrupt_callback.callback = interrupt_callback;
    formatCtx->interrupt_callback.opaque = this;

    // Find stream information
    if (avformat_find_stream_info(formatCtx, nullptr) < 0)
    {
        avformat_close_input(&formatCtx);
        throw std::runtime_error("[Video] Unable to read stream info for: " + videoPath.string());
    }

    // Find the video stream
    videoStreamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoStreamIndex < 0)
    {
        avformat_close_input(&formatCtx);
        throw std::runtime_error("[Video] No video stream found in file: " + videoPath.string());
    }

    AVStream *videoStream = formatCtx->streams[videoStreamIndex];
    
    // Find the decoder
    const AVCodec *codec = avcodec_find_decoder(videoStream->codecpar->codec_id);
    if (!codec)
    {
        avformat_close_input(&formatCtx);
        throw std::runtime_error("[Video] Decoder not found for stream.");
    }

    // Allocate codec context
    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx)
    {
        avformat_close_input(&formatCtx);
        throw std::runtime_error("[Video] Failed to allocate codec context.");
    }

    // Copy codec parameters
    if (avcodec_parameters_to_context(codecCtx, videoStream->codecpar) < 0)
    {
        avcodec_free_context(&codecCtx);
        avformat_close_input(&formatCtx);
        throw std::runtime_error("[Video] Unable to copy codec parameters.");
    }
    
    hwDeviceType = AV_HWDEVICE_TYPE_VULKAN;
    hwPixelFormat = AV_PIX_FMT_VULKAN;

    // Configure Vulkan device context
    hwDeviceCtx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (!hwDeviceCtx)
    {
        hardwareInitFailureReason = "Failed to allocate Vulkan device context";
        return;
    }

    AVHWDeviceContext *deviceCtx = reinterpret_cast<AVHWDeviceContext *>(hwDeviceCtx->data);
    AVVulkanDeviceContext *vulkanCtx = static_cast<AVVulkanDeviceContext *>(deviceCtx->hwctx);

    // Initialize all fields to zero
    memset(vulkanCtx, 0, sizeof(*vulkanCtx));

    // Set only the absolutely required fields
    vulkanCtx->inst = engine->instance;
    vulkanCtx->phys_dev = engine->physicalDevice;
    vulkanCtx->act_dev = engine->logicalDevice;
    
    // Set get_proc_addr to system function
    vulkanCtx->get_proc_addr = vkGetInstanceProcAddr;
    
    // Set deprecated queue family fields for compatibility
    // queue_family_index should be graphics queue family
    vulkanCtx->queue_family_index = engine->graphicsQueueFamilyIndex;
    vulkanCtx->nb_graphics_queues = 1;
    
    // Transfer queue family (use graphics queue)
    vulkanCtx->queue_family_tx_index = engine->graphicsQueueFamilyIndex;
    vulkanCtx->nb_tx_queues = 1;
    
    // Compute queue family (use graphics queue)
    vulkanCtx->queue_family_comp_index = engine->graphicsQueueFamilyIndex;
    vulkanCtx->nb_comp_queues = 1;
    
    // Video decode queue family
    vulkanCtx->queue_family_decode_index = engine->videoQueueFamilyIndex;
    vulkanCtx->nb_decode_queues = 1;
    
    // Video encode queue family (not available, set to -1)
    vulkanCtx->queue_family_encode_index = -1;
    vulkanCtx->nb_encode_queues = 0;
    
    // Set queue families in qf array (new API)
    // Must include all queue families mentioned in deprecated fields
    int qf_index = 0;
    
    // Graphics queue family (also used for transfer and compute)
    vulkanCtx->qf[qf_index].idx = engine->graphicsQueueFamilyIndex;
    vulkanCtx->qf[qf_index].num = 1;
    vulkanCtx->qf[qf_index].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
    vulkanCtx->qf[qf_index].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    qf_index++;
    
    // Video decode queue family
    vulkanCtx->qf[qf_index].idx = engine->videoQueueFamilyIndex;
    vulkanCtx->qf[qf_index].num = 1;
    vulkanCtx->qf[qf_index].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_VIDEO_DECODE_BIT_KHR);
    vulkanCtx->qf[qf_index].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(
        VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR | VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR);
    qf_index++;
    
    // Add transfer queue family (same as graphics, but required for compatibility)
    vulkanCtx->qf[qf_index].idx = engine->graphicsQueueFamilyIndex;
    vulkanCtx->qf[qf_index].num = 1;
    vulkanCtx->qf[qf_index].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_TRANSFER_BIT);
    vulkanCtx->qf[qf_index].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    qf_index++;
    
    // Add compute queue family (same as graphics, but required for compatibility)
    vulkanCtx->qf[qf_index].idx = engine->graphicsQueueFamilyIndex;
    vulkanCtx->qf[qf_index].num = 1;
    vulkanCtx->qf[qf_index].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_COMPUTE_BIT);
    vulkanCtx->qf[qf_index].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    qf_index++;
    
    vulkanCtx->nb_qf = qf_index;

    // Set lock/unlock queue functions (NULL = use internal implementation)
    vulkanCtx->lock_queue = nullptr;
    vulkanCtx->unlock_queue = nullptr;

    // Set extension arrays
    static const char* instanceExtensions[] = {
        "VK_KHR_surface",
        "VK_KHR_xcb_surface"
    };
    
    static const char* deviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
        VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
        VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME,
        VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME,
        VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME,
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME
    };
    
    vulkanCtx->enabled_inst_extensions = instanceExtensions;
    vulkanCtx->nb_enabled_inst_extensions = 2;
    vulkanCtx->enabled_dev_extensions = deviceExtensions;
    vulkanCtx->nb_enabled_dev_extensions = 20;

    // Set minimal device features
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures(engine->physicalDevice, &features2.features);
    vulkanCtx->device_features = features2;

    int ret = av_hwdevice_ctx_init(hwDeviceCtx);
    if (ret < 0)
    {
        hardwareInitFailureReason = "Failed to initialize Vulkan device context: " + ffmpegErrorString(ret);
        av_buffer_unref(&hwDeviceCtx);
    }

    codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx);

    const int streamBitDepth = determineStreamBitDepth(videoStream, codecCtx);
    AVPixelFormat sourceFormat = static_cast<AVPixelFormat>(videoStream->codecpar->format);
    
    // Allow FFmpeg to spin multiple worker threads for decode if possible.
    unsigned int hwThreads = std::thread::hardware_concurrency();
    codecCtx->thread_count = hwThreads > 0 ? static_cast<int>(hwThreads) : 0;
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    if (avcodec_open2(codecCtx, codec, nullptr) < 0)
    {
        std::cerr << "[Video] Failed to open codec." << std::endl;
    }

    width = codecCtx->width;
    height = codecCtx->height;
    frame = av_frame_alloc();
    packet = av_packet_alloc();
    if (!frame || !packet)
    {
        std::cerr << "[Video] Failed to allocate FFmpeg structures." << std::endl;
    }

    streamTimeBase = videoStream->time_base;
    fallbackPtsSeconds = 0.0;
    framesDecoded = 0;
    colorSpace = codecCtx->colorspace;
    colorRange = codecCtx->color_range;

    if (!configureFormatForPixelFormat(codecCtx->pix_fmt))
    {
        std::cerr << "[Video] Unsupported pixel format for DecoderVulkan: "
                  << pixelFormatDescription(codecCtx->pix_fmt) << std::endl;
    }

    if (debugLogging)
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

    if (debugLogging)
    {
        std::cout << "[Video] Using " << implementationName;
        std::cout << " (hw " << pixelFormatDescription(hwPixelFormat) << ")";
        if (requestedSwPixelFormat != AV_PIX_FMT_NONE)
        {
            std::cout << " -> sw " << pixelFormatDescription(requestedSwPixelFormat);
        }
        std::cout << ")";
        std::cout << " DecoderVulkan" << std::endl;
    }

    // Calculate duration
    durationSeconds = 0.0;
    if (formatCtx && formatCtx->duration > 0)
    {
        durationSeconds = static_cast<double>(formatCtx->duration) / static_cast<double>(AV_TIME_BASE);
    }

    // Initialize video resources
    sampler = VK_NULL_HANDLE;
    externalLumaView = VK_NULL_HANDLE;
    externalChromaView = VK_NULL_HANDLE;
    usingExternal = false;

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

// Configure format for pixel format
bool DecoderVulkan::configureFormatForPixelFormat(AVPixelFormat pix_fmt)
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

// Destructor
DecoderVulkan::~DecoderVulkan()
{
    stopAsyncDecoding();
    if (packet)
    {
        av_packet_free(&packet);
    }
    if (frame)
    {
        av_frame_free(&frame);
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
        destroyExternalVideoViews();
        if (sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, sampler, nullptr);
            sampler = VK_NULL_HANDLE;
        }
    }
}

VkSampler DecoderVulkan::createLinearClampSampler(Engine2D *engine)
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

void DecoderVulkan::destroyExternalVideoViews()
{
    if (!engine)
    {
        return;
    }
    if (externalLumaView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(engine->logicalDevice, externalLumaView, nullptr);
        externalLumaView = VK_NULL_HANDLE;
    }
    if (externalChromaView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(engine->logicalDevice, externalChromaView, nullptr);
        externalChromaView = VK_NULL_HANDLE;
    }
    usingExternal = false;
}


bool DecoderVulkan::seek(float timeSeconds)
{
    if (!formatCtx)
    {
        return false;
    }

    timeSeconds = std::clamp(timeSeconds, 0.0f, static_cast<float>(durationSeconds));
    currentTimeSeconds = timeSeconds;

    stopAsyncDecoding();
    
    if (!formatCtx || !codecCtx)
    {
        return false;
    }

    // Clear any stale frames before performing the seek.
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        frameQueue.clear();
    }

    // For async DecoderVulkans, we need to stop the background thread, seek, then restart.
    bool wasAsync = asyncDecoding;
    if (wasAsync)
    {
        stopAsyncDecoding();
    }

    std::cout << "[Video] Seeking to " << timeSeconds << "s..." << std::endl;
    seekTargetMicroseconds.store(static_cast<int64_t>(timeSeconds * 1000000.0));

    // Convert target seconds to the stream's timebase
    AVStream *videoStream = formatCtx->streams[videoStreamIndex];
    const int64_t targetTimestamp =
        av_rescale_q(static_cast<int64_t>(timeSeconds * AV_TIME_BASE),
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

    // Flush the DecoderVulkan buffers
    avcodec_flush_buffers(codecCtx);
    avformat_flush(formatCtx);

    // Reset DecoderVulkan state
    finished.store(false);
    draining = false;
    fallbackPtsSeconds = timeSeconds;
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

    if (engine)
    {
        vkDeviceWaitIdle(engine->logicalDevice);
    }

    pendingFrames.clear();
    playbackClockInitialized = false;

    return startAsyncDecoding(12);
}

bool DecoderVulkan::decodeNextFrame(DecodedFrame &decodedFrame)
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
                        std::cerr << "[Video] Failed to send packet to DecoderVulkan" << std::endl;
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
            std::cout << "[Video] Received frame: width=" << frame->width << " height=" << frame->height
                      << " format=" << frame->format << " (" << pixelFormatDescription(static_cast<AVPixelFormat>(frame->format)) << ")"
                      << " data[0]=" << (void*)frame->data[0] << " linesize[0]=" << frame->linesize[0] << std::endl;

                      
            const bool isHardwareVulkan =
                hwDeviceType == AV_HWDEVICE_TYPE_VULKAN &&
                frame->format == AV_PIX_FMT_VULKAN;

            const AVPixelFormat frameFormat = static_cast<AVPixelFormat>(frame->format);

            const AVFrame *ptsFrame = frame;

            // Update DecoderVulkan dimensions/pixel format even if we skip buffer copy
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

                std::cout << "[Video] DecoderVulkan output pixel format changed to "
                          << pixelFormatDescription(frameFormat) << std::endl;
            }

            buffer.clear();

            // Populate Vulkan surface info for GPU path
            if (frame->format == AV_PIX_FMT_VULKAN)
            {
                const AVVkFrame *vkf = reinterpret_cast<AVVkFrame *>(frame->data[0]);
                if (vkf)
                {
                    vkSurface.valid = true;
                    vkSurface.width = static_cast<uint32_t>(frame->width);
                    vkSurface.height = static_cast<uint32_t>(frame->height);
                    const VkFormat *vkFormats = av_vkfmt_from_pixfmt(codecCtx->sw_pix_fmt);
                    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(codecCtx->sw_pix_fmt);
                    uint32_t planes = desc ? desc->nb_components : 0;
                    planes = std::min<uint32_t>(planes, 2u);
                    vkSurface.planes = planes;
                    for (uint32_t i = 0; i < planes; ++i)
                    {
                        vkSurface.images[i] = vkf->img[i];
                        vkSurface.layouts[i] = vkf->layout[i];
                        vkSurface.semaphores[i] = vkf->sem[i];
                        vkSurface.semaphoreValues[i] = vkf->sem_value[i];
                        vkSurface.queueFamily[i] = vkf->queue_family[i];
                        vkSurface.planeFormats[i] = vkFormats ? vkFormats[i] : VK_FORMAT_UNDEFINED;
                    }
                }
            }
            else
            {
                std::cerr << "[Video] Non-Vulkan frame format detected: " << frame->format 
                          << ". Vulkan hardware decoding required." << std::endl;
                return false;
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
            ptsSeconds = ptsSeconds;

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
            std::cerr << "[Video] DecoderVulkan error: " << ffmpegErrorString(receiveResult) << std::endl;
            return false;
        }
    }
}

void DecoderVulkan::asyncDecodeLoop()
{
    std::cout << "[Video] asyncDecodeLoop started" << std::endl;

    DecodedFrame localFrame;
    localFrame.buffer.reserve(bufferSize);
    int frameCount = 0;

    while (!stopRequested.load())
    {
        frameCount++;
        bool decodeSuccess = decodeNextFrame(localFrame);

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

        // Verify we have a valid Vulkan surface
        if (!localFrame.vkSurface.valid)
        {
            std::cerr << "[Video] Zero-copy Vulkan frame missing Vulkan surface." << std::endl;
            break;
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

bool DecoderVulkan::startAsyncDecoding(size_t maxBufferedFrames)
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
        decodeThread = std::thread(&DecoderVulkan::asyncDecodeLoop, this);
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

bool DecoderVulkan::acquireDecodedFrame(DecodedFrame &outFrame)
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

void DecoderVulkan::stopAsyncDecoding()
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

bool DecoderVulkan::waitForVulkanFrameReady(Engine2D *engine, const DecodedFrame::VulkanSurface &surface)
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

void DecoderVulkan::pumpDecodedFrames()
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

double DecoderVulkan::advancePlayback()
{
    if (sampler == VK_NULL_HANDLE)
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

        return true;
    }

    // CPU fallback is not supported
    std::cerr << "[Video2D] No valid Vulkan surface available. CPU fallback is not supported." << std::endl;
    return false;

    lastFramePtsSeconds = frame.ptsSeconds;
    lastFrameRenderTime = currentTime;
    lastDisplayedSeconds = std::max(0.0, lastFramePtsSeconds - basePtsSeconds);
    return lastDisplayedSeconds;
}

int runDecodeOnlyBenchmark(const std::filesystem::path &videoPath,
                                    double benchmarkSeconds)
{
    const double kBenchmarkSeconds = benchmarkSeconds > 0.0 ? benchmarkSeconds : 5.0;
    if (!std::filesystem::exists(videoPath))
    {
        std::cerr << "[DecodeOnly] Missing video file: " << videoPath << std::endl;
        return 1;
    }

    bool requireGraphicsQueue = false;

    try
    {
        DecoderVulkan DecoderVulkan(videoPath, requireGraphicsQueue);

        DecodedFrame frame;
        frame.buffer.reserve(DecoderVulkan.bufferSize);

        auto start = std::chrono::steady_clock::now();
        size_t framesDecoded = 0;
        double firstPts = -1.0;
        double decodedSeconds = 0.0;
        while (DecoderVulkan.decodeNextFrame(frame))
        {
            framesDecoded++;
            if (firstPts < 0.0)
            {
                firstPts = frame.ptsSeconds;
            }

            // Stop after decoding the first kBenchmarkSeconds worth of 
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
                  << "s of source) in " << seconds << "s -> " << fps << " fps" << std::endl;
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

// Interrupt callback for FFmpeg
static int interrupt_callback(void *opaque)
{
    DecoderVulkan *DecoderVulkan = static_cast<DecoderVulkan *>(opaque);
    if (DecoderVulkan && DecoderVulkan->isStopRequested())
        return 1;
    return 0;
}
