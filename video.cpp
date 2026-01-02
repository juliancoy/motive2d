#include "video.h"
#include "engine2d.h"
#include "display2d.h"
#include "utils.h"
#include "glyph.h"
#include "video_frame_utils.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <system_error>
#include <cstring>
#include <sstream>
#include <deque>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext_vulkan.h>
}

namespace {

using video::DecodeImplementation;
using video::VideoDecoder;

#define VIDEO_DEBUG_LOG(statement) \
    do { if (video::debugLoggingEnabled()) { statement; } } while (0)

std::string ffmpegErrorString(int errnum)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, buffer, sizeof(buffer));
    return std::string(buffer);
}

const char* hwDeviceTypeName(AVHWDeviceType type)
{
    switch (type)
    {
    case AV_HWDEVICE_TYPE_VAAPI:
        return "VAAPI";
    case AV_HWDEVICE_TYPE_DXVA2:
        return "DXVA2";
    case AV_HWDEVICE_TYPE_D3D11VA:
        return "D3D11";
    case AV_HWDEVICE_TYPE_CUDA:
        return "CUDA";
    case AV_HWDEVICE_TYPE_VDPAU:
        return "VDPAU";
    case AV_HWDEVICE_TYPE_VIDEOTOOLBOX:
        return "VideoToolbox";
    case AV_HWDEVICE_TYPE_QSV:
        return "QuickSync";
    case AV_HWDEVICE_TYPE_MEDIACODEC:
        return "MediaCodec";
    case AV_HWDEVICE_TYPE_VULKAN:
        return "Vulkan";
    default:
        return "Unknown";
    }
}

const char* pixelFormatName(AVPixelFormat fmt)
{
    const char* name = av_get_pix_fmt_name(fmt);
    return name ? name : "unknown";
}

int interrupt_callback(void* opaque)
{
    auto* decoder = reinterpret_cast<VideoDecoder*>(opaque);
    if (!decoder)
    {
        return 0;
    }
    return decoder->stopRequested.load() ? 1 : 0;
}

std::string pixelFormatDescription(AVPixelFormat fmt)
{
    std::ostringstream oss;
    oss << pixelFormatName(fmt) << " (" << static_cast<int>(fmt) << ")";
    return oss.str();
}

const char* colorRangeName(AVColorRange range)
{
    switch (range)
    {
    case AVCOL_RANGE_JPEG:
        return "Full";
    case AVCOL_RANGE_MPEG:
        return "Limited";
    default:
        return "Unspecified";
    }
}

const char* colorSpaceName(AVColorSpace cs)
{
    switch (cs)
    {
    case AVCOL_SPC_BT709:
        return "BT.709";
    case AVCOL_SPC_BT470BG:
        return "BT.470BG";
    case AVCOL_SPC_SMPTE170M:
        return "SMPTE170M/BT.601";
    case AVCOL_SPC_BT2020_CL:
        return "BT.2020_CL";
    case AVCOL_SPC_BT2020_NCL:
        return "BT.2020_NCL";
    default:
        return "Unspecified";
    }
}

int determineStreamBitDepth(const AVStream* videoStream, const AVCodecContext* codecCtx)
{
    if (videoStream && videoStream->codecpar && videoStream->codecpar->bits_per_raw_sample > 0)
    {
        return videoStream->codecpar->bits_per_raw_sample;
    }

    if (codecCtx && codecCtx->bits_per_raw_sample > 0)
    {
        return codecCtx->bits_per_raw_sample;
    }

    if (videoStream && videoStream->codecpar)
    {
        const AVPixFmtDescriptor* desc =
            av_pix_fmt_desc_get(static_cast<AVPixelFormat>(videoStream->codecpar->format));
        if (desc && desc->comp[0].depth > 0)
        {
            return desc->comp[0].depth;
        }
    }

    return 8;
}

AVPixelFormat pickPreferredSwPixelFormat(int bitDepth, AVPixelFormat sourceFormat)
{
    // Check if the source format is already a good software format
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(sourceFormat);
    if (desc) {
        // If source is already a software format, use it
        if (!(desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
            return sourceFormat;
        }
    }
    
    // Otherwise, choose based on bit depth
#if defined(AV_PIX_FMT_P010)
    if (bitDepth > 8)
    {
        // For 10-bit, try to match the chroma subsampling
        if (desc) {
            // Check if source is 4:2:2
            if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0) {
                // 4:2:2 - use YUV422P10LE if available
                return AV_PIX_FMT_YUV422P10LE;
            }
            // 4:2:0 or other - use P010
        }
        return AV_PIX_FMT_P010;
    }
#endif
    return AV_PIX_FMT_NV12;
}

AVPixelFormat getHardwareFormat(AVCodecContext* ctx, const AVPixelFormat* pixFmts)
{
    auto* decoder = reinterpret_cast<VideoDecoder*>(ctx->opaque);
    if (!decoder)
    {
        return pixFmts[0];
    }

    for (const AVPixelFormat* fmt = pixFmts; *fmt != AV_PIX_FMT_NONE; ++fmt)
    {
        if (*fmt == decoder->hwPixelFormat)
        {
            return *fmt;
        }
    }

    std::cerr << "[Video] Requested hardware pixel format not supported by FFmpeg." << std::endl;
    return pixFmts[0];
}

bool codecSupportsDevice(const AVCodec* codec, AVHWDeviceType type, AVPixelFormat& outFormat)
{
    for (int i = 0;; ++i)
    {
        const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
        if (!config)
        {
            break;
        }

        if ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) &&
            config->device_type == type)
        {
            outFormat = config->pix_fmt;
            return true;
        }
    }
    return false;
}

bool trySetupHardwareDecoder(VideoDecoder& decoder,
                             const AVCodec* codec,
                             AVHWDeviceType type,
                             DecodeImplementation implementation,
                             const std::optional<video::VulkanInteropContext>& vulkanInterop,
                             bool requireGraphicsQueue,
                             std::string* failureReason)
{
    VIDEO_DEBUG_LOG(std::cout << "[Video] Trying " << hwDeviceTypeName(type) << " hardware decode for codec "
                              << codec->name << "..." << std::endl);
    AVPixelFormat hwFormat = AV_PIX_FMT_NONE;
    if (!codecSupportsDevice(codec, type, hwFormat))
    {
        if (failureReason)
        {
            *failureReason = std::string("Codec lacks hardware configuration for ") + hwDeviceTypeName(type);
        }
        return false;
    }

    AVBufferRef* hwDeviceCtx = nullptr;
    if (type == AV_HWDEVICE_TYPE_VULKAN && vulkanInterop.has_value())
    {
        hwDeviceCtx = av_hwdevice_ctx_alloc(type);
        if (!hwDeviceCtx)
        {
            std::cerr << "[Video] Failed to allocate Vulkan hwdevice context." << std::endl;
            if (failureReason)
            {
                *failureReason = "Failed to allocate Vulkan hardware context.";
            }
            return false;
        }

        AVHWDeviceContext* devCtx = reinterpret_cast<AVHWDeviceContext*>(hwDeviceCtx->data);
        auto* vkctx = reinterpret_cast<AVVulkanDeviceContext*>(devCtx->hwctx);
        vkctx->get_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(vkGetInstanceProcAddr);
        vkctx->inst = vulkanInterop->instance;
        vkctx->phys_dev = vulkanInterop->physicalDevice;
        vkctx->act_dev = vulkanInterop->device;
        vkctx->device_features = VkPhysicalDeviceFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

        int err = av_hwdevice_ctx_init(hwDeviceCtx);
        if (err < 0)
        {
            const std::string errMsg = ffmpegErrorString(err);
            std::cerr << "[Video] Failed to init Vulkan hwdevice with external context: "
                      << errMsg << std::endl;
            if (failureReason)
            {
                *failureReason = std::string("Failed to initialize Vulkan hwdevice: ") + errMsg;
            }
            av_buffer_unref(&hwDeviceCtx);
            return false;
        }
    }
    else
    {
        // When no Vulkan context is provided, av_hwdevice_ctx_create handles device creation.
        // FFmpeg's default device selection logic tends to prefer queues that support graphics.
        // There is no simple option to force selection of a non-graphics (e.g., compute-only) queue.
        // The existing decode-only benchmark works because FFmpeg is able to find a suitable
        // device/queue on its own. If more fine-grained control is needed, the caller should
        // create their own Vulkan instance/device and pass it via the VulkanInteropContext.
        int err = av_hwdevice_ctx_create(&hwDeviceCtx, type, nullptr, nullptr, 0);
        if (err < 0)
        {
            const std::string errMsg = ffmpegErrorString(err);
            std::cerr << "[Video] Failed to create " << hwDeviceTypeName(type)
                      << " hardware context: " << errMsg << std::endl;
            if (failureReason)
            {
                *failureReason = std::string("Failed to create ") + hwDeviceTypeName(type)
                                 + " hardware context: " + errMsg;
            }
            return false;
        }
    }

    decoder.hwDeviceCtx = hwDeviceCtx;
    decoder.hwPixelFormat = hwFormat;
    decoder.hwDeviceType = type;
    decoder.codecCtx->opaque = &decoder;
    decoder.codecCtx->get_format = getHardwareFormat;
    decoder.codecCtx->hw_device_ctx = av_buffer_ref(decoder.hwDeviceCtx);
    if (!decoder.codecCtx->hw_device_ctx)
    {
            std::cerr << "[Video] Failed to reference hardware context." << std::endl;
            if (failureReason)
            {
                *failureReason = "Failed to reference hardware context.";
            }
            av_buffer_unref(&decoder.hwDeviceCtx);
            decoder.hwDeviceCtx = nullptr;
            return false;
        }

    decoder.implementation = implementation;
    std::string label = implementation == DecodeImplementation::Vulkan
                            ? "Vulkan"
                            : hwDeviceTypeName(type);
    decoder.implementationName = label + " hardware";
    VIDEO_DEBUG_LOG(std::cout << "[Video] " << label << " decoder reports hardware pixel format "
                              << pixelFormatDescription(hwFormat) << std::endl);
    return true;
}

bool configureDecodeImplementation(VideoDecoder& decoder,
                                   const AVCodec* codec,
                                   DecodeImplementation implementation,
                                   const std::optional<video::VulkanInteropContext>& vulkanInterop,
                                   bool requireGraphicsQueue,
                                   bool debugLogging)
{
    decoder.implementation = DecodeImplementation::Software;
    decoder.implementationName = "Software (CPU)";
    decoder.hwDeviceType = AV_HWDEVICE_TYPE_NONE;
    decoder.hwPixelFormat = AV_PIX_FMT_NONE;
    if (implementation == DecodeImplementation::Software)
    {
        return true;
    }

    auto attemptStart = std::chrono::steady_clock::now();
    std::string failureReason;
    decoder.hardwareInitFailureReason.clear();
    if (implementation == DecodeImplementation::Vulkan)
    {
        if (trySetupHardwareDecoder(decoder,
                                    codec,
                                    AV_HWDEVICE_TYPE_VULKAN,
                                    implementation,
                                    vulkanInterop,
                                    requireGraphicsQueue,
                                    &failureReason))
        {
            auto attemptDuration =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                       attemptStart);
            VIDEO_DEBUG_LOG(std::cout << "[Video] Selected decode implementation: " << decoder.implementationName << std::endl);
            if (debugLogging)
            {
                VIDEO_DEBUG_LOG(std::cout << "[Video] Hardware decode setup completed in " << attemptDuration.count()
                                        << " ms." << std::endl);
            }
            return true;
        }
        decoder.hardwareInitFailureReason = failureReason;
        auto attemptDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                   attemptStart);
        std::cerr << "[Video] Vulkan decode not available for this codec/hardware combination. "
                  << "Selected hardware attempt took " << attemptDuration.count() << " ms";
        if (!failureReason.empty())
        {
            std::cerr << " (reason: " << failureReason << ")";
        }
        std::cerr << std::endl;
        return false;
    }

    std::cerr << "[Video] Unsupported decode implementation requested." << std::endl;
    return false;
}

} // namespace

namespace video
{
bool debugLoggingEnabled()
{
    static const bool enabled = (::getenv("MOTIVE2D_DEBUG_VIDEO") != nullptr);
    return enabled;
}
} // namespace video

namespace video {

bool initializeVideoDecoder(const std::filesystem::path& videoPath,
                            VideoDecoder& decoder,
                            const DecoderInitParams& initParams)
{
    decoder.seekTargetMicroseconds.store(-1);
    if (avformat_open_input(&decoder.formatCtx, videoPath.string().c_str(), nullptr, nullptr) < 0)
    {
        std::cerr << "[Video] Failed to open file: " << videoPath << std::endl;
        return false;
    }

    decoder.formatCtx->interrupt_callback.callback = interrupt_callback;
    decoder.formatCtx->interrupt_callback.opaque = &decoder;

    if (avformat_find_stream_info(decoder.formatCtx, nullptr) < 0)
    {
        std::cerr << "[Video] Unable to read stream info for: " << videoPath << std::endl;
        return false;
    }

    decoder.videoStreamIndex = av_find_best_stream(decoder.formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (decoder.videoStreamIndex < 0)
    {
        std::cerr << "[Video] No video stream found in file: " << videoPath << std::endl;
        return false;
    }

    AVStream* videoStream = decoder.formatCtx->streams[decoder.videoStreamIndex];
    const AVCodec* codec = avcodec_find_decoder(videoStream->codecpar->codec_id);
    if (!codec)
    {
        std::cerr << "[Video] Decoder not found for stream." << std::endl;
        return false;
    }

    decoder.codecCtx = avcodec_alloc_context3(codec);
    if (!decoder.codecCtx)
    {
        std::cerr << "[Video] Failed to allocate codec context." << std::endl;
        return false;
    }

    if (avcodec_parameters_to_context(decoder.codecCtx, videoStream->codecpar) < 0)
    {
        std::cerr << "[Video] Unable to copy codec parameters." << std::endl;
        return false;
    }

    if (!configureDecodeImplementation(decoder,
                                       codec,
                                       initParams.implementation,
                                       initParams.vulkanInterop,
                                       initParams.requireGraphicsQueue,
                                       initParams.debugLogging))
    {
        return false;
    }

    const int streamBitDepth = determineStreamBitDepth(videoStream, decoder.codecCtx);
    if (decoder.implementation != DecodeImplementation::Software)
    {
        // Get the source pixel format from the stream
        AVPixelFormat sourceFormat = static_cast<AVPixelFormat>(videoStream->codecpar->format);
        decoder.requestedSwPixelFormat = pickPreferredSwPixelFormat(streamBitDepth, sourceFormat);
        decoder.codecCtx->sw_pix_fmt = decoder.requestedSwPixelFormat;
        VIDEO_DEBUG_LOG(std::cout << "[Video] Requesting " << pixelFormatDescription(decoder.requestedSwPixelFormat)
                                << " software frames from " << decoder.implementationName
                                << " decoder (bit depth " << streamBitDepth << ")" << std::endl);
    }
    else
    {
        decoder.requestedSwPixelFormat = decoder.codecCtx->pix_fmt;
    }

    // Allow FFmpeg to spin multiple worker threads for decode if possible.
    unsigned int hwThreads = std::thread::hardware_concurrency();
    decoder.codecCtx->thread_count = hwThreads > 0 ? static_cast<int>(hwThreads) : 0;
    decoder.codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    if (avcodec_open2(decoder.codecCtx, codec, nullptr) < 0)
    {
        std::cerr << "[Video] Failed to open codec." << std::endl;
        return false;
    }

    decoder.width = decoder.codecCtx->width;
    decoder.height = decoder.codecCtx->height;
    decoder.frame = av_frame_alloc();
    if (decoder.implementation != DecodeImplementation::Software)
    {
        decoder.swFrame = av_frame_alloc();
    }
    decoder.packet = av_packet_alloc();
    if (!decoder.frame || !decoder.packet ||
        (decoder.implementation != DecodeImplementation::Software && !decoder.swFrame))
    {
        std::cerr << "[Video] Failed to allocate FFmpeg structures." << std::endl;
        return false;
    }

    decoder.streamTimeBase = videoStream->time_base;
    decoder.fallbackPtsSeconds = 0.0;
    decoder.framesDecoded = 0;
    decoder.colorSpace = decoder.codecCtx->colorspace;
    decoder.colorRange = decoder.codecCtx->color_range;
    decoder.planarYuv = false;
    decoder.chromaInterleaved = false;
    decoder.outputFormat = PrimitiveYuvFormat::NV12;
    decoder.bytesPerComponent = 1;
    decoder.bitDepth = 8;
    decoder.chromaDivX = 2;
    decoder.chromaDivY = 2;
    decoder.chromaWidth = std::max<uint32_t>(1u, static_cast<uint32_t>((decoder.width + decoder.chromaDivX - 1) / decoder.chromaDivX));
    decoder.chromaHeight = std::max<uint32_t>(1u, static_cast<uint32_t>((decoder.height + decoder.chromaDivY - 1) / decoder.chromaDivY));
    decoder.yPlaneBytes = 0;
    decoder.uvPlaneBytes = 0;

    if (!configureFormatForPixelFormat(decoder, decoder.codecCtx->pix_fmt))
    {
        std::cerr << "[Video] Unsupported pixel format for decoder: "
                  << pixelFormatDescription(decoder.codecCtx->pix_fmt) << std::endl;
        return false;
    }

    VIDEO_DEBUG_LOG(std::cout << "[Video] Stream pixel format: " << pixelFormatDescription(decoder.sourcePixelFormat)
                              << " -> outputFormat=" << static_cast<int>(decoder.outputFormat)
                              << " chromaDiv=" << decoder.chromaDivX << "x" << decoder.chromaDivY
                              << " bytesPerComponent=" << decoder.bytesPerComponent
                              << " bitDepth=" << decoder.bitDepth
                              << " swapUV=" << (decoder.swapChromaUV ? "yes" : "no")
                              << " colorSpace=" << colorSpaceName(decoder.colorSpace)
                              << " colorRange=" << colorRangeName(decoder.colorRange)
                              << " implementation=" << decoder.implementationName
                              << std::endl);

    AVRational frameRate = av_guess_frame_rate(decoder.formatCtx, videoStream, nullptr);
    double fps = (frameRate.num != 0 && frameRate.den != 0) ? av_q2d(frameRate) : 30.0;
    decoder.fps = fps > 0.0 ? fps : 30.0;
    VIDEO_DEBUG_LOG(std::cout << "[Video] Using " << decoder.implementationName);
    if (decoder.implementation != DecodeImplementation::Software)
    {
        VIDEO_DEBUG_LOG(std::cout << " (hw " << pixelFormatName(decoder.hwPixelFormat));
        if (decoder.requestedSwPixelFormat != AV_PIX_FMT_NONE)
        {
            VIDEO_DEBUG_LOG(std::cout << " -> sw " << pixelFormatName(decoder.requestedSwPixelFormat));
        }
        VIDEO_DEBUG_LOG(std::cout << ")");
    }
    VIDEO_DEBUG_LOG(std::cout << " decoder" << std::endl);
    return true;
}

bool seekVideoDecoder(VideoDecoder& decoder, double targetSeconds)
{
    if (!decoder.formatCtx || !decoder.codecCtx)
    {
        return false;
    }

    // Clear any stale frames before performing the seek.
    {
        std::lock_guard<std::mutex> lock(decoder.frameMutex);
        decoder.frameQueue.clear();
    }

    // For async decoders, we need to stop the background thread, seek, then restart.
    bool wasAsync = decoder.asyncDecoding;
    if (wasAsync)
    {
        stopAsyncDecoding(decoder);
    }

    VIDEO_DEBUG_LOG(std::cout << "[Video] Seeking to " << targetSeconds << "s..." << std::endl);
    decoder.seekTargetMicroseconds.store(static_cast<int64_t>(targetSeconds * 1000000.0));

    // Convert target seconds to the stream's timebase
    AVStream* videoStream = decoder.formatCtx->streams[decoder.videoStreamIndex];
    const int64_t targetTimestamp =
        av_rescale_q(static_cast<int64_t>(targetSeconds * AV_TIME_BASE),
                     AV_TIME_BASE_Q,
                     videoStream->time_base);

    // Seek to the nearest keyframe before the target timestamp
    int ret = avformat_seek_file(decoder.formatCtx,
                                 decoder.videoStreamIndex,
                                 std::numeric_limits<int64_t>::min(),
                                 targetTimestamp,
                                 targetTimestamp,
                                 AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        std::cerr << "[Video] Failed to seek: " << ffmpegErrorString(ret) << std::endl;
        decoder.seekTargetMicroseconds.store(-1);
        // Even if seek fails, we might need to restart the async thread
        if (wasAsync)
        {
            startAsyncDecoding(decoder, decoder.maxBufferedFrames);
        }
        return false;
    }

    // Flush the decoder buffers
    avcodec_flush_buffers(decoder.codecCtx);
    avformat_flush(decoder.formatCtx);

    // Reset decoder state
    decoder.finished.store(false);
    decoder.draining = false;
    decoder.fallbackPtsSeconds = targetSeconds;
    decoder.framesDecoded = 0;

    // Restart async decoding if it was active
    if (wasAsync)
    {
        if (!startAsyncDecoding(decoder, decoder.maxBufferedFrames))
        {
            std::cerr << "[Video] Failed to restart async decoding after seek." << std::endl;
            return false;
        }
    }

    return true;
}

bool decodeNextFrame(VideoDecoder& decoder, DecodedFrame& decodedFrame, bool copyFrameBuffer)
{
    if (decoder.finished.load())
    {
    VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: decoder finished, returning false" << std::endl);
        return false;
    }

    static int callCount = 0;
    callCount++;
    auto decodeStart = std::chrono::steady_clock::now();
    
    while (true)
    {
        if (!decoder.draining)
        {
            auto readStart = std::chrono::steady_clock::now();
            int readResult = av_read_frame(decoder.formatCtx, decoder.packet);
            auto readEnd = std::chrono::steady_clock::now();
            auto readDuration = std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart);
            
            if (readDuration.count() > 100 && callCount % 10 == 0)
            {
                VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: av_read_frame took " << readDuration.count() 
                                          << " ms (call " << callCount << ")" << std::endl);
            }
            
            if (readResult >= 0)
            {
                if (decoder.packet->stream_index == decoder.videoStreamIndex)
                {
                    auto sendStart = std::chrono::steady_clock::now();
                    if (avcodec_send_packet(decoder.codecCtx, decoder.packet) < 0)
                    {
                        std::cerr << "[Video] Failed to send packet to decoder." << std::endl;
                    }
                    auto sendEnd = std::chrono::steady_clock::now();
                    auto sendDuration = std::chrono::duration_cast<std::chrono::milliseconds>(sendEnd - sendStart);
                    
                    if (sendDuration.count() > 50 && callCount % 10 == 0)
                    {
                        VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: avcodec_send_packet took " 
                                                  << sendDuration.count() << " ms" << std::endl);
                    }
                }
                av_packet_unref(decoder.packet);
            }
            else
            {
                av_packet_unref(decoder.packet);
                decoder.draining = true;
                VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: starting drain mode" << std::endl);
                avcodec_send_packet(decoder.codecCtx, nullptr);
            }
        }

        auto receiveStart = std::chrono::steady_clock::now();
        int receiveResult = avcodec_receive_frame(decoder.codecCtx, decoder.frame);
        auto receiveEnd = std::chrono::steady_clock::now();
        auto receiveDuration = std::chrono::duration_cast<std::chrono::milliseconds>(receiveEnd - receiveStart);
        
        if (receiveDuration.count() > 100 && callCount % 10 == 0)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: avcodec_receive_frame took " 
                                      << receiveDuration.count() << " ms, result=" << receiveResult 
                                      << " (call " << callCount << ")" << std::endl);
        }
        
        if (receiveResult == 0)
        {
            decodedFrame.vkSurface = {};
            const bool isHardwareVulkan =
                decoder.implementation == DecodeImplementation::Vulkan &&
                decoder.hwDeviceType == AV_HWDEVICE_TYPE_VULKAN &&
                decoder.frame->format == AV_PIX_FMT_VULKAN;

            const AVPixelFormat frameFormat = isHardwareVulkan
                                                  ? decoder.codecCtx->sw_pix_fmt
                                                  : static_cast<AVPixelFormat>(decoder.frame->format);

            const AVFrame* ptsFrame = decoder.frame;
            // Update decoder dimensions/pixel format even if we skip buffer copy
            if (decoder.width != decoder.frame->width || decoder.height != decoder.frame->height ||
                frameFormat != decoder.sourcePixelFormat)
            {
                decoder.width = decoder.frame->width;
                decoder.height = decoder.frame->height;
                if (!configureFormatForPixelFormat(decoder, frameFormat))
                {
                    std::cerr << "[Video] Unsupported pixel format during decode: "
                              << pixelFormatDescription(frameFormat) << std::endl;
                    return false;
                }

                VIDEO_DEBUG_LOG(std::cout << "[Video] Decoder output pixel format changed to "
                                          << pixelFormatDescription(frameFormat) << std::endl);
            }

            // If Vulkan hardware frames are unavailable, force a CPU copy so playback still works.
            bool doCopy = copyFrameBuffer;
            if (!doCopy)
            {
                if (decoder.frame->format != AV_PIX_FMT_VULKAN ||
                    decoder.hwDeviceType != AV_HWDEVICE_TYPE_VULKAN)
                {
                    doCopy = true;
                }
            }

            if (doCopy)
            {
                AVFrame* workingFrame = decoder.frame;
                if (decoder.implementation != DecodeImplementation::Software &&
                    decoder.frame->format == decoder.hwPixelFormat)
                {
                    if (!decoder.swFrame)
                    {
                        decoder.swFrame = av_frame_alloc();
                        if (!decoder.swFrame)
                        {
                            std::cerr << "[Video] Failed to allocate sw transfer frame." << std::endl;
                            return false;
                        }
                    }
                    av_frame_unref(decoder.swFrame);
                    int transferResult = av_hwframe_transfer_data(decoder.swFrame, decoder.frame, 0);
                    if (transferResult < 0)
                    {
                        std::cerr << "[Video] Failed to transfer hardware frame: "
                                  << ffmpegErrorString(transferResult) << std::endl;
                        return false;
                    }
                    workingFrame = decoder.swFrame;
                }

                copyDecodedFrameToBuffer(decoder, workingFrame, decodedFrame.buffer);
                ptsFrame = workingFrame;
            }
            else
            {
                decodedFrame.buffer.clear();

                // Populate Vulkan surface info when we stay on the GPU path
                if (decoder.implementation == DecodeImplementation::Vulkan &&
                    decoder.hwDeviceType == AV_HWDEVICE_TYPE_VULKAN &&
                    decoder.frame->format == AV_PIX_FMT_VULKAN)
                {
                    const AVVkFrame* vkf = reinterpret_cast<AVVkFrame*>(decoder.frame->data[0]);
                    if (vkf)
                    {
                        decodedFrame.vkSurface.valid = true;
                        decodedFrame.vkSurface.width = static_cast<uint32_t>(decoder.frame->width);
                        decodedFrame.vkSurface.height = static_cast<uint32_t>(decoder.frame->height);
                        const VkFormat* vkFormats = av_vkfmt_from_pixfmt(decoder.codecCtx->sw_pix_fmt);
                        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(decoder.codecCtx->sw_pix_fmt);
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

            double ptsSeconds = decoder.fallbackPtsSeconds;
            const int64_t bestTimestamp = ptsFrame->best_effort_timestamp;
            if (bestTimestamp != AV_NOPTS_VALUE)
            {
                const double timeBase = decoder.streamTimeBase.den != 0
                                            ? static_cast<double>(decoder.streamTimeBase.num) /
                                                  static_cast<double>(decoder.streamTimeBase.den)
                                            : 0.0;
                ptsSeconds = timeBase * static_cast<double>(bestTimestamp);
            }
            else
            {
                const double frameDuration = decoder.fps > 0.0 ? (1.0 / decoder.fps) : (1.0 / 30.0);
                ptsSeconds = decoder.framesDecoded > 0
                                 ? decoder.fallbackPtsSeconds + frameDuration
                                 : 0.0;
            }
            decoder.fallbackPtsSeconds = ptsSeconds;
            decoder.framesDecoded++;
            decodedFrame.ptsSeconds = ptsSeconds;

            av_frame_unref(decoder.frame);
            
            auto decodeEnd = std::chrono::steady_clock::now();
            auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(decodeEnd - decodeStart);
            if (totalDuration.count() > 200 && callCount % 5 == 0)
            {
                VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: successfully decoded frame " << callCount 
                                          << " in " << totalDuration.count() << " ms total" << std::endl);
            }
            
            return true;
        }
        else if (receiveResult == AVERROR(EAGAIN))
        {
            continue;
        }
        else if (receiveResult == AVERROR_EOF)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: EOF reached after " << callCount << " calls" << std::endl);
            decoder.finished.store(true);
            return false;
        }
        else
        {
            std::cerr << "[Video] Decoder error: " << receiveResult << std::endl;
            return false;
        }
    }
}

namespace
{
void asyncDecodeLoop(VideoDecoder* decoder)
{
    VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop started, stopRequested=" << decoder->stopRequested.load() 
                              << ", bufferSize=" << decoder->bufferSize << std::endl);
    
    video::DecodedFrame localFrame;
    localFrame.buffer.reserve(decoder->bufferSize);
    int frameCount = 0;
    auto loopStartTime = std::chrono::steady_clock::now();
    bool preferZeroCopy =
        decoder->implementation == DecodeImplementation::Vulkan &&
        decoder->hwDeviceType == AV_HWDEVICE_TYPE_VULKAN;
    
    while (!decoder->stopRequested.load())
    {
        frameCount++;
        auto decodeStart = std::chrono::steady_clock::now();
        bool decodeSuccess = video::decodeNextFrame(*decoder, localFrame, /*copyFrameBuffer=*/!preferZeroCopy);
        auto decodeEnd = std::chrono::steady_clock::now();
        auto decodeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(decodeEnd - decodeStart);
        
        if (!decodeSuccess)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop: decodeNextFrame returned false at frame " 
                                      << frameCount << ", finished=" << decoder->finished.load() 
                                      << ", decode took " << decodeDuration.count() << " ms" << std::endl);
            break;
        }

        int64_t currentSeekTarget = decoder->seekTargetMicroseconds.load();
        if (currentSeekTarget >= 0)
            {
                VIDEO_DEBUG_LOG(std::cout << "[Video] Post-seek decode: PTS " << localFrame.ptsSeconds << "s (target " << (double)currentSeekTarget / 1000000.0 << "s)" << std::endl);
            const int64_t frameMicros = static_cast<int64_t>(localFrame.ptsSeconds * 1000000.0);
            if (frameMicros < currentSeekTarget)
            {
                // Drop frames prior to the seek target so we don't stall filling the queue
                // with early data.
                continue;
            }
            if (frameMicros >= currentSeekTarget)
            {
                decoder->seekTargetMicroseconds.store(-1);
                VIDEO_DEBUG_LOG(std::cout << "[Video] Seek target reached." << std::endl);
            }
        }
        
        if (frameCount % 30 == 0)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop: decoded frame " << frameCount 
                                      << ", pts=" << localFrame.ptsSeconds 
                                      << "s, decode took " << decodeDuration.count() << " ms" << std::endl);
        }

        std::unique_lock<std::mutex> lock(decoder->frameMutex);
        auto waitStart = std::chrono::steady_clock::now();
        decoder->frameCond.wait(lock, [decoder]() {
            return decoder->stopRequested.load() || decoder->frameQueue.size() < decoder->maxBufferedFrames;
        });
        auto waitEnd = std::chrono::steady_clock::now();
        auto waitDuration = std::chrono::duration_cast<std::chrono::milliseconds>(waitEnd - waitStart);
        
        if (waitDuration.count() > 10 && frameCount % 10 == 0 && false)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop: waited " << waitDuration.count() 
                                      << " ms for frame queue, size=" << decoder->frameQueue.size() 
                                      << "/" << decoder->maxBufferedFrames << std::endl);
        }

        if (decoder->stopRequested.load())
            {
                VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop: stopRequested detected, breaking loop at frame " 
                                          << frameCount << std::endl);
                break;
            }

        // If zero-copy was requested but the frame did not carry a Vulkan surface, disable it.
        if (preferZeroCopy && !localFrame.vkSurface.valid)
        {
            preferZeroCopy = false;
        }

        decoder->frameQueue.emplace_back(std::move(localFrame));
        lock.unlock();
        decoder->frameCond.notify_all();
        localFrame = video::DecodedFrame{};
        localFrame.buffer.reserve(decoder->bufferSize);
    }

    auto loopEndTime = std::chrono::steady_clock::now();
    auto loopDuration = std::chrono::duration_cast<std::chrono::milliseconds>(loopEndTime - loopStartTime);
    
    VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop exiting after " << frameCount << " frames, " 
                              << loopDuration.count() << " ms, stopRequested=" << decoder->stopRequested.load() 
                              << ", threadRunning=" << decoder->threadRunning.load() << std::endl);
    
    decoder->threadRunning.store(false);
    decoder->frameCond.notify_all();
}
} // namespace

bool startAsyncDecoding(VideoDecoder& decoder, size_t maxBufferedFrames)
{
    if (decoder.asyncDecoding)
    {
        return true;
    }

    decoder.maxBufferedFrames = std::max<size_t>(1, maxBufferedFrames);
    decoder.stopRequested.store(false);
    decoder.threadRunning.store(true);
    decoder.asyncDecoding = true;
    decoder.frameQueue.clear();

    try
    {
        decoder.decodeThread = std::thread(asyncDecodeLoop, &decoder);
    }
    catch (const std::system_error& err)
    {
        std::cerr << "[Video] Failed to start decode thread: " << err.what() << std::endl;
        decoder.threadRunning.store(false);
        decoder.asyncDecoding = false;
        return false;
    }

    return true;
}

bool acquireDecodedFrame(VideoDecoder& decoder, DecodedFrame& outFrame)
{
    std::unique_lock<std::mutex> lock(decoder.frameMutex);
    if (decoder.frameQueue.empty())
    {
        return false;
    }

    outFrame = std::move(decoder.frameQueue.front());
    decoder.frameQueue.pop_front();
    lock.unlock();
    decoder.frameCond.notify_all();
    return true;
}

void stopAsyncDecoding(VideoDecoder& decoder)
{
    VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding called, asyncDecoding=" << decoder.asyncDecoding 
                              << ", threadRunning=" << decoder.threadRunning.load() 
                              << ", stopRequested=" << decoder.stopRequested.load() << std::endl);
    
    if (!decoder.asyncDecoding)
    {
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: not async decoding, returning early" << std::endl);
        return;
    }

    auto startTime = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(decoder.frameMutex);
        decoder.stopRequested.store(true);
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: set stopRequested=true, frameQueue size=" 
                              << decoder.frameQueue.size() << std::endl);
    }
    decoder.frameCond.notify_all();
    VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: notified condition variable" << std::endl);

    if (decoder.decodeThread.joinable())
    {
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: joining decode thread..." << std::endl);
        auto joinStart = std::chrono::steady_clock::now();
        decoder.decodeThread.join();
        auto joinEnd = std::chrono::steady_clock::now();
        auto joinDuration = std::chrono::duration_cast<std::chrono::milliseconds>(joinEnd - joinStart);
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: decode thread joined after " 
                                  << joinDuration.count() << " ms" << std::endl);
    }
    else
    {
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: decode thread not joinable" << std::endl);
    }

    decoder.frameQueue.clear();
    decoder.asyncDecoding = false;
    decoder.threadRunning.store(false);
    decoder.stopRequested.store(false);
    
    auto endTime = std::chrono::steady_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding completed in " << totalDuration.count() << " ms" << std::endl);
}

void cleanupVideoDecoder(VideoDecoder& decoder)
{
    stopAsyncDecoding(decoder);
    if (decoder.packet)
    {
        av_packet_free(&decoder.packet);
    }
    if (decoder.frame)
    {
        av_frame_free(&decoder.frame);
    }
    if (decoder.swFrame)
    {
        av_frame_free(&decoder.swFrame);
    }
    if (decoder.hwFramesCtx)
    {
        av_buffer_unref(&decoder.hwFramesCtx);
        decoder.hwFramesCtx = nullptr;
    }
    if (decoder.hwDeviceCtx)
    {
        av_buffer_unref(&decoder.hwDeviceCtx);
        decoder.hwDeviceCtx = nullptr;
    }
    if (decoder.codecCtx)
    {
        avcodec_free_context(&decoder.codecCtx);
    }
    if (decoder.formatCtx)
    {
        avformat_close_input(&decoder.formatCtx);
    }
}

namespace
{
video::VideoColorSpace mapColorSpace(const VideoDecoder& decoder)
{
    switch (decoder.colorSpace)
    {
    case AVCOL_SPC_BT709:
        return video::VideoColorSpace::BT709;
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
        return video::VideoColorSpace::BT2020;
    case AVCOL_SPC_SMPTE170M:
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_FCC:
        return video::VideoColorSpace::BT601;
    default:
        return decoder.height >= 720 ? video::VideoColorSpace::BT709 : video::VideoColorSpace::BT601;
    }
}

video::VideoColorRange mapColorRange(const VideoDecoder& decoder)
{
    if (decoder.colorRange == AVCOL_RANGE_JPEG)
    {
        return video::VideoColorRange::Full;
    }
    return video::VideoColorRange::Limited;
}
} // namespace

VideoColorInfo deriveVideoColorInfo(const VideoDecoder& decoder)
{
    VideoColorInfo info{};
    info.colorSpace = mapColorSpace(decoder);
    info.colorRange = mapColorRange(decoder);
    return info;
}

namespace
{
struct YuvCoefficients
{
    float yR;
    float yG;
    float yB;
    float uR;
    float uG;
    float uB;
    float vR;
    float vG;
    float vB;
};

YuvCoefficients getCoefficients(VideoColorSpace space)
{
    switch (space)
    {
    case VideoColorSpace::BT601:
        return {0.299f, 0.587f, 0.114f,
                -0.168736f, -0.331264f, 0.5f,
                0.5f, -0.418688f, -0.081312f};
    case VideoColorSpace::BT2020:
        return {0.2627f, 0.6780f, 0.0593f,
                -0.13963f, -0.36037f, 0.5f,
                0.5f, -0.459786f, -0.040214f};
    case VideoColorSpace::BT709:
    default:
        return {0.2126f, 0.7152f, 0.0722f,
                -0.114572f, -0.385428f, 0.5f,
                0.5f, -0.454153f, -0.045847f};
    }
}

inline uint8_t convertLumaByte(float value, VideoColorRange range)
{
    value = std::clamp(value, 0.0f, 1.0f);
    if (range == VideoColorRange::Full)
    {
        return static_cast<uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
    }
    return static_cast<uint8_t>(std::clamp(value * 219.0f + 16.0f, 0.0f, 255.0f));
}

inline uint8_t convertChromaByte(float value, VideoColorRange range)
{
    if (range == VideoColorRange::Full)
    {
        return static_cast<uint8_t>(std::clamp((value + 0.5f) * 255.0f, 0.0f, 255.0f));
    }
    return static_cast<uint8_t>(std::clamp(value * 224.0f + 128.0f, 0.0f, 255.0f));
}
} // namespace

Nv12Overlay convertOverlayToNv12(const glyph::OverlayBitmap& bitmap, const VideoColorInfo& colorInfo)
{
    Nv12Overlay overlay;
    overlay.width = bitmap.width;
    overlay.height = bitmap.height;
    overlay.offsetX = bitmap.offsetX;
    overlay.offsetY = bitmap.offsetY;

    if (overlay.width == 0 || overlay.height == 0 || bitmap.pixels.empty())
    {
        return overlay;
    }

    overlay.uvWidth = (overlay.width + 1) / 2;
    overlay.uvHeight = (overlay.height + 1) / 2;
    const size_t overlayPixelCount = static_cast<size_t>(overlay.width) * overlay.height;
    overlay.yPlane.resize(overlayPixelCount, 0);
    overlay.yAlpha.resize(overlayPixelCount, 0);
    overlay.uvPlane.resize(static_cast<size_t>(overlay.uvWidth) * overlay.uvHeight * 2, 128);
    overlay.uvAlpha.resize(static_cast<size_t>(overlay.uvWidth) * overlay.uvHeight, 0);

    const YuvCoefficients coeffs = getCoefficients(colorInfo.colorSpace);
    std::vector<float> uvAccumU(overlay.uvWidth * overlay.uvHeight, 0.0f);
    std::vector<float> uvAccumV(overlay.uvWidth * overlay.uvHeight, 0.0f);
    std::vector<float> uvAccumAlpha(overlay.uvWidth * overlay.uvHeight, 0.0f);

    for (uint32_t y = 0; y < overlay.height; ++y)
    {
        for (uint32_t x = 0; x < overlay.width; ++x)
        {
            const size_t pixelIndex = static_cast<size_t>(y) * overlay.width + x;
            const size_t rgbaIndex = pixelIndex * 4;
            const uint8_t alphaByte = bitmap.pixels[rgbaIndex + 3];
            overlay.yAlpha[pixelIndex] = alphaByte;
            if (alphaByte == 0)
            {
                continue;
            }

            const float alpha = static_cast<float>(alphaByte) / 255.0f;
            const float r = static_cast<float>(bitmap.pixels[rgbaIndex + 0]) / 255.0f;
            const float g = static_cast<float>(bitmap.pixels[rgbaIndex + 1]) / 255.0f;
            const float b = static_cast<float>(bitmap.pixels[rgbaIndex + 2]) / 255.0f;

            const float yComponent = coeffs.yR * r + coeffs.yG * g + coeffs.yB * b;
            const float uComponent = coeffs.uR * r + coeffs.uG * g + coeffs.uB * b;
            const float vComponent = coeffs.vR * r + coeffs.vG * g + coeffs.vB * b;

            overlay.yPlane[pixelIndex] = convertLumaByte(yComponent, colorInfo.colorRange);

            const uint8_t uByte = convertChromaByte(uComponent, colorInfo.colorRange);
            const uint8_t vByte = convertChromaByte(vComponent, colorInfo.colorRange);

            const uint32_t blockX = x / 2;
            const uint32_t blockY = y / 2;
            const size_t blockIndex = static_cast<size_t>(blockY) * overlay.uvWidth + blockX;
            uvAccumU[blockIndex] += alpha * static_cast<float>(uByte);
            uvAccumV[blockIndex] += alpha * static_cast<float>(vByte);
            uvAccumAlpha[blockIndex] += alpha;
        }
    }

    for (size_t blockIndex = 0; blockIndex < uvAccumAlpha.size(); ++blockIndex)
    {
        const float accumulatedAlpha = std::min(1.0f, uvAccumAlpha[blockIndex]);
        overlay.uvAlpha[blockIndex] = static_cast<uint8_t>(std::clamp(accumulatedAlpha * 255.0f, 0.0f, 255.0f));
        if (uvAccumAlpha[blockIndex] > 0.0f)
        {
            const float invAlpha = 1.0f / uvAccumAlpha[blockIndex];
            const uint8_t uByte = static_cast<uint8_t>(std::clamp(uvAccumU[blockIndex] * invAlpha, 0.0f, 255.0f));
            const uint8_t vByte = static_cast<uint8_t>(std::clamp(uvAccumV[blockIndex] * invAlpha, 0.0f, 255.0f));
            overlay.uvPlane[blockIndex * 2] = uByte;
            overlay.uvPlane[blockIndex * 2 + 1] = vByte;
        }
        else
        {
            overlay.uvPlane[blockIndex * 2] = 128;
            overlay.uvPlane[blockIndex * 2 + 1] = 128;
        }
    }

    return overlay;
}

void applyNv12Overlay(std::vector<uint8_t>& nv12Buffer,
                      uint32_t frameWidth,
                      uint32_t frameHeight,
                      const Nv12Overlay& overlay)
{
    if (!overlay.isValid() || nv12Buffer.empty() || frameWidth == 0 || frameHeight == 0)
    {
        return;
    }

    const size_t lumaSize = static_cast<size_t>(frameWidth) * frameHeight;
    if (nv12Buffer.size() < lumaSize + lumaSize / 2)
    {
        return;
    }

    uint8_t* yPlane = nv12Buffer.data();
    uint8_t* uvPlane = nv12Buffer.data() + lumaSize;

    for (uint32_t y = 0; y < overlay.height; ++y)
    {
        const uint32_t frameY = overlay.offsetY + y;
        if (frameY >= frameHeight)
        {
            break;
        }

        for (uint32_t x = 0; x < overlay.width; ++x)
        {
            const uint32_t frameX = overlay.offsetX + x;
            if (frameX >= frameWidth)
            {
                break;
            }

            const size_t overlayIndex = static_cast<size_t>(y) * overlay.width + x;
            const uint8_t alphaByte = overlay.yAlpha[overlayIndex];
            if (alphaByte == 0)
            {
                continue;
            }

            const float alpha = static_cast<float>(alphaByte) / 255.0f;
            const size_t dstIndex = static_cast<size_t>(frameY) * frameWidth + frameX;
            const uint8_t overlayY = overlay.yPlane[overlayIndex];
            const uint8_t baseY = yPlane[dstIndex];
            yPlane[dstIndex] = static_cast<uint8_t>(alpha * overlayY + (1.0f - alpha) * baseY + 0.5f);
        }
    }

    const uint32_t frameUvWidth = frameWidth / 2;
    const uint32_t frameUvHeight = frameHeight / 2;
    if (frameUvWidth == 0 || frameUvHeight == 0)
    {
        return;
    }

    for (uint32_t by = 0; by < overlay.uvHeight; ++by)
    {
        const uint32_t frameBlockY = (overlay.offsetY / 2) + by;
        if (frameBlockY >= frameUvHeight)
        {
            break;
        }

        for (uint32_t bx = 0; bx < overlay.uvWidth; ++bx)
        {
            const uint32_t frameBlockX = (overlay.offsetX / 2) + bx;
            if (frameBlockX >= frameUvWidth)
            {
                break;
            }

            const size_t blockIndex = static_cast<size_t>(by) * overlay.uvWidth + bx;
            const uint8_t alphaByte = overlay.uvAlpha[blockIndex];
            if (alphaByte == 0)
            {
                continue;
            }

            const float alpha = static_cast<float>(alphaByte) / 255.0f;
            const size_t dstIndex = static_cast<size_t>(frameBlockY) * frameWidth + frameBlockX * 2;
            const uint8_t overlayU = overlay.uvPlane[blockIndex * 2];
            const uint8_t overlayV = overlay.uvPlane[blockIndex * 2 + 1];
            const uint8_t baseU = uvPlane[dstIndex];
            const uint8_t baseV = uvPlane[dstIndex + 1];
            uvPlane[dstIndex] = static_cast<uint8_t>(alpha * overlayU + (1.0f - alpha) * baseU + 0.5f);
            uvPlane[dstIndex + 1] = static_cast<uint8_t>(alpha * overlayV + (1.0f - alpha) * baseV + 0.5f);
        }
    }
}

namespace {

uint32_t computeBitDepthScale(uint32_t bitDepth)
{
    if (bitDepth <= 8)
    {
        return 1u;
    }
    const uint32_t shift = bitDepth - 8;
    if (shift >= 24)
    {
        return 1u << 24;
    }
    return 1u << shift;
}

uint8_t blendComponent8(uint8_t base, uint8_t overlay, float alpha)
{
    const float blended = alpha * static_cast<float>(overlay) +
                          (1.0f - alpha) * static_cast<float>(base);
    return static_cast<uint8_t>(std::clamp(blended + 0.5f, 0.0f, 255.0f));
}

uint16_t blendComponent16(uint16_t base,
                          uint16_t overlay,
                          float alpha,
                          uint16_t maxCode)
{
    const float blended = alpha * static_cast<float>(overlay) +
                          (1.0f - alpha) * static_cast<float>(base);
    return static_cast<uint16_t>(std::clamp(blended + 0.5f,
                                             0.0f,
                                             static_cast<float>(maxCode)));
}

void applyPlanarOverlay8(std::vector<uint8_t>& buffer,
                         const VideoDecoder& decoder,
                         const Nv12Overlay& overlay)
{
    if (decoder.width == 0 || decoder.height == 0)
    {
        return;
    }

    uint8_t* yPlane = buffer.data();
    for (uint32_t y = 0; y < overlay.height; ++y)
    {
        const uint32_t frameY = overlay.offsetY + y;
        if (frameY >= static_cast<uint32_t>(decoder.height))
        {
            break;
        }

        uint8_t* dstRow = yPlane + static_cast<size_t>(frameY) * decoder.width;
        for (uint32_t x = 0; x < overlay.width; ++x)
        {
            const uint32_t frameX = overlay.offsetX + x;
            if (frameX >= static_cast<uint32_t>(decoder.width))
            {
                break;
            }

            const size_t overlayIndex = static_cast<size_t>(y) * overlay.width + x;
            const uint8_t alphaByte = overlay.yAlpha[overlayIndex];
            if (alphaByte == 0)
            {
                continue;
            }

            const float alpha = static_cast<float>(alphaByte) / 255.0f;
            const uint8_t overlayValue = overlay.yPlane[overlayIndex];
            uint8_t& dstValue = dstRow[frameX];
            dstValue = blendComponent8(dstValue, overlayValue, alpha);
        }
    }

    if (decoder.chromaWidth == 0 || decoder.chromaHeight == 0)
    {
        return;
    }

    uint8_t* uvPlane = buffer.data() + decoder.yPlaneBytes;
    const size_t uvStride = static_cast<size_t>(decoder.chromaWidth) * 2u;
    const uint32_t chromaOffsetX = decoder.chromaDivX > 0 ? overlay.offsetX / decoder.chromaDivX : overlay.offsetX;
    const uint32_t chromaOffsetY = decoder.chromaDivY > 0 ? overlay.offsetY / decoder.chromaDivY : overlay.offsetY;

    for (uint32_t by = 0; by < overlay.uvHeight; ++by)
    {
        const uint32_t frameBlockY = chromaOffsetY + by;
        if (frameBlockY >= decoder.chromaHeight)
        {
            break;
        }

        uint8_t* dstRow = uvPlane + static_cast<size_t>(frameBlockY) * uvStride;
        for (uint32_t bx = 0; bx < overlay.uvWidth; ++bx)
        {
            const uint32_t frameBlockX = chromaOffsetX + bx;
            if (frameBlockX >= decoder.chromaWidth)
            {
                break;
            }

            const size_t blockIndex = static_cast<size_t>(by) * overlay.uvWidth + bx;
            const uint8_t alphaByte = overlay.uvAlpha[blockIndex];
            if (alphaByte == 0)
            {
                continue;
            }

            const float alpha = static_cast<float>(alphaByte) / 255.0f;
            uint8_t* dst = dstRow + static_cast<size_t>(frameBlockX) * 2u;
            const uint8_t overlayU = overlay.uvPlane[blockIndex * 2u];
            const uint8_t overlayV = overlay.uvPlane[blockIndex * 2u + 1u];
            dst[0] = blendComponent8(dst[0], overlayU, alpha);
            dst[1] = blendComponent8(dst[1], overlayV, alpha);
        }
    }
}

void applyPlanarOverlay16(std::vector<uint8_t>& buffer,
                          const VideoDecoder& decoder,
                          const Nv12Overlay& overlay)
{
    if (decoder.width == 0 || decoder.height == 0)
    {
        return;
    }

    uint16_t* yPlane = reinterpret_cast<uint16_t*>(buffer.data());
    const uint32_t maxCode = decoder.bitDepth > 0 ? ((1u << decoder.bitDepth) - 1u) : 0xffffu;
    const uint32_t scaleFactor = computeBitDepthScale(decoder.bitDepth);

    for (uint32_t y = 0; y < overlay.height; ++y)
    {
        const uint32_t frameY = overlay.offsetY + y;
        if (frameY >= static_cast<uint32_t>(decoder.height))
        {
            break;
        }

        uint16_t* dstRow = yPlane + static_cast<size_t>(frameY) * decoder.width;
        for (uint32_t x = 0; x < overlay.width; ++x)
        {
            const uint32_t frameX = overlay.offsetX + x;
            if (frameX >= static_cast<uint32_t>(decoder.width))
            {
                break;
            }

            const size_t overlayIndex = static_cast<size_t>(y) * overlay.width + x;
            const uint8_t alphaByte = overlay.yAlpha[overlayIndex];
            if (alphaByte == 0)
            {
                continue;
            }

            const float alpha = static_cast<float>(alphaByte) / 255.0f;
            const uint32_t overlayValue = std::min<uint32_t>(
                static_cast<uint32_t>(overlay.yPlane[overlayIndex]) * scaleFactor,
                maxCode);
            uint16_t& dstValue = dstRow[frameX];
            dstValue = blendComponent16(dstValue,
                                        static_cast<uint16_t>(overlayValue),
                                        alpha,
                                        static_cast<uint16_t>(maxCode));
        }
    }

    if (decoder.chromaWidth == 0 || decoder.chromaHeight == 0)
    {
        return;
    }

    uint16_t* uvPlane = reinterpret_cast<uint16_t*>(buffer.data() + decoder.yPlaneBytes);
    const size_t uvStride = static_cast<size_t>(decoder.chromaWidth) * 2u;
    const uint32_t chromaOffsetX = decoder.chromaDivX > 0 ? overlay.offsetX / decoder.chromaDivX : overlay.offsetX;
    const uint32_t chromaOffsetY = decoder.chromaDivY > 0 ? overlay.offsetY / decoder.chromaDivY : overlay.offsetY;

    for (uint32_t by = 0; by < overlay.uvHeight; ++by)
    {
        const uint32_t frameBlockY = chromaOffsetY + by;
        if (frameBlockY >= decoder.chromaHeight)
        {
            break;
        }

        uint16_t* dstRow = uvPlane + static_cast<size_t>(frameBlockY) * uvStride;
        for (uint32_t bx = 0; bx < overlay.uvWidth; ++bx)
        {
            const uint32_t frameBlockX = chromaOffsetX + bx;
            if (frameBlockX >= decoder.chromaWidth)
            {
                break;
            }

            const size_t blockIndex = static_cast<size_t>(by) * overlay.uvWidth + bx;
            const uint8_t alphaByte = overlay.uvAlpha[blockIndex];
            if (alphaByte == 0)
            {
                continue;
            }

            const float alpha = static_cast<float>(alphaByte) / 255.0f;
            uint16_t* dst = dstRow + static_cast<size_t>(frameBlockX) * 2u;
            const uint32_t overlayU = std::min<uint32_t>(
                static_cast<uint32_t>(overlay.uvPlane[blockIndex * 2u]) * scaleFactor,
                maxCode);
            const uint32_t overlayV = std::min<uint32_t>(
                static_cast<uint32_t>(overlay.uvPlane[blockIndex * 2u + 1u]) * scaleFactor,
                maxCode);
            dst[0] = blendComponent16(dst[0], static_cast<uint16_t>(overlayU), alpha, static_cast<uint16_t>(maxCode));
            dst[1] = blendComponent16(dst[1], static_cast<uint16_t>(overlayV), alpha, static_cast<uint16_t>(maxCode));
        }
    }
}

} // namespace

void applyOverlayToDecodedFrame(std::vector<uint8_t>& buffer,
                                const VideoDecoder& decoder,
                                const Nv12Overlay& overlay)
{
    if (!overlay.isValid() || buffer.empty())
    {
        return;
    }

    if (decoder.outputFormat == PrimitiveYuvFormat::NV12)
    {
        applyNv12Overlay(buffer,
                         static_cast<uint32_t>(decoder.width),
                         static_cast<uint32_t>(decoder.height),
                         overlay);
        return;
    }

    if (!decoder.planarYuv)
    {
        return;
    }

    if (decoder.bytesPerComponent <= 1)
    {
        applyPlanarOverlay8(buffer, decoder, overlay);
    }
    else
    {
        applyPlanarOverlay16(buffer, decoder, overlay);
    }
}

} // namespace video

#undef VIDEO_DEBUG_LOG
