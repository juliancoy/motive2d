#include "decoder_cpu.h"

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

#include "display2d.h"
#include "engine2d.h"
#include "fps.h"
#include "color_grading_ui.h"
#include "utils.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
}

// Look for the sample video in the current directory (files were moved up).
const std::filesystem::path kDefaultVideoPath = std::filesystem::path("P1090533_main8_hevc_fast.mkv");

// Interrupt callback for FFmpeg
static int interrupt_callback(void *opaque)
{
    DecoderCPU *decoder = static_cast<DecoderCPU *>(opaque);
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

// Configure format for pixel format
bool DecoderCPU::configureFormatForPixelFormat(AVPixelFormat pix_fmt)
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
void DecoderCPU::copyDecodedFrameToBuffer(const AVFrame *frame, std::vector<uint8_t> &buffer)
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
DecoderCPU::DecoderCPU(const std::filesystem::path &videoPath,
    bool debugLogging)
    : engine(nullptr)
{
    
    std::cout << "[Video] Loading video file: " << videoPath << std::endl;
    if (avformat_open_input(&formatCtx, videoPath.string().c_str(), nullptr, nullptr) < 0)
    {
        std::cerr << "[Video] Failed to open file: " << videoPath << std::endl;
    }

    formatCtx->interrupt_callback.callback = interrupt_callback;
    formatCtx->interrupt_callback.opaque = this;

    if (avformat_find_stream_info(formatCtx, nullptr) < 0)
    {
        std::cerr << "[Video] Unable to read stream info for: " << videoPath << std::endl;
    }

    videoStreamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoStreamIndex < 0)
    {
        std::cerr << "[Video] No video stream found in file: " << videoPath << std::endl;
    }

    AVStream *videoStream = formatCtx->streams[videoStreamIndex];
    const AVCodec *codec = avcodec_find_decoder(videoStream->codecpar->codec_id);

    if (avcodec_parameters_to_context(codecCtx, videoStream->codecpar) < 0)
    {
        std::cerr << "[Video] Unable to copy codec parameters." << std::endl;
    }

    
    implementationName = "Software (CPU)";

    const int streamBitDepth = determineStreamBitDepth(videoStream, codecCtx);
    
    // Always use software decoding
    AVPixelFormat sourceFormat = static_cast<AVPixelFormat>(videoStream->codecpar->format);
    requestedSwPixelFormat = pickPreferredSwPixelFormat(streamBitDepth, sourceFormat);
    codecCtx->sw_pix_fmt = requestedSwPixelFormat;

    if (debugLogging)
    {
        std::cout << "[Video] Requesting " << pixelFormatDescription(requestedSwPixelFormat)
                  << " software frames from CPU decoder (bit depth " << streamBitDepth << ")" << std::endl;
    }

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

    if (!configureFormatForPixelFormat(requestedSwPixelFormat))
    {
        std::cerr << "[Video] Unsupported pixel format for decoder: "
                  << pixelFormatDescription(requestedSwPixelFormat) << std::endl;
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
        std::cout << "[Video] Using " << implementationName
                  << " (sw " << pixelFormatDescription(requestedSwPixelFormat) << ")"
                  << " decoder" << std::endl;
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

// Destructor
DecoderCPU::~DecoderCPU()
{
    stopAsyncDecoding();
    cleanupVideoDecoderCPU();
}

VkSampler DecoderCPU::createLinearClampSampler(Engine2D *engine)
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

void DecoderCPU::destroyExternalVideoViews()
{
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

bool DecoderCPU::seekVideoDecoderCPU(double targetSeconds)
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

bool DecoderCPU::seek(float timeSeconds)
{
    if (!formatCtx)
    {
        return false;
    }

    timeSeconds = std::clamp(timeSeconds, 0.0f, static_cast<float>(durationSeconds));
    currentTimeSeconds = timeSeconds;

    stopAsyncDecoding();
    if (!seekVideoDecoderCPU(timeSeconds))
    {
        std::cerr << "[DecoderCPU] Failed to seek video decoder.\n";
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

bool DecoderCPU::decodeNextFrame(DecodedFrame &decodedFrame)
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
            std::cout << "[Video] Received frame: width=" << frame->width << " height=" << frame->height
                      << " format=" << frame->format << " (" << pixelFormatDescription(static_cast<AVPixelFormat>(frame->format)) << ")"
                      << " data[0]=" << (void*)frame->data[0] << " linesize[0]=" << frame->linesize[0] << std::endl;
            
            // Always use CPU frame format for pure software decoding
            const AVPixelFormat frameFormat = static_cast<AVPixelFormat>(frame->format);
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

                std::cout << "[Video] DecoderCPU output pixel format changed to "
                          << pixelFormatDescription(frameFormat) << std::endl;
            }

            // Always copy frame buffer for CPU decoding
            bool doCopy = true;

            if (doCopy)
            {
                std::cout << "[Video] copyDecodedFrameToBuffer: width=" << width << " height=" << height
                          << " bufferSize=" << bufferSize << " outputFormat=" << static_cast<int>(outputFormat)
                          << " bytesPerComponent=" << bytesPerComponent << std::endl;
                copyDecodedFrameToBuffer(frame, decodedFrame.buffer);
                std::cout << "[Video] after copy, buffer size=" << decodedFrame.buffer.size() << std::endl;
                ptsFrame = frame;
            }
            else
            {
                decodedFrame.buffer.clear();
                // For CPU-only decoding, we always need to copy the buffer
                // since there are no Vulkan surfaces available
                copyDecodedFrameToBuffer(frame, decodedFrame.buffer);
                ptsFrame = frame;
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
            std::cerr << "[Video] DecoderCPU error: " << ffmpegErrorString(receiveResult) << std::endl;
            return false;
        }
    }
}

void DecoderCPU::asyncDecodeLoop()
{
    std::cout << "[Video] asyncDecodeLoop started" << std::endl;

    DecodedFrame localFrame;
    localFrame.buffer.reserve(bufferSize);
    int frameCount = 0;
    bool preferZeroCopy = false; // Always false for CPU-only decoding

    while (!stopRequested.load())
    {
        frameCount++;
        bool decodeSuccess = decodeNextFrame(localFrame); // Always copy for CPU

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

bool DecoderCPU::startAsyncDecoding(size_t maxBufferedFrames)
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
        decodeThread = std::thread(&DecoderCPU::asyncDecodeLoop, this);
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

bool DecoderCPU::acquireDecodedFrame(DecodedFrame &outFrame)
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

void DecoderCPU::stopAsyncDecoding()
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

void DecoderCPU::pumpDecodedFrames()
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

double DecoderCPU::advancePlayback()
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

    lastFramePtsSeconds = frame.ptsSeconds;
    lastFrameRenderTime = currentTime;
    lastDisplayedSeconds = std::max(0.0, lastFramePtsSeconds - basePtsSeconds);
    return lastDisplayedSeconds;
}

void DecoderCPU::cleanupVideoDecoderCPU()
{
    if (packet)
    {
        av_packet_free(&packet);
    }
    if (frame)
    {
        av_frame_free(&frame);
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

int DecoderCPU::runDecodeOnlyBenchmark(const std::filesystem::path &videoPath,
                                    const std::optional<bool> &swapUvOverride,
                                    double benchmarkSeconds)
{
    const double kBenchmarkSeconds = benchmarkSeconds > 0.0 ? benchmarkSeconds : 5.0;
    if (!std::filesystem::exists(videoPath))
    {
        std::cerr << "[DecodeOnly] Missing video file: " << videoPath << std::endl;
        return 1;
    }

    try
    {
        DecoderCPU decoder(videoPath, true);

        DecodedFrame frame;
        frame.buffer.reserve(decoder.bufferSize);

        auto start = std::chrono::steady_clock::now();
        size_t framesDecoded = 0;
        double firstPts = -1.0;
        double decodedSeconds = 0.0;
        while (decoder.decodeNextFrame(frame))
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