#include "decoder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
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

extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

namespace
{
    // Look for the sample video in the current directory (files were moved up).
    const std::filesystem::path kDefaultVideoPath = std::filesystem::path("P1090533_main8_hevc_fast.mkv");
}

VkSampler Decoder::createLinearClampSampler(Engine2D *engine)
{
    VkSampler sampler = VK_NULL_HANDLE;
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = engine->getDeviceProperties().limits.maxSamplerAnisotropy;
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

Decoder::Decoder(const std::filesystem::path& videoPath,
                            VideoDecoder& decoder,
                            const DecoderInitParams& initParams)
{
    seekTargetMicroseconds.store(-1);
    if (avformat_open_input(&formatCtx, videoPath.string().c_str(), nullptr, nullptr) < 0)
    {
        std::cerr << "[Video] Failed to open file: " << videoPath << std::endl;
        return false;
    }

    formatCtx->interrupt_callback.callback = interrupt_callback;
    formatCtx->interrupt_callback.opaque = &decoder;

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

    AVStream* videoStream = formatCtx->streams[videoStreamIndex];
    const AVCodec* codec = avcodec_find_decoder(videoStream->codecpar->codec_id);
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

    if (!configureDecodeImplementation(decoder,
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
        // Get the source pixel format from the stream
        AVPixelFormat sourceFormat = static_cast<AVPixelFormat>(videoStream->codecpar->format);
        requestedSwPixelFormat = pickPreferredSwPixelFormat(streamBitDepth, sourceFormat);
        codecCtx->sw_pix_fmt = requestedSwPixelFormat;
        VIDEO_DEBUG_LOG(std::cout << "[Video] Requesting " << pixelFormatDescription(requestedSwPixelFormat)
                                << " software frames from " << implementationName
                                << " decoder (bit depth " << streamBitDepth << ")" << std::endl);
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
    planarYuv = false;
    chromaInterleaved = false;
    outputFormat = PrimitiveYuvFormat::NV12;
    bytesPerComponent = 1;
    bitDepth = 8;
    chromaDivX = 2;
    chromaDivY = 2;
    chromaWidth = std::max<uint32_t>(1u, static_cast<uint32_t>((width + chromaDivX - 1) / chromaDivX));
    chromaHeight = std::max<uint32_t>(1u, static_cast<uint32_t>((height + chromaDivY - 1) / chromaDivY));
    yPlaneBytes = 0;
    uvPlaneBytes = 0;

    if (!configureFormatForPixelFormat(decoder, codecCtx->pix_fmt))
    {
        std::cerr << "[Video] Unsupported pixel format for decoder: "
                  << pixelFormatDescription(codecCtx->pix_fmt) << std::endl;
        return false;
    }

    VIDEO_DEBUG_LOG(std::cout << "[Video] Stream pixel format: " << pixelFormatDescription(sourcePixelFormat)
                              << " -> outputFormat=" << static_cast<int>(outputFormat)
                              << " chromaDiv=" << chromaDivX << "x" << chromaDivY
                              << " bytesPerComponent=" << bytesPerComponent
                              << " bitDepth=" << bitDepth
                              << " swapUV=" << (swapChromaUV ? "yes" : "no")
                              << " colorSpace=" << colorSpaceName(colorSpace)
                              << " colorRange=" << colorRangeName(colorRange)
                              << " implementation=" << implementationName
                              << std::endl);

    AVRational frameRate = av_guess_frame_rate(formatCtx, videoStream, nullptr);
    double fps = (frameRate.num != 0 && frameRate.den != 0) ? av_q2d(frameRate) : 30.0;
    fps = fps > 0.0 ? fps : 30.0;
    VIDEO_DEBUG_LOG(std::cout << "[Video] Using " << implementationName);
    if (implementation != DecodeImplementation::Software)
    {
        VIDEO_DEBUG_LOG(std::cout << " (hw " << pixelFormatName(hwPixelFormat));
        if (requestedSwPixelFormat != AV_PIX_FMT_NONE)
        {
            VIDEO_DEBUG_LOG(std::cout << " -> sw " << pixelFormatName(requestedSwPixelFormat));
        }
        VIDEO_DEBUG_LOG(std::cout << ")");
    }
    VIDEO_DEBUG_LOG(std::cout << " decoder" << std::endl);
    

    engine = engine;
    if (!std::filesystem::exists(videoPath))
    {
        std::cerr << "[Video2D] Missing video file: " << videoPath << std::endl;
        return false;
    }

    DecoderInitParams params{};
    params.implementation = DecodeImplementation::Vulkan;
    if (engine)
    {
        VulkanInteropContext interop{};
        interop.instance = engine->instance;
        interop.physicalDevice = engine->physicalDevice;
        interop.device = engine->logicalDevice;
        interop.graphicsQueue = engine->getGraphicsQueue();
        interop.graphicsQueueFamilyIndex = engine->getGraphicsQueueFamilyIndex();
        interop.videoQueue = engine->getVideoQueue();
        interop.videoQueueFamilyIndex = engine->getVideoQueueFamilyIndex();
        params.vulkanInterop = interop;
    }
    if (!initializeVideoDecoder(videoPath, decoder, params))
    {
        std::cerr << "[Video2D] Vulkan decode unavailable";
        if (!hardwareInitFailureReason.empty())
        {
            std::cerr << " (" << hardwareInitFailureReason << ")";
        }
        std::cerr << ", falling back to software." << std::endl;
        params.implementation = DecodeImplementation::Software;
        if (!initializeVideoDecoder(videoPath, decoder, params))
        {
            std::cerr << "[Video2D] Failed to initialize decoder" << std::endl;
            return false;
        }
    }
    if (implementation != DecodeImplementation::Vulkan)
    {
        std::cerr << "[Video2D] Warning: hardware Vulkan decode not active; using "
                  << implementationName << std::endl;
    }

    durationSeconds = 0.0;
    if (formatCtx && formatCtx->duration > 0)
    {
        durationSeconds = static_cast<double>(formatCtx->duration) / static_cast<double>(AV_TIME_BASE);
    }

    colorInfo = deriveVideoColorInfo(decoder);

    try
    {
        video.sampler = createLinearClampSampler(engine);
        overlay.sampler = createLinearClampSampler(engine);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[Video2D] Failed to create samplers: " << ex.what() << std::endl;
        cleanupVideoDecoder(decoder);
        return false;
    }

    DecodedFrame initialDecoded{};
    initialDecoded.buffer.assign(static_cast<size_t>(bufferSize), 0);
    if (outputFormat == PrimitiveYuvFormat::NV12)
    {
        const size_t yBytes = yPlaneBytes;
        if (yBytes > 0 && yBytes <= initialDecoded.buffer.size())
        {
            std::fill(initialDecoded.buffer.begin(), initialDecoded.buffer.begin() + yBytes, 0x80);
            std::fill(initialDecoded.buffer.begin() + yBytes, initialDecoded.buffer.end(), 0x80);
        }
    }
    else
    {
        const size_t yBytes = yPlaneBytes;
        const size_t uvBytes = uvPlaneBytes;
        const bool sixteenBit = bytesPerComponent > 1;
        if (sixteenBit)
        {
            const uint32_t bitDepth = bitDepth > 0 ? bitDepth : 8;
            const uint32_t shift = bitDepth >= 16 ? 0u : 16u - bitDepth;
            const uint16_t baseValue = static_cast<uint16_t>(1u << (bitDepth > 0 ? bitDepth - 1 : 7));
            const uint16_t fillValue = static_cast<uint16_t>(baseValue << shift);
            if (yBytes >= sizeof(uint16_t))
            {
                uint16_t *yDst = reinterpret_cast<uint16_t *>(initialDecoded.buffer.data());
                std::fill(yDst, yDst + (yBytes / sizeof(uint16_t)), fillValue);
            }
            if (uvBytes >= sizeof(uint16_t))
            {
                uint16_t *uvDst = reinterpret_cast<uint16_t *>(initialDecoded.buffer.data() + yBytes);
                std::fill(uvDst, uvDst + (uvBytes / sizeof(uint16_t)), fillValue);
            }
        }
        else
        {
            if (yBytes > 0 && yBytes <= initialDecoded.buffer.size())
            {
                std::fill(initialDecoded.buffer.begin(), initialDecoded.buffer.begin() + yBytes, 0x80);
            }
            if (uvBytes > 0 && yBytes + uvBytes <= initialDecoded.buffer.size())
            {
                std::fill(initialDecoded.buffer.begin() + yBytes, initialDecoded.buffer.begin() + yBytes + uvBytes, 0x80);
            }
        }
    }

    if (!uploadDecodedFrame(video, engine, decoder, initialDecoded))
    {
        std::cerr << "[Video2D] Failed to upload initial frame." << std::endl;
        cleanupVideoDecoder(decoder);
        destroyImageResource(engine, video.lumaImage);
        destroyImageResource(engine, video.chromaImage);
        if (video.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, video.sampler, nullptr);
            video.sampler = VK_NULL_HANDLE;
        }
        if (overlay.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, overlay.sampler, nullptr);
            overlay.sampler = VK_NULL_HANDLE;
        }
        return false;
    }

    if (!startAsyncDecoding(decoder, 12))
    {
        std::cerr << "[Video2D] Failed to start async decoder" << std::endl;
        cleanupVideoDecoder(decoder);
        destroyImageResource(engine, video.lumaImage);
        destroyImageResource(engine, video.chromaImage);
        if (video.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, video.sampler, nullptr);
            video.sampler = VK_NULL_HANDLE;
        }
        if (overlay.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, overlay.sampler, nullptr);
            overlay.sampler = VK_NULL_HANDLE;
        }
        return false;
    }

    if (swapUvOverride.has_value())
    {
        swapChromaUV = swapUvOverride.value();
        if (debugLoggingEnabled())
        {
            std::cout << "[Video2D] Forcing UV swap to "
                      << (swapChromaUV ? "ON" : "OFF") << std::endl;
        }
    }

    stagingFrame = DecodedFrame{};
    stagingFrame.buffer.reserve(static_cast<size_t>(bufferSize));
    pendingFrames.clear();
    playbackClockInitialized = false;
    lastDisplayedSeconds = 0.0;
    return true;
}

bool seekVideoDecoder()
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
        stopAsyncDecoding(decoder);
    }

    VIDEO_DEBUG_LOG(std::cout << "[Video] Seeking to " << targetSeconds << "s..." << std::endl);
    seekTargetMicroseconds.store(static_cast<int64_t>(targetSeconds * 1000000.0));

    // Convert target seconds to the stream's timebase
    AVStream* videoStream = formatCtx->streams[videoStreamIndex];
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
            startAsyncDecoding(decoder, maxBufferedFrames);
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
        if (!startAsyncDecoding(decoder, maxBufferedFrames))
        {
            std::cerr << "[Video] Failed to restart async decoding after seek." << std::endl;
            return false;
        }
    }

    return true;
}

bool decodeNextFrame(VideoDecoder& decoder, DecodedFrame& decodedFrame, bool copyFrameBuffer)
{
    if (finished.load())
    {
    VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: decoder finished, returning false" << std::endl);
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
            auto readDuration = std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart);
            
            if (readDuration.count() > 100 && callCount % 10 == 0)
            {
                VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: av_read_frame took " << readDuration.count() 
                                          << " ms (call " << callCount << ")" << std::endl);
            }
            
            if (readResult >= 0)
            {
                if (packet->stream_index == videoStreamIndex)
                {
                    auto sendStart = std::chrono::steady_clock::now();
                    if (avcodec_send_packet(codecCtx, packet) < 0)
                    {
                        std::cerr << "[Video] Failed to send packet to " << std::endl;
                    }
                    auto sendEnd = std::chrono::steady_clock::now();
                    auto sendDuration = std::chrono::duration_cast<std::chrono::milliseconds>(sendEnd - sendStart);
                    
                    if (sendDuration.count() > 50 && callCount % 10 == 0)
                    {
                        VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: avcodec_send_packet took " 
                                                  << sendDuration.count() << " ms" << std::endl);
                    }
                }
                av_packet_unref(packet);
            }
            else
            {
                av_packet_unref(packet);
                draining = true;
                VIDEO_DEBUG_LOG(std::cout << "[Video] decodeNextFrame: starting drain mode" << std::endl);
                avcodec_send_packet(codecCtx, nullptr);
            }
        }

        auto receiveStart = std::chrono::steady_clock::now();
        int receiveResult = avcodec_receive_frame(codecCtx, frame);
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
                implementation == DecodeImplementation::Vulkan &&
                hwDeviceType == AV_HWDEVICE_TYPE_VULKAN &&
                frame->format == AV_PIX_FMT_VULKAN;

            const AVPixelFormat frameFormat = isHardwareVulkan
                                                  ? codecCtx->sw_pix_fmt
                                                  : static_cast<AVPixelFormat>(frame->format);

            const AVFrame* ptsFrame = frame;
            // Update decoder dimensions/pixel format even if we skip buffer copy
            if (width != frame->width || height != frame->height ||
                frameFormat != sourcePixelFormat)
            {
                width = frame->width;
                height = frame->height;
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
                if (frame->format != AV_PIX_FMT_VULKAN ||
                    hwDeviceType != AV_HWDEVICE_TYPE_VULKAN)
                {
                    doCopy = true;
                }
            }

            if (doCopy)
            {
                AVFrame* workingFrame = frame;
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

                copyDecodedFrameToBuffer(decoder, workingFrame, decodedFrame.buffer);
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
                    const AVVkFrame* vkf = reinterpret_cast<AVVkFrame*>(frame->data[0]);
                    if (vkf)
                    {
                        decodedFrame.vkSurface.valid = true;
                        decodedFrame.vkSurface.width = static_cast<uint32_t>(frame->width);
                        decodedFrame.vkSurface.height = static_cast<uint32_t>(frame->height);
                        const VkFormat* vkFormats = av_vkfmt_from_pixfmt(codecCtx->sw_pix_fmt);
                        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codecCtx->sw_pix_fmt);
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
            finished.store(true);
            return false;
        }
        else
        {
            std::cerr << "[Video] Decoder error: " << receiveResult << std::endl;
            return false;
        }
    }
}

void Decoder::asyncDecodeLoop(VideoDecoder* decoder)
{
    VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop started, stopRequested=" << stopRequested.load() 
                              << ", bufferSize=" << bufferSize << std::endl);
    
    DecodedFrame localFrame;
    localFrame.buffer.reserve(bufferSize);
    int frameCount = 0;
    auto loopStartTime = std::chrono::steady_clock::now();
    bool preferZeroCopy =
        implementation == DecodeImplementation::Vulkan &&
        hwDeviceType == AV_HWDEVICE_TYPE_VULKAN;
    
    while (!stopRequested.load())
    {
        frameCount++;
        auto decodeStart = std::chrono::steady_clock::now();
        bool decodeSuccess = decodeNextFrame(*decoder, localFrame, /*copyFrameBuffer=*/!preferZeroCopy);
        auto decodeEnd = std::chrono::steady_clock::now();
        auto decodeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(decodeEnd - decodeStart);
        
        if (!decodeSuccess)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop: decodeNextFrame returned false at frame " 
                                      << frameCount << ", finished=" << finished.load() 
                                      << ", decode took " << decodeDuration.count() << " ms" << std::endl);
            break;
        }

        int64_t currentSeekTarget = seekTargetMicroseconds.load();
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
                seekTargetMicroseconds.store(-1);
                VIDEO_DEBUG_LOG(std::cout << "[Video] Seek target reached." << std::endl);
            }
        }
        
        if (frameCount % 30 == 0)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop: decoded frame " << frameCount 
                                      << ", pts=" << localFrame.ptsSeconds 
                                      << "s, decode took " << decodeDuration.count() << " ms" << std::endl);
        }

        std::unique_lock<std::mutex> lock(frameMutex);
        auto waitStart = std::chrono::steady_clock::now();
        frameCond.wait(lock, [decoder]() {
            return stopRequested.load() || frameQueue.size() < maxBufferedFrames;
        });
        auto waitEnd = std::chrono::steady_clock::now();
        auto waitDuration = std::chrono::duration_cast<std::chrono::milliseconds>(waitEnd - waitStart);
        
        if (waitDuration.count() > 10 && frameCount % 10 == 0 && false)
        {
            VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop: waited " << waitDuration.count() 
                                      << " ms for frame queue, size=" << frameQueue.size() 
                                      << "/" << maxBufferedFrames << std::endl);
        }

        if (stopRequested.load())
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

        frameQueue.emplace_back(std::move(localFrame));
        lock.unlock();
        frameCond.notify_all();
        localFrame = DecodedFrame{};
        localFrame.buffer.reserve(bufferSize);
    }

    auto loopEndTime = std::chrono::steady_clock::now();
    auto loopDuration = std::chrono::duration_cast<std::chrono::milliseconds>(loopEndTime - loopStartTime);
    
    VIDEO_DEBUG_LOG(std::cout << "[Video] asyncDecodeLoop exiting after " << frameCount << " frames, " 
                              << loopDuration.count() << " ms, stopRequested=" << stopRequested.load() 
                              << ", threadRunning=" << threadRunning.load() << std::endl);
    
    threadRunning.store(false);
    frameCond.notify_all();
}

bool Decoder::startAsyncDecoding(VideoDecoder& decoder, size_t maxBufferedFrames)
{
    if (asyncDecoding)
    {
        return true;
    }

    maxBufferedFrames = std::max<size_t>(1, maxBufferedFrames);
    stopRequested.store(false);
    threadRunning.store(true);
    asyncDecoding = true;
    frameQueue.clear();

    try
    {
        decodeThread = std::thread(asyncDecodeLoop, &decoder);
    }
    catch (const std::system_error& err)
    {
        std::cerr << "[Video] Failed to start decode thread: " << err.what() << std::endl;
        threadRunning.store(false);
        asyncDecoding = false;
        return false;
    }

    return true;
}

bool Decoder::acquireDecodedFrame(VideoDecoder& decoder, DecodedFrame& outFrame)
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

void Decoder::stopAsyncDecoding(VideoDecoder& decoder)
{
    VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding called, asyncDecoding=" << asyncDecoding 
                              << ", threadRunning=" << threadRunning.load() 
                              << ", stopRequested=" << stopRequested.load() << std::endl);
    
    if (!asyncDecoding)
    {
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: not async decoding, returning early" << std::endl);
        return;
    }

    auto startTime = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        stopRequested.store(true);
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: set stopRequested=true, frameQueue size=" 
                              << frameQueue.size() << std::endl);
    }
    frameCond.notify_all();
    VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: notified condition variable" << std::endl);

    if (decodeThread.joinable())
    {
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: joining decode thread..." << std::endl);
        auto joinStart = std::chrono::steady_clock::now();
        decodeThread.join();
        auto joinEnd = std::chrono::steady_clock::now();
        auto joinDuration = std::chrono::duration_cast<std::chrono::milliseconds>(joinEnd - joinStart);
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: decode thread joined after " 
                                  << joinDuration.count() << " ms" << std::endl);
    }
    else
    {
        VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding: decode thread not joinable" << std::endl);
    }

    frameQueue.clear();
    asyncDecoding = false;
    threadRunning.store(false);
    stopRequested.store(false);
    
    auto endTime = std::chrono::steady_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    VIDEO_DEBUG_LOG(std::cout << "[Video] stopAsyncDecoding completed in " << totalDuration.count() << " ms" << std::endl);
}

void Decoder::~Decoder()
{
    stopAsyncDecoding(decoder);
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

bool Decoder::uploadDecodedFrame(VideoResources &video,
                        Engine2D *engine,
                        const VideoDecoder &decoder,
                        const DecodedFrame &frame)
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
        destroyExternalVideoViews(engine, video);

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
            destroyExternalVideoViews(engine, video);
            return false;
        }

        video.externalLumaView = lumaView;
        video.externalChromaView = chromaView != VK_NULL_HANDLE ? chromaView : lumaView;
        video.usingExternal = true;

        video.descriptors.width = frame.vkSurface.width;
        video.descriptors.height = frame.vkSurface.height;
        video.descriptors.chromaDivX = chromaDivX;
        video.descriptors.chromaDivY = chromaDivY;
        video.descriptors.luma.view = video.externalLumaView;
        video.descriptors.luma.sampler = video.sampler;
        video.descriptors.chroma.view = video.externalChromaView;
        video.descriptors.chroma.sampler = video.sampler;
        if (debugLoggingEnabled())
        {
            std::cout << "[Video2D] Uploaded frame " << video.descriptors.width << "x" << video.descriptors.height
                      << " chromaDiv=" << video.descriptors.chromaDivX << "x" << video.descriptors.chromaDivY
                      << " external=" << (video.usingExternal ? "yes" : "no") << std::endl;
        }
        return true;
    }

    // CPU upload path
    if (video.usingExternal)
    {
        vkDeviceWaitIdle(engine->logicalDevice);
        destroyExternalVideoViews(engine, video);
    }

    if (frame.buffer.empty())
    {
        return false;
    }

    if (outputFormat == PrimitiveYuvFormat::NV12)
    {
        const size_t ySize = yPlaneBytes;
        const size_t uvSize = uvPlaneBytes;
        if (frame.buffer.size() < ySize + uvSize)
        {
            std::cerr << "[Video2D] NV12 frame smaller than expected." << std::endl;
            return false;
        }
        const uint8_t *yPlane = frame.buffer.data();
        const uint8_t *uvPlane = yPlane + ySize;
        if (!uploadImageData(engine,
                                      video.lumaImage,
                                      yPlane,
                                      ySize,
                                      width,
                                      height,
                                      VK_FORMAT_R8_UNORM))
        {
            return false;
        }
        if (!uploadImageData(engine,
                                      video.chromaImage,
                                      uvPlane,
                                      uvSize,
                                      chromaWidth,
                                      chromaHeight,
                                      VK_FORMAT_R8G8_UNORM))
        {
            return false;
        }
    }
    else
    {
        const uint8_t *yPlane = frame.buffer.data();
        const uint8_t *uvPlane = yPlane + yPlaneBytes;
        const bool sixteenBit = bytesPerComponent > 1;
        const VkFormat lumaFormat = sixteenBit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
        const VkFormat chromaFormat = sixteenBit ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM;
        if (!uploadImageData(engine,
                                      video.lumaImage,
                                      yPlane,
                                      yPlaneBytes,
                                      width,
                                      height,
                                      lumaFormat))
        {
            return false;
        }
        if (!uploadImageData(engine,
                                      video.chromaImage,
                                      uvPlane,
                                      uvPlaneBytes,
                                      chromaWidth,
                                      chromaHeight,
                                      chromaFormat))
        {
            return false;
        }
    }

    video.descriptors.width = width;
    video.descriptors.height = height;
    video.descriptors.chromaDivX = chromaDivX;
    video.descriptors.chromaDivY = chromaDivY;
    video.descriptors.luma.view = video.lumaImage.view;
    video.descriptors.luma.sampler = video.sampler;
    video.descriptors.chroma.view = video.chromaImage.view ? video.chromaImage.view : video.lumaImage.view;
    video.descriptors.chroma.sampler = video.sampler;
    if (debugLoggingEnabled())
    {
        std::cout << "[Video2D] Uploaded frame " << video.descriptors.width << "x" << video.descriptors.height
                  << " chromaDiv=" << video.descriptors.chromaDivX << "x" << video.descriptors.chromaDivY
                  << " external=" << (video.usingExternal ? "yes" : "no") << std::endl;
    }
    return true;
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

    VideoDecoder decoder;
    DecoderInitParams params{};
    params.implementation = DecodeImplementation::Vulkan;
    params.requireGraphicsQueue = false;

    if (!initializeVideoDecoder(videoPath, decoder, params))
    {
        std::cerr << "[DecodeOnly] Vulkan decode unavailable";
        if (!hardwareInitFailureReason.empty())
        {
            std::cerr << " (" << hardwareInitFailureReason << ")";
        }
        std::cerr << ", falling back to software." << std::endl;
        params.implementation = DecodeImplementation::Software;
        if (!initializeVideoDecoder(videoPath, decoder, params))
        {
            std::cerr << "[DecodeOnly] Failed to initialize " << std::endl;
            return 1;
        }
    }

    if (swapUvOverride.has_value())
    {
        swapChromaUV = swapUvOverride.value();
    }

    DecodedFrame frame;
    frame.buffer.reserve(static_cast<size_t>(bufferSize));

    auto start = std::chrono::steady_clock::now();
    size_t framesDecoded = 0;
    double firstPts = -1.0;
    double decodedSeconds = 0.0;
    while (decodeNextFrame(decoder, frame, /*copyFrameBuffer=*/false))
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
              << implementationName << std::endl;

    cleanupVideoDecoder(decoder);
    return 0;
}
void pumpDecodedFrames()
{
    constexpr size_t kMaxPendingFrames = 6;
    while (pendingFrames.size() < kMaxPendingFrames &&
           acquireDecodedFrame(decoder, stagingFrame))
    {
        pendingFrames.emplace_back(std::move(stagingFrame));
        stagingFrame = DecodedFrame{};
        stagingFrame.buffer.reserve(static_cast<size_t>(bufferSize));
    }
}

double advancePlayback()
{
    if (video.sampler == VK_NULL_HANDLE)
    {
        return 0.0;
    }

    pumpDecodedFrames(state);

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

    if (!uploadDecodedFrame(video, engine, decoder, frame))
    {
        std::cerr << "[Video2D] Failed to upload decoded frame." << std::endl;
    }

    lastFramePtsSeconds = frame.ptsSeconds;
    lastFrameRenderTime = currentTime;
    lastDisplayedSeconds = std::max(0.0, lastFramePtsSeconds - basePtsSeconds);
    return lastDisplayedSeconds;
}

void Decoder::seek(float timeSeconds)
{
    if (!videoLoaded)
    {
        return;
    }

    timeSeconds = std::clamp(timeSeconds, 0.0f, duration);
    currentTime = timeSeconds;

    stopAsyncDecoding(playbackdecoder);
    if (!seekVideoDecoder(playbackdecoder, timeSeconds))
    {
        std::cerr << "[Engine2D] Failed to seek video decoder.\n";
    }

    vkDeviceWaitIdle(logicalDevice);

    playbackpendingFrames.clear();
    playbackplaybackClockInitialized = false;

    startAsyncDecoding(playbackdecoder, 12);
}
