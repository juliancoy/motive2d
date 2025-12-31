#include "decode.h"

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
#include "overlay.hpp"
#include "grading.hpp"
#include "utils.h"
#include "video.h"

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

VkSampler createLinearClampSampler(Engine2D *engine)
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

void destroyExternalVideoViews(Engine2D *engine, VideoResources &video)
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

bool waitForVulkanFrameReady(Engine2D *engine, const video::DecodedFrame::VulkanSurface &surface)
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

bool uploadDecodedFrame(VideoResources &video,
                        Engine2D *engine,
                        const video::VideoDecoder &decoder,
                        const video::DecodedFrame &frame)
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
        video.descriptors.chromaDivX = decoder.chromaDivX;
        video.descriptors.chromaDivY = decoder.chromaDivY;
        video.descriptors.luma.view = video.externalLumaView;
        video.descriptors.luma.sampler = video.sampler;
        video.descriptors.chroma.view = video.externalChromaView;
        video.descriptors.chroma.sampler = video.sampler;
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

    if (decoder.outputFormat == PrimitiveYuvFormat::NV12)
    {
        const size_t ySize = decoder.yPlaneBytes;
        const size_t uvSize = decoder.uvPlaneBytes;
        if (frame.buffer.size() < ySize + uvSize)
        {
            std::cerr << "[Video2D] NV12 frame smaller than expected." << std::endl;
            return false;
        }
        const uint8_t *yPlane = frame.buffer.data();
        const uint8_t *uvPlane = yPlane + ySize;
        if (!overlay::uploadImageData(engine,
                                      video.lumaImage,
                                      yPlane,
                                      ySize,
                                      decoder.width,
                                      decoder.height,
                                      VK_FORMAT_R8_UNORM))
        {
            return false;
        }
        if (!overlay::uploadImageData(engine,
                                      video.chromaImage,
                                      uvPlane,
                                      uvSize,
                                      decoder.chromaWidth,
                                      decoder.chromaHeight,
                                      VK_FORMAT_R8G8_UNORM))
        {
            return false;
        }
    }
    else
    {
        const uint8_t *yPlane = frame.buffer.data();
        const uint8_t *uvPlane = yPlane + decoder.yPlaneBytes;
        const bool sixteenBit = decoder.bytesPerComponent > 1;
        const VkFormat lumaFormat = sixteenBit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
        const VkFormat chromaFormat = sixteenBit ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM;
        if (!overlay::uploadImageData(engine,
                                      video.lumaImage,
                                      yPlane,
                                      decoder.yPlaneBytes,
                                      decoder.width,
                                      decoder.height,
                                      lumaFormat))
        {
            return false;
        }
        if (!overlay::uploadImageData(engine,
                                      video.chromaImage,
                                      uvPlane,
                                      decoder.uvPlaneBytes,
                                      decoder.chromaWidth,
                                      decoder.chromaHeight,
                                      chromaFormat))
        {
            return false;
        }
    }

    video.descriptors.width = decoder.width;
    video.descriptors.height = decoder.height;
    video.descriptors.chromaDivX = decoder.chromaDivX;
    video.descriptors.chromaDivY = decoder.chromaDivY;
    video.descriptors.luma.view = video.lumaImage.view;
    video.descriptors.luma.sampler = video.sampler;
    video.descriptors.chroma.view = video.chromaImage.view ? video.chromaImage.view : video.lumaImage.view;
    video.descriptors.chroma.sampler = video.sampler;
    return true;
}

int runDecodeOnlyBenchmark(const std::filesystem::path &videoPath, const std::optional<bool> &swapUvOverride)
{
    constexpr double kBenchmarkSeconds = 5.0; // default window to measure decode speed
    if (!std::filesystem::exists(videoPath))
    {
        std::cerr << "[DecodeOnly] Missing video file: " << videoPath << std::endl;
        return 1;
    }

    video::VideoDecoder decoder;
    video::DecoderInitParams params{};
    params.implementation = video::DecodeImplementation::Vulkan;
    params.requireGraphicsQueue = false;

    if (!video::initializeVideoDecoder(videoPath, decoder, params))
    {
        std::cerr << "[DecodeOnly] Vulkan decode unavailable, falling back to software." << std::endl;
        params.implementation = video::DecodeImplementation::Software;
        if (!video::initializeVideoDecoder(videoPath, decoder, params))
        {
            std::cerr << "[DecodeOnly] Failed to initialize decoder." << std::endl;
            return 1;
        }
    }

    if (swapUvOverride.has_value())
    {
        decoder.swapChromaUV = swapUvOverride.value();
    }

    video::DecodedFrame frame;
    frame.buffer.reserve(static_cast<size_t>(decoder.bufferSize));

    auto start = std::chrono::steady_clock::now();
    size_t framesDecoded = 0;
    double firstPts = -1.0;
    while (video::decodeNextFrame(decoder, frame, /*copyFrameBuffer=*/false))
    {
        framesDecoded++;
        if (firstPts < 0.0)
        {
            firstPts = frame.ptsSeconds;
        }

        // Stop after decoding the first kBenchmarkSeconds worth of video.
        double elapsedPts = frame.ptsSeconds - (firstPts < 0.0 ? 0.0 : firstPts);
        if (elapsedPts >= kBenchmarkSeconds)
        {
            break;
        }

        frame.buffer.clear();
    }
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double fps = seconds > 0.0 ? static_cast<double>(framesDecoded) / seconds : 0.0;

    std::cout << "[DecodeOnly] Decoded " << framesDecoded << " frames in " << seconds
              << "s -> " << fps << " fps using " << decoder.implementationName
              << " over ~" << kBenchmarkSeconds << "s of content" << std::endl;

    video::cleanupVideoDecoder(decoder);
    return 0;
}

bool initializeVideoPlayback(const std::filesystem::path &videoPath,
                             Engine2D *engine,
                             VideoPlaybackState &state,
                             double &durationSeconds,
                             const std::optional<bool> &swapUvOverride)
{
    state.engine = engine;
    if (!std::filesystem::exists(videoPath))
    {
        std::cerr << "[Video2D] Missing video file: " << videoPath << std::endl;
        return false;
    }

    video::DecoderInitParams params{};
    params.implementation = video::DecodeImplementation::Vulkan;
    if (engine)
    {
        video::VulkanInteropContext interop{};
        interop.instance = engine->instance;
        interop.physicalDevice = engine->physicalDevice;
        interop.device = engine->logicalDevice;
        interop.graphicsQueue = engine->getGraphicsQueue();
        interop.graphicsQueueFamilyIndex = engine->getGraphicsQueueFamilyIndex();
        interop.videoQueue = engine->getVideoQueue();
        interop.videoQueueFamilyIndex = engine->getVideoQueueFamilyIndex();
        params.vulkanInterop = interop;
    }
    if (!video::initializeVideoDecoder(videoPath, state.decoder, params))
    {
        std::cerr << "[Video2D] Vulkan decode unavailable, falling back to software." << std::endl;
        params.implementation = video::DecodeImplementation::Software;
        if (!video::initializeVideoDecoder(videoPath, state.decoder, params))
        {
            std::cerr << "[Video2D] Failed to initialize decoder" << std::endl;
            return false;
        }
    }
    if (state.decoder.implementation != video::DecodeImplementation::Vulkan)
    {
        std::cerr << "[Video2D] Warning: hardware Vulkan decode not active; using "
                  << state.decoder.implementationName << std::endl;
    }

    durationSeconds = 0.0;
    if (state.decoder.formatCtx && state.decoder.formatCtx->duration > 0)
    {
        durationSeconds = static_cast<double>(state.decoder.formatCtx->duration) / static_cast<double>(AV_TIME_BASE);
    }

    state.colorInfo = video::deriveVideoColorInfo(state.decoder);

    try
    {
        state.video.sampler = createLinearClampSampler(engine);
        state.overlay.sampler = createLinearClampSampler(engine);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[Video2D] Failed to create samplers: " << ex.what() << std::endl;
        video::cleanupVideoDecoder(state.decoder);
        return false;
    }

    video::DecodedFrame initialDecoded{};
    initialDecoded.buffer.assign(static_cast<size_t>(state.decoder.bufferSize), 0);
    if (state.decoder.outputFormat == PrimitiveYuvFormat::NV12)
    {
        const size_t yBytes = state.decoder.yPlaneBytes;
        if (yBytes > 0 && yBytes <= initialDecoded.buffer.size())
        {
            std::fill(initialDecoded.buffer.begin(), initialDecoded.buffer.begin() + yBytes, 0x80);
            std::fill(initialDecoded.buffer.begin() + yBytes, initialDecoded.buffer.end(), 0x80);
        }
    }
    else
    {
        const size_t yBytes = state.decoder.yPlaneBytes;
        const size_t uvBytes = state.decoder.uvPlaneBytes;
        const bool sixteenBit = state.decoder.bytesPerComponent > 1;
        if (sixteenBit)
        {
            const uint32_t bitDepth = state.decoder.bitDepth > 0 ? state.decoder.bitDepth : 8;
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

    if (!uploadDecodedFrame(state.video, engine, state.decoder, initialDecoded))
    {
        std::cerr << "[Video2D] Failed to upload initial frame." << std::endl;
        video::cleanupVideoDecoder(state.decoder);
        overlay::destroyImageResource(engine, state.video.lumaImage);
        overlay::destroyImageResource(engine, state.video.chromaImage);
        if (state.video.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, state.video.sampler, nullptr);
            state.video.sampler = VK_NULL_HANDLE;
        }
        if (state.overlay.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, state.overlay.sampler, nullptr);
            state.overlay.sampler = VK_NULL_HANDLE;
        }
        return false;
    }

    if (!video::startAsyncDecoding(state.decoder, 12))
    {
        std::cerr << "[Video2D] Failed to start async decoder" << std::endl;
        video::cleanupVideoDecoder(state.decoder);
        overlay::destroyImageResource(engine, state.video.lumaImage);
        overlay::destroyImageResource(engine, state.video.chromaImage);
        if (state.video.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, state.video.sampler, nullptr);
            state.video.sampler = VK_NULL_HANDLE;
        }
        if (state.overlay.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(engine->logicalDevice, state.overlay.sampler, nullptr);
            state.overlay.sampler = VK_NULL_HANDLE;
        }
        return false;
    }

    if (swapUvOverride.has_value())
    {
        state.decoder.swapChromaUV = swapUvOverride.value();
        std::cout << "[Video2D] Forcing UV swap to "
                  << (state.decoder.swapChromaUV ? "ON" : "OFF") << std::endl;
    }

    state.stagingFrame = video::DecodedFrame{};
    state.stagingFrame.buffer.reserve(static_cast<size_t>(state.decoder.bufferSize));
    state.pendingFrames.clear();
    state.playbackClockInitialized = false;
    state.lastDisplayedSeconds = 0.0;
    return true;
}

void pumpDecodedFrames(VideoPlaybackState &state)
{
    constexpr size_t kMaxPendingFrames = 6;
    while (state.pendingFrames.size() < kMaxPendingFrames &&
           video::acquireDecodedFrame(state.decoder, state.stagingFrame))
    {
        state.pendingFrames.emplace_back(std::move(state.stagingFrame));
        state.stagingFrame = video::DecodedFrame{};
        state.stagingFrame.buffer.reserve(static_cast<size_t>(state.decoder.bufferSize));
    }
}

double advancePlayback(VideoPlaybackState &state, bool playing)
{
    if (state.video.sampler == VK_NULL_HANDLE)
    {
        return 0.0;
    }

    pumpDecodedFrames(state);

    if (!playing)
    {
        return state.lastDisplayedSeconds;
    }

    if (state.pendingFrames.empty())
    {
        if (state.decoder.finished.load() && !state.decoder.threadRunning.load())
        {
            // End of video reached
        }
        return state.lastDisplayedSeconds;
    }

    auto currentTime = std::chrono::steady_clock::now();
    auto &nextFrame = state.pendingFrames.front();

    if (!state.playbackClockInitialized)
    {
        state.playbackClockInitialized = true;
        // Keep the visual scrub position continuous across seeks by anchoring the base
        // to the requested playback time (lastDisplayedSeconds) rather than resetting to zero.
        state.basePtsSeconds = nextFrame.ptsSeconds - state.lastDisplayedSeconds;
        state.lastFramePtsSeconds = nextFrame.ptsSeconds;
        state.lastFrameRenderTime = currentTime;
    }

    double frameDelta = nextFrame.ptsSeconds - state.lastFramePtsSeconds;
    if (frameDelta < 1e-6)
    {
        frameDelta = 1.0 / std::max(30.0, state.decoder.fps);
    }

    auto targetTime = state.lastFrameRenderTime + std::chrono::duration<double>(frameDelta);
    if (currentTime + std::chrono::milliseconds(1) < targetTime)
    {
        return state.lastDisplayedSeconds;
    }

    auto frame = std::move(nextFrame);
    state.pendingFrames.pop_front();

    if (!uploadDecodedFrame(state.video, state.engine, state.decoder, frame))
    {
        std::cerr << "[Video2D] Failed to upload decoded frame." << std::endl;
    }

    state.lastFramePtsSeconds = frame.ptsSeconds;
    state.lastFrameRenderTime = currentTime;
    state.lastDisplayedSeconds = std::max(0.0, state.lastFramePtsSeconds - state.basePtsSeconds);
    return state.lastDisplayedSeconds;
}
