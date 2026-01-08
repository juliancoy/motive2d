// motive2d.cpp
//
// Consumer side for FFmpeg Vulkan zero-copy decode + NV12->RGBA + ColorGrading (optional).
// - NO CPU copies of decoded frames.
// - Reads decoder-provided VkImageViews (Y + UV) directly.
// - Dispatches NV12->RGBA compute into a device-local RGBA8 image (owned by the pass).
// - Optionally dispatches ColorGrading (RGBA->RGBA) into another pass-owned output.
// - Publishes per-window PresentInput via Display2D::setPresentInput().
//
// Assumptions / requirements for correctness:
// 1) DecoderVulkan exposes the *current* VulkanSurface metadata for the latched frame.
//      bool getCurrentSurface(VulkanSurface& out) const;
//    advancePlayback() latches:
//      - externalLumaView / externalChromaView
//      - matching VulkanSurface snapshot (images/layouts/queueFamily)
// 2) Decoded VkImages must be usable as STORAGE_IMAGE (read-only) if your NV12 shader uses storage.
// 3) If decode happens on a separate queue family, ownership transfer must occur (we do it here).
// 4) We transition decode images to GENERAL for compute, then back to surf.layouts[] afterwards.

#include "motive2d.h"

#include "crop.h"
#include "decoder_vulkan.h"
#include "debug_logging.h"
#include "engine2d.h"
#include "fps.h"
#include "pose_overlay.h"
#include "scrubber.h"
#include "subtitle.h"
#include "utils.h"

// NV12->RGBA pass (pass-owned output)
#include "nv12_to_rgba.h"

// RGBA->RGBA pass (pass-owned output)
#include "color_grading_pass.h"

#include "display2d.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

std::mutex g_stageMutex;
std::condition_variable g_stageCv;

// ----------------------------------------
// Barrier helpers
// ----------------------------------------
static void imageBarrier(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t srcQF,
    uint32_t dstQF,
    VkPipelineStageFlags srcStage,
    VkAccessFlags srcAccess,
    VkPipelineStageFlags dstStage,
    VkAccessFlags dstAccess)
{
    if (image == VK_NULL_HANDLE) return;

    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;

    if (srcQF == dstQF) {
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    } else {
        b.srcQueueFamilyIndex = srcQF;
        b.dstQueueFamilyIndex = dstQF;
    }

    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static void makeReadableForStorageCompute(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    uint32_t srcQF,
    uint32_t dstQF)
{
    imageBarrier(
        cmd,
        image,
        oldLayout,
        VK_IMAGE_LAYOUT_GENERAL,
        srcQF,
        dstQF,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
}

static void makeReadableForSampling(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout)
{
    imageBarrier(
        cmd,
        image,
        oldLayout,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
}

static void makeReadableForSamplingCompute(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    uint32_t srcQF,
    uint32_t dstQF)
{
    imageBarrier(
        cmd,
        image,
        oldLayout,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        srcQF,
        dstQF,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
}

// ----------------------------------------
// Motive2D
// ----------------------------------------
Motive2D::Motive2D(CliOptions cliOptions)
    : options(cliOptions)
{
    if (options.debugLogging)
    {
        setDebugLoggingEnabled(true);
        setRenderDebugEnabled(true);
    }

    engine = new Engine2D();
    if (!engine || !engine->initialize())
        throw std::runtime_error("Failed to initialize Vulkan engine");

    const std::filesystem::path subtitlePath =
        cliOptions.videoPath.parent_path() / (cliOptions.videoPath.stem().string() + ".json");

    //subtitle    = new Subtitle(subtitlePath, engine);
    rectOverlay = new RectOverlay(engine);
    poseOverlay = new PoseOverlay(engine);
    crop        = new Crop();
    scrubber    = new Scrubber(engine);
    //fpsOverlay  = new FpsOverlay(engine);

    if (!cliOptions.gpuDecode)
        throw std::runtime_error("Only GPU decode is supported in this build");

    std::cout << "[Motive2D] GPU decode requested (Vulkan/FFmpeg)\n";

    decoder = new DecoderVulkan(cliOptions.videoPath, engine);
    if (!decoder || !decoder->valid)
        throw std::runtime_error("DecoderVulkan invalid: " + decoder->getHardwareInitFailureReason());

    // Start async decoding (producer). Decoder should internally cap (e.g. 10 frames).
    decoder->startAsyncDecoding(/*ignored or fixed internally*/);

    // Create windows
    if (options.showInput)
    {
        inputWindow = new Display2D(engine, 800, 600, "Input");
        windows.emplace_back(inputWindow);
        std::cout << "[Motive2D] Created input window\n";
    }
    if (options.showRegion)
    {
        regionWindow = new Display2D(engine, 800, 600, "Region");
        windows.emplace_back(regionWindow);
        std::cout << "[Motive2D] Created region window\n";
    }
    if (options.showGrading)
    {
        gradingWindow = new Display2D(engine, 800, 600, "Grading");
        windows.emplace_back(gradingWindow);
        std::cout << "[Motive2D] Created grading window\n";
        colorGrading = new ColorGrading(engine, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT));
    }

    if (options.pipelineTest)
        throw std::runtime_error("pipelineTest/export disabled in strict zero-copy build");

    // Create per-frame sync (cmd/fence/semaphore)
    createSynchronizationObjects();

    // Create pass-owned NV12->RGBA pipeline/output
    {
        const int w = decoder->getWidth();
        const int h = decoder->getHeight();

        // If your decode images are STORAGE_IMAGE readable, keep STORAGE_IMAGE.
        // Otherwise, switch this pass/shader to sampled inputs.
        nv12Pass = new Nv12ToRgbaPass(engine,
                                      static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                                      w, h,
                                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        nv12Pass->initialize();
    }
}

Motive2D::~Motive2D()
{
    destroySynchronizationObjects();

    delete colorGrading;
    colorGrading = nullptr;

    delete nv12Pass;
    nv12Pass = nullptr;

    //delete subtitle;
    delete rectOverlay;
    delete poseOverlay;
    delete crop;
    delete scrubber;
    //delete fpsOverlay;
    delete decoder;
    delete engine;
}

void Motive2D::createSynchronizationObjects()
{
    frames.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        FrameResources& fr = frames[i];

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = engine->renderDevice.getCommandPool();
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(engine->logicalDevice, &allocInfo, &fr.commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate command buffer for frame");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateFence(engine->logicalDevice, &fenceInfo, nullptr, &fr.fence) != VK_SUCCESS)
            throw std::runtime_error("Failed to create fence for frame");

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(engine->logicalDevice, &semInfo, nullptr, &fr.computeCompleteSemaphore) != VK_SUCCESS)
            throw std::runtime_error("Failed to create compute complete semaphore");
    }

    std::cout << "[Motive2D] Created synchronization objects for " << MAX_FRAMES_IN_FLIGHT << " frames\n";
}

void Motive2D::destroySynchronizationObjects()
{
    if (!engine) return;

    vkDeviceWaitIdle(engine->logicalDevice);

    for (auto& fr : frames)
    {
        if (fr.commandBuffer != VK_NULL_HANDLE)
            vkFreeCommandBuffers(engine->logicalDevice, engine->renderDevice.getCommandPool(), 1, &fr.commandBuffer);

        if (fr.fence != VK_NULL_HANDLE)
            vkDestroyFence(engine->logicalDevice, fr.fence, nullptr);

        if (fr.computeCompleteSemaphore != VK_NULL_HANDLE)
            vkDestroySemaphore(engine->logicalDevice, fr.computeCompleteSemaphore, nullptr);
    }

    frames.clear();
}

void Motive2D::recordComputeCommands(VkCommandBuffer cmd, int frameIndex, const VulkanSurface& surf)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin command buffer");

    const uint32_t gfxQF = engine->graphicsQueueFamilyIndex;

    // ---- Transition decode images to SHADER_READ_ONLY_OPTIMAL for sampled image reads (and queue-family transfer if needed) ----
    if (surf.valid && surf.planes >= 1 && surf.images[0] != VK_NULL_HANDLE)
        makeReadableForSamplingCompute(cmd, surf.images[0], surf.layouts[0], surf.queueFamily[0], gfxQF);

    if (surf.valid && surf.planes >= 2 && surf.images[1] != VK_NULL_HANDLE)
        makeReadableForSamplingCompute(cmd, surf.images[1], surf.layouts[1], surf.queueFamily[1], gfxQF);

    // ---- NV12 -> RGBA (pass-owned output) ----
    {
        // Set inputs only if the view handles changed (avoid churn).
        static VkImageView lastY = VK_NULL_HANDLE;
        static VkImageView lastUV = VK_NULL_HANDLE;

        VkImageView yView  = decoder->externalLumaView;
        VkImageView uvView = decoder->externalChromaView;

        if (yView != lastY || uvView != lastUV)
        {
            lastY = yView;
            lastUV = uvView;
            nv12Pass->setInputNV12(yView, uvView);
        }

        nv12Pass->pushConstants.rgbaSize = glm::ivec2(decoder->getWidth(), decoder->getHeight());
        nv12Pass->pushConstants.uvSize   = glm::ivec2(decoder->getWidth() / 2, decoder->getHeight() / 2);
        nv12Pass->pushConstants.colorSpace = 0;
        nv12Pass->pushConstants.colorRange = 1;

        nv12Pass->dispatch(cmd, static_cast<uint32_t>(frameIndex));
    }

    // ---- Make NV12->RGBA output readable for sampling (ColorGrading + common presenters) ----
    // Nv12ToRgbaPass leaves output in GENERAL; convert to SHADER_READ for downstream sampling.
    makeReadableForSampling(cmd,
                            nv12Pass->outputImage(static_cast<uint32_t>(frameIndex)),
                            VK_IMAGE_LAYOUT_GENERAL);

    // ---- Optional: Color grading (RGBA sampled in -> storage out) ----
    if (colorGrading)
    {
        colorGrading->setInputRGBA(nv12Pass->outputView(static_cast<uint32_t>(frameIndex)),
                                   nv12Pass->outputSampler());

        // ColorGrading internally transitions its own output to GENERAL for writes.
        colorGrading->dispatch(cmd, static_cast<uint32_t>(frameIndex));
    }

    // ---- Transition decode images back to original layout (best-effort) ----
    if (surf.valid && surf.planes >= 1 && surf.images[0] != VK_NULL_HANDLE)
    {
        imageBarrier(cmd,
                     surf.images[0],
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     surf.layouts[0],
                     gfxQF,
                     surf.queueFamily[0],
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                     VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }
    if (surf.valid && surf.planes >= 2 && surf.images[1] != VK_NULL_HANDLE)
    {
        imageBarrier(cmd,
                     surf.images[1],
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     surf.layouts[1],
                     gfxQF,
                     surf.queueFamily[1],
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                     VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("Failed to end command buffer");
}

void Motive2D::run()
{
    int iteration = 0;
    while (!windows.empty())
    {
        FrameResources& fr = frames[currentFrame];

        // Wait for previous GPU work for this in-flight slot
        vkWaitForFences(engine->logicalDevice, 1, &fr.fence, VK_TRUE, UINT64_MAX);

        // Tick decoder: latch a frame + external views + current surface
        decoder->advancePlayback();

        if (decoder->externalLumaView == VK_NULL_HANDLE || decoder->externalChromaView == VK_NULL_HANDLE)
        {
            if (iteration % 100 == 0) {
                std::cout << "[Motive2D] Waiting for decoder frames... (iteration " << iteration << ")\n";
            }
            glfwPollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            iteration++;
            continue;
        }

        // We have valid views, so we'll submit work - reset the fence now
        vkResetFences(engine->logicalDevice, 1, &fr.fence);

        VulkanSurface surf{};
        if (!decoder->getCurrentSurface(surf) || !surf.valid)
        {
            throw std::runtime_error(
                "Decoder has views but no current VulkanSurface metadata (need getCurrentSurface())");
        }

        // Record compute (decode barriers + nv12->rgba + optional grading)
        recordComputeCommands(fr.commandBuffer, currentFrame, surf);

        // Decide what each window presents this frame.
        // Input/Region show NV12->RGBA output (pre-grading).
        // Grading window shows ColorGrading output if enabled, else same as input.
        if (inputWindow)
        {
            PresentInput in = nv12Pass->output(static_cast<uint32_t>(currentFrame));
            in.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // we transitioned it
            inputWindow->setPresentInput(in);
        }

        if (regionWindow)
        {
            PresentInput in = nv12Pass->output(static_cast<uint32_t>(currentFrame));
            in.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // we transitioned it
            regionWindow->setPresentInput(in);
        }

        if (gradingWindow)
        {
            if (colorGrading)
            {
                // ColorGrading::dispatch already published to its Display2D (gradingWindow)
                // via publishOutputToDisplay(), but it's harmless to be explicit:
                ColorGrading::Output out = colorGrading->output(static_cast<uint32_t>(currentFrame));
                PresentInput in{};
                in.image  = out.image;
                in.view   = out.view;
                in.layout = out.layout;
                in.extent = out.extent;
                in.format = out.format;
                gradingWindow->setPresentInput(in);
            }
            else
            {
                PresentInput in = nv12Pass->output(static_cast<uint32_t>(currentFrame));
                in.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                gradingWindow->setPresentInput(in);
            }
        }

        // Submit compute (same queue submission order is enough if Display2D submits later on same queue)
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &fr.commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &fr.computeCompleteSemaphore;

        if (vkQueueSubmit(engine->graphicsQueue, 1, &submitInfo, fr.fence) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit compute queue");

        // Render all windows (each will acquire swapchain image + record presenter work)
        for (auto& w : windows)
            w->renderFrame(fr.computeCompleteSemaphore, VK_PIPELINE_STAGE_TRANSFER_BIT);


        glfwPollEvents();

        bool anyOpen = false;
        for (auto& w : windows)
        {
            if (!w->shouldClose())
            {
                anyOpen = true;
                break;
            }
        }
        if (!anyOpen) break;

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
