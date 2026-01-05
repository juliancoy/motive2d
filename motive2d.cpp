#include "debug_logging.h"
#include "engine2d.h"
#include "motive2d.h"
#include "text.h"
#include "video.h"
#include "subtitle.h"

#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    double g_scrollDelta = 0.0;
}

static void onScroll(GLFWwindow *, double, double yoffset)
{
    g_scrollDelta += yoffset;
}

Motive2D::Motive2D(CliOptions cli)
{
    setRenderDebugEnabled(cli.debugLogging);
    if (cli.debugLogging)
    {
        setDebugLoggingEnabled(true);
    }
    Engine2D engine; // this is kiond of like the context

    Subtitle subtitle = Subtitle();
    Display2D *inputWindow = nullptr;
    Display2D *regionWindow = nullptr;
    Display2D *gradingWindow = nullptr;
    PoseOverlay poseOverlay(cli.videoPath);

    RectOverlay rectOverlay = RectOverlay(&engine);
    PoseOverlay poseOverlay = PoseOverlay(&engine);
    ImageResource blackLuma(&engine, &luma, sizeof(luma), 1, 1, VK_FORMAT_R8_UNORM);
    ImageResource blackChroma(&engine, chroma, sizeof(chroma), 1, 1, VK_FORMAT_R8G8_UNORM);
    VkSampler blackSampler = VK_NULL_HANDLE;
    VideoImageSet blackVideo{};

    bool subtitleLoaded = false;

    double durationSeconds = 0.0;
    auto decoder = Decoder(filePath, this, durationSeconds);

    videoLoaded = true;
    duration = static_cast<float>(durationSeconds);
    currentTime = 0.0f;
    playing = true;

    std::cout << "[Moteve2d] Loaded video: " << filePath << "\n";
    std::cout << "  Resolution: " << playbackState.decoder.width << "x" << playbackState.decoder.height << "\n";
    std::cout << "  Framerate: " << playbackState.decoder.fps << " fps\n";
    std::cout << "  Duration: " << duration << " seconds\n";

    const std::filesystem::path subtitlePath =
        cli.videoPath.parent_path() / (cli.videoPath.stem().string() + ".json");
    if (std::filesystem::exists(subtitlePath))
    {
        subtitleLoaded = subtitle.load(subtitlePath);
        if (!subtitleLoaded)
        {
            std::cout << "[subtitle] Failed to load subtitles from " << subtitlePath << "\n";
        }
    }

    detectionEntries.reserve(64);
    poseObjects.reserve(8);
    double lastDetectedPosePts = -1.0;
    bool detectionTogglePrev = false;

    engine.setDecodeDebugEnabled(cli.debugDecode);

    if (cli.showInput)
    {
        inputWindow = engine.createWindow(1280, 720, "Input");
        if (!inputWindow)
        {
            std::cerr << "[motive2d] Failed to create main window.\n";
            return 1;
        }
        glfwSetScrollCallback(inputWindow->window, onScroll);
        inputWindow->setScrubberPassEnabled(cli.scrubberEnabled);
        if (cli.skipBlit)
        {
            inputWindow->setVideoPassEnabled(false);
            inputWindow->setPassthroughPassEnabled(true);
        }

        auto window = std::make_unique<Display2D>(this, width, height, title);
        Display2D* ptr = window.get();
        windows.push_back(std::move(window));
        std::cout << "[Engine2D] Created window: " << title
                  << " (" << width << "x" << height << ")\n";
    }
    if (cli.showRegion)
    {
        regionWindow = engine.createWindow(360, 640, "Region View");
        if (!regionWindow)
        {
            std::cerr << "[motive2d] Failed to create region window.\n";
            return 1;
        }
        regionWindow->setScrubberPassEnabled(cli.scrubberEnabled);
        if (cli.skipBlit)
        {
            regionWindow->setVideoPassEnabled(false);
            regionWindow->setPassthroughPassEnabled(true);
        }
    }
    if (cli.showGrading)
    {
        gradingWindow = engine.createWindow(420, 880, "Grading");
        if (!gradingWindow)
        {
            std::cerr << "[motive2d] Failed to create grading window.\n";
            return 1;
        }
        gradingWindow->setScrubberPassEnabled(cli.scrubberEnabled);
        if (cli.skipBlit)
        {
            gradingWindow->setVideoPassEnabled(false);
            gradingWindow->setPassthroughPassEnabled(true);
        }
    }

    uint8_t luma = 0;
    uint8_t chroma[2] = {128, 128};
    try
    {
        blackSampler = decoder.createLinearClampSampler(&engine);
    }
    catch (const std::exception &)
    {
        blackSampler = playbackState.overlay.sampler;
    }
    blackVideo.width = 1;
    blackVideo.height = 1;
    blackVideo.chromaDivX = 1;
    blackVideo.chromaDivY = 1;
    blackVideo.luma.view = blackLuma.view;
    blackVideo.luma.sampler = blackSampler;
    blackVideo.chroma.view = blackChroma.view;
    blackVideo.chroma.sampler = blackSampler;

    bool playing = true;
    bool spaceHeld = false;
    bool mouseHeld = false;
    bool scrubDragging = false;
    double scrubDragStartX = 0.0;
    float scrubDragStartProgress = 0.0f;
    float scrubProgressUi = 0.0f;
    glm::vec2 rectCenter(640.0f, 360.0f);
    float rectHeight = 360.0f;
    float rectWidth = rectHeight * (9.0f / 16.0f);
    float windowWidth = 1280.0f;
    float windowHeight = 720.0f;
    GradingSettings gradingSettings{};
    grading::setGradingDefaults(gradingSettings);
    // Try to load saved grading settings at startup
    if (!grading::loadGradingSettings("blit_settings.json", gradingSettings))
    {
        std::cout << "[motive2d] Using default grading settings (blit_settings.json not found or invalid)\n";
    }
    else
    {
        std::cout << "[motive2d] Loaded grading settings from blit_settings.json\n";
    }
    std::array<float, kCurveLutSize> curveLut{};
    bool curveDirty = true;
    bool gradingOverlayDirty = true;
    uint32_t gradingFbWidth = 0;
    uint32_t gradingFbHeight = 0;
    bool gradingMouseHeld = false;
    bool gradingRightHeld = false;
    grading::SliderLayout gradingLayout{};
    bool gradingPreviewEnabled = true; // Enable grading preview by default
    bool detectionEnabled = false;

    auto runGradingClick = [&](bool rightClick)
    {
        if (!gradingWindow)
        {
            return;
        }
        double gx = 0.0;
        double gy = 0.0;
        glfwGetCursorPos(gradingWindow->window, &gx, &gy);
        bool loadRequested = false;
        bool saveRequested = false;
        bool previewToggle = false;
        bool detectionToggle = false;
        if (grading::handleOverlayClick(gradingLayout,
                                        gx,
                                        gy,
                                        gradingSettings,
                                        /*doubleClick=*/false,
                                        rightClick,
                                        &loadRequested,
                                        &saveRequested,
                                        &previewToggle,
                                        &detectionToggle))
        {
            gradingOverlayDirty = true;
            curveDirty = true;
            if (loadRequested)
            {
                if (!grading::loadGradingSettings("blit_settings.json", gradingSettings))
                {
                    std::cerr << "[motive2d] Failed to load grading settings.\n";
                }
            }
            if (saveRequested)
            {
                if (!grading::saveGradingSettings("blit_settings.json", gradingSettings))
                {
                    std::cerr << "[motive2d] Failed to save grading settings.\n";
                }
            }
            if (previewToggle)
            {
                gradingPreviewEnabled = !gradingPreviewEnabled;
            }
            if (detectionToggle)
            {
                detectionEnabled = !detectionEnabled;
            }
        }
    };

    colorGrading.destroyCurveResources();

    ~scrubber();
    scrubPipeline = VK_NULL_HANDLE;

    // Descriptor set layout
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    // 0: swapchain storage image
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 1: overlay (rectangle)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 2: fps overlay (text)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 3: luma
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 4: chroma
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 5: curve LUT UBO
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(engine->logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor set layout for Display2D");
    }

    // Pipeline layout with push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(CropPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(engine->logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout for Display2D");
    }

    // Compute pipeline
    colorGrading.destroyPipeline();
    colorGrading.createPipeline(pipelineLayout);

    colorGrading.createCurveResources();


    // Descriptor pool and sets (one per swapchain image)
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(swapchainImages.size());
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(swapchainImages.size()) * 4;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = static_cast<uint32_t>(swapchainImages.size());

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(swapchainImages.size());

    if (vkCreateDescriptorPool(engine->logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor pool for Display2D");
    }

    std::vector<VkDescriptorSetLayout> layouts(swapchainImages.size(), descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(layouts.size());
    if (vkAllocateDescriptorSets(engine->logicalDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate descriptor sets for Display2D");
    }

    try
    {
        scrubPipeline = Scrubber(engine);
    }
    catch (...)
    {
        colorGrading.destroyPipeline();
        throw;
    }

    if (renderDebugEnabled())
    {
        LOG_DEBUG(std::cout << "[Display2D] renderFrame image=" << imageIndex
                  << " videoSize=" << videoImages.width << "x" << videoImages.height
                  << " swapchainExtent=" << swapchainExtent.width << "x" << swapchainExtent.height
                  << " reservedHeight=" << reservedHeight << " availableHeight=" << availableHeight
                  << " videoAspect=" << videoAspect << " outputAspect=" << outputAspect
                  << " targetOrigin=(" << cropPushConstants.targetOrigin.x << "," << cropPushConstants.targetOrigin.y << ")"
                  << " targetSize=" << cropPushConstants.targetSize.x << "x" << cropPushConstants.targetSize.y
                  << " overlayEnabled=" << overlayValid
                  << " fpsOverlayEnabled=" << fpsOverlayValid
                  << " overlaySize=" << cropPushConstants.overlaySize.x << "x" << cropPushConstants.overlaySize.y
                  << " fpsSize=" << cropPushConstants.fpsOverlaySize.x << "x" << cropPushConstants.fpsOverlaySize.y
                  << std::endl);
    }

    bool shouldExit = false;
    bool frameCaptured = false;
    while (!shouldExit)
    {
        if (inputWindow)
        {
            inputWindow->pollEvents();
        }
        if (regionWindow)
        {
            regionWindow->pollEvents();
        }
        if (gradingWindow)
        {
            gradingWindow->pollEvents();
        }

        if ((inputWindow && inputWindow->shouldClose()) ||
            (regionWindow && regionWindow->shouldClose()) ||
            (gradingWindow && gradingWindow->shouldClose()))
        {
            break;
        }

        // Single frame capture logic
        if (cli.singleFrame && !frameCaptured)
        {
            const auto &decoder = playbackState.decoder;
            if (!playbackState.pendingFrames.empty())
            {
                const auto &frame = playbackState.pendingFrames.front();
                const size_t requiredBytes = static_cast<size_t>(decoder.yPlaneBytes) + static_cast<size_t>(decoder.uvPlaneBytes);
                if (frame.buffer.size() >= requiredBytes)
                {
                    // Save raw frame data
                    std::filesystem::path rawPath = cli.outputImagePath;
                    rawPath.replace_extension(".raw");
                    if (saveRawFrameData(rawPath, frame.buffer.data(), frame.buffer.size()))
                    {
                        std::cout << "[motive2d] Saved raw frame data to: " << rawPath << "\n";
                    }

                    // Save PPM image (RGB conversion)
                    std::filesystem::path ppmPath = cli.outputImagePath;
                    ppmPath.replace_extension(".ppm");
                    if (savePpmFromYuv(ppmPath,
                                       frame.buffer.data(),
                                       decoder.width,
                                       decoder.height,
                                       decoder.bytesPerComponent,
                                       decoder.yPlaneBytes,
                                       decoder.uvPlaneBytes))
                    {
                        std::cout << "[motive2d] Saved PPM image to: " << ppmPath << "\n";
                    }

                    // Also save as text file with metadata
                    std::filesystem::path metaPath = cli.outputImagePath;
                    metaPath.replace_extension(".txt");
                    std::ofstream meta(metaPath);
                    if (meta)
                    {
                        meta << "Frame metadata:\n";
                        meta << "  Width: " << decoder.width << "\n";
                        meta << "  Height: " << decoder.height << "\n";
                        meta << "  Format: " << static_cast<int>(decoder.outputFormat) << "\n";
                        meta << "  Bytes per component: " << decoder.bytesPerComponent << "\n";
                        meta << "  Y plane bytes: " << decoder.yPlaneBytes << "\n";
                        meta << "  UV plane bytes: " << decoder.uvPlaneBytes << "\n";
                        meta << "  Total frame size: " << frame.buffer.size() << "\n";
                        meta << "  PTS: " << frame.ptsSeconds << "\n";
                        meta.close();
                        std::cout << "[motive2d] Saved frame metadata to: " << metaPath << "\n";
                    }

                    frameCaptured = true;
                    shouldExit = true;
                    std::cout << "[motive2d] Single frame captured and saved. Exiting.\n";
                    break;
                }
            }
        }

        if (inputWindow)
        {
            int mouseState = glfwGetMouseButton(inputWindow->window, GLFW_MOUSE_BUTTON_LEFT);
            if (mouseState == GLFW_PRESS && !mouseHeld)
            {
                double cursorX = 0.0;
                double cursorY = 0.0;
                glfwGetCursorPos(inputWindow->window, &cursorX, &cursorY);
                // Scale mouse coordinates from window to framebuffer for hit testing
                double scaleX = static_cast<double>(windowWidth) / static_cast<double>(inputWindow->width);
                double scaleY = static_cast<double>(windowHeight) / static_cast<double>(inputWindow->height);
                double scaledCursorX = cursorX * scaleX;
                double scaledCursorY = cursorY * scaleY;

                if (cursorInPlayButton(scaledCursorX, scaledCursorY, static_cast<int>(windowWidth), static_cast<int>(windowHeight)))
                {
                    playing = !playing;
                }
                else if (cursorInScrubber(scaledCursorX, scaledCursorY, static_cast<int>(windowWidth), static_cast<int>(windowHeight)))
                {
                    scrubDragging = true;
                    scrubDragStartX = cursorX;
                    scrubDragStartProgress = scrubProgressUi;
                    mouseHeld = true;
                    playing = false;
                }
                else
                {
                    glm::vec2 scale(static_cast<float>(scaleX), static_cast<float>(scaleY));
                    rectCenter = glm::vec2(static_cast<float>(cursorX) * scale.x,
                                           static_cast<float>(cursorY) * scale.y);
                }
            }
            else if (mouseState == GLFW_RELEASE)
            {
                if (scrubDragging)
                {
                    // Scale mouse coordinates from window to framebuffer
                    double x = 0.0;
                    glfwGetCursorPos(inputWindow->window, &x, nullptr);
                    // Convert window coordinates to framebuffer coordinates
                    double scaleX = static_cast<double>(windowWidth) / static_cast<double>(inputWindow->width);
                    x *= scaleX;

                    const ScrubberUi ui = computeScrubberUi(static_cast<int>(windowWidth), static_cast<int>(windowHeight));
                    double progress = (x - ui.left) / (ui.right - ui.left);
                    progress = std::clamp(progress, 0.0, 1.0);
                    scrubProgressUi = static_cast<float>(progress);
                    const float seekTime = static_cast<float>(progress * engine.getDuration());
                    engine.seek(seekTime);
                    playbackState.lastDisplayedSeconds = seekTime;
                    scrubDragging = false;
                    playing = true;
                }
                mouseHeld = false;
            }
        }

        // Update scrub progress during dragging
        if (scrubDragging && inputWindow)
        {
            // Scale mouse coordinates from window to framebuffer
            double x = 0.0;
            glfwGetCursorPos(inputWindow->window, &x, nullptr);
            // Convert window coordinates to framebuffer coordinates
            double scaleX = static_cast<double>(windowWidth) / static_cast<double>(inputWindow->width);
            x *= scaleX;

            const ScrubberUi ui = computeScrubberUi(static_cast<int>(windowWidth), static_cast<int>(windowHeight));
            double progress = (x - ui.left) / (ui.right - ui.left);
            progress = std::clamp(progress, 0.0, 1.0);
            scrubProgressUi = static_cast<float>(progress);
        }

        double scrollDelta = g_scrollDelta;
        g_scrollDelta = 0.0;
        if (std::abs(scrollDelta) > 0.0)
        {
            float scale = 1.0f + static_cast<float>(scrollDelta) * 0.05f;
            rectHeight = std::clamp(rectHeight * scale, 50.0f, windowHeight);
            rectWidth = rectHeight * (9.0f / 16.0f);
        }

        if (gradingWindow)
        {
            int gradingMouseState = glfwGetMouseButton(gradingWindow->window, GLFW_MOUSE_BUTTON_LEFT);
            if (gradingMouseState == GLFW_PRESS && !gradingMouseHeld)
            {
                gradingMouseHeld = true;
                runGradingClick(/*rightClick=*/false);
            }
            else if (gradingMouseState == GLFW_RELEASE)
            {
                gradingMouseHeld = false;
            }

            int gradingRightState = glfwGetMouseButton(gradingWindow->window, GLFW_MOUSE_BUTTON_RIGHT);
            if (gradingRightState == GLFW_PRESS && !gradingRightHeld)
            {
                gradingRightHeld = true;
                runGradingClick(/*rightClick=*/true);
            }
            else if (gradingRightState == GLFW_RELEASE)
            {
                gradingRightHeld = false;
            }
        }

        if (inputWindow)
        {
            if (glfwGetKey(inputWindow->window, GLFW_KEY_SPACE) == GLFW_PRESS && !spaceHeld)
            {
                playing = !playing;
            }
            spaceHeld = glfwGetKey(inputWindow->window, GLFW_KEY_SPACE) == GLFW_PRESS;
        }

        detectionTogglePrev = detectionEnabled;

        double playbackSeconds = advancePlayback(playbackState, playing && !scrubDragging);
        engine.setCurrentTime(static_cast<float>(playbackSeconds));
        const float totalDuration = engine.getDuration();
        // Only update scrub progress from playback when not dragging
        if (totalDuration > 0.0f && !scrubDragging)
        {
            scrubProgressUi = static_cast<float>(playbackSeconds / totalDuration);
        }

        uint32_t fbWidth = 0;
        uint32_t fbHeight = 0;
        if (inputWindow)
        {
            int fbWidthInt = 0;
            int fbHeightInt = 0;
            glfwGetFramebufferSize(inputWindow->window, &fbWidthInt, &fbHeightInt);
            fbWidth = static_cast<uint32_t>(std::max(1, fbWidthInt));
            fbHeight = static_cast<uint32_t>(std::max(1, fbHeightInt));
            windowWidth = static_cast<float>(fbWidth);
            windowHeight = static_cast<float>(fbHeight);
        }

        const uint32_t currentFrameIndex = getFrameIndex(playbackSeconds, playbackState.decoder.fps);
        const auto &savedOverlayEntries = poseOverlay.entriesForFrame(currentFrameIndex);
        const bool hasSavedOverlay = !savedOverlayEntries.empty();
        const DetectionEntry *savedOverlayData = hasSavedOverlay ? savedOverlayEntries.data() : nullptr;

        glm::vec2 overlayRectCenter = rectCenter;
        glm::vec2 overlayRectSize(rectWidth, rectHeight);
        float overlayOuterThickness = 3.0f;
        float overlayInnerThickness = 3.0f;
        bool overlayActive = true;

        uint32_t overlayCount = 0;
        const DetectionEntry *overlaySource = nullptr;
        if (!detectionEntries.empty())
        {
            overlaySource = detectionEntries.data();
            overlayCount = static_cast<uint32_t>(detectionEntries.size());
        }
        else if (hasSavedOverlay)
        {
            overlaySource = savedOverlayData;
            overlayCount = static_cast<uint32_t>(savedOverlayEntries.size());
        }

        const float poseOverlayEnabled = overlayCount > 0 ? 1.0f : 0.0f;
        // Run pose overlay directly to the main overlay image
        poseOverlay.run(&engine,
                              poseOverlayCompute,
                              playbackState.overlay.image,
                              fbWidth,
                              fbHeight,
                              overlayRectCenter,
                              overlayRectSize,
                              overlayOuterThickness,
                              overlayInnerThickness,
                              poseOverlayEnabled,
                              overlaySource,
                              overlayCount);
        const float rectDetectionFlag = detectionEnabled ? 1.0f : 0.0f;
        // Run rectangle overlay reading from and writing to the same image
        rectOverlay.run(
                              playbackState.overlay.image,
                              playbackState.overlay.image,
                              fbWidth,
                              fbHeight,
                              overlayRectCenter,
                              overlayRectSize,
                              overlayOuterThickness,
                              overlayInnerThickness,
                              rectDetectionFlag,
                              overlayActive ? 1.0f : 0.0f);
        // Always update curve LUT when dirty, regardless of preview state
        if (curveDirty)
        {
            grading::buildCurveLut(gradingSettings, curveLut);
            curveDirty = false;
        }

        ColorAdjustments adjustments{};
        if (gradingPreviewEnabled)
        {
            adjustments.exposure = gradingSettings.exposure;
            adjustments.contrast = gradingSettings.contrast;
            adjustments.saturation = gradingSettings.saturation;
            adjustments.shadows = gradingSettings.shadows;
            adjustments.midtones = gradingSettings.midtones;
            adjustments.highlights = gradingSettings.highlights;
            adjustments.curveLut = curveLut;
            adjustments.curveEnabled = true;
        }
        double regionWindowWidth = regionWindow ? regionWindow->width : 0;
        double regionWindowHeight = regionWindow ? regionWindow->height : 0;
        RenderOverrides regionOverrides;
        if (regionWindow && fbWidth > 0 && fbHeight > 0)
        {
            const float vidW = static_cast<float>(playbackState.video.descriptors.width);
            const float vidH = static_cast<float>(playbackState.video.descriptors.height);
            const float outputAspect = windowWidth / windowHeight;
            const float videoAspect = vidH > 0.0f ? vidW / vidH : 1.0f;

            float targetW = windowWidth;
            float targetH = windowHeight;
            if (videoAspect > outputAspect)
            {
                targetH = targetW / videoAspect;
            }
            else
            {
                targetW = targetH * videoAspect;
            }
            const float targetX = (windowWidth - targetW) * 0.5f;
            const float targetY = (windowHeight - targetH) * 0.5f;

            const float rectLeft = rectCenter.x - rectWidth * 0.5f;
            const float rectRight = rectCenter.x + rectWidth * 0.5f;
            const float rectTop = rectCenter.y - rectHeight * 0.5f;
            const float rectBottom = rectCenter.y + rectHeight * 0.5f;

            const float cropLeft = std::clamp(rectLeft, targetX, targetX + targetW);
            const float cropRight = std::clamp(rectRight, targetX, targetX + targetW);
            const float cropTop = std::clamp(rectTop, targetY, targetY + targetH);
            const float cropBottom = std::clamp(rectBottom, targetY, targetY + targetH);

            const float cropW = std::max(0.0f, cropRight - cropLeft);
            const float cropH = std::max(0.0f, cropBottom - cropTop);
            if (cropW > 1.0f && cropH > 1.0f)
            {
                const float u0 = (cropLeft - targetX) / targetW;
                const float v0 = (cropTop - targetY) / targetH;
                const float u1 = (cropRight - targetX) / targetW;
                const float v1 = (cropBottom - targetY) / targetH;

                regionOverrides.useTargetOverride = true;
                regionOverrides.targetOrigin = glm::vec2(0.0f, 0.0f);
                regionOverrides.targetSize = glm::vec2(static_cast<float>(regionWindowWidth),
                                                       static_cast<float>(regionWindowHeight));
                regionOverrides.useCrop = true;
                regionOverrides.cropOrigin = glm::vec2(u0, v0);
                regionOverrides.cropSize = glm::vec2(u1 - u0, v1 - v0);
            }
        }

        bool sizeChanged = false;
        uint32_t gradingWindowFbW = 0;
        uint32_t gradingWindowFbH = 0;
        if (gradingWindow)
        {
            int gradingWidthInt = 0;
            int gradingHeightInt = 0;
            glfwGetFramebufferSize(gradingWindow->window, &gradingWidthInt, &gradingHeightInt);
            gradingWindowFbW = static_cast<uint32_t>(std::max(1, gradingWidthInt));
            gradingWindowFbH = static_cast<uint32_t>(std::max(1, gradingHeightInt));
            if (gradingWindowFbW != gradingFbWidth || gradingWindowFbH != gradingFbHeight)
            {
                gradingFbWidth = gradingWindowFbW;
                gradingFbHeight = gradingWindowFbH;
                sizeChanged = true;
            }
        }
        bool rebuildGradingOverlay = gradingOverlayDirty || sizeChanged;
        if (gradingWindow && rebuildGradingOverlay)
        {
            bool success = grading::buildGradingOverlay(&engine,
                                                        gradingSettings,
                                                        gradingOverlayImage,
                                                        gradingOverlayInfo,
                                                        gradingFbWidth,
                                                        gradingFbHeight,
                                                        gradingLayout,
                                                        gradingPreviewEnabled,
                                                        detectionEnabled);
            if (success)
            {
                gradingOverlayInfo.overlay.sampler = playbackState.overlay.sampler;
                gradingOverlayDirty = false;
            }
            else
            {
                gradingOverlayDirty = true;
            }
        }

        engine.refreshFpsOverlay();

        bool subtitleRendered = false;
        if (subtitleLoaded && cli.overlaysEnabled)
        {
            subtitleRendered = updateSubtitle(&engine,
                                              subtitleResources,
                                              subtitle,
                                              playbackSeconds,
                                              fbWidth,
                                              fbHeight,
                                              overlayRectCenter,
                                              overlayRectSize,
                                              playbackState.overlay.image,
                                              playbackState.overlay.sampler,
                                              playbackState.overlay.sampler,
                                              2,
                                              cli.subtitleBackground);
            if (subtitleRendered)
            {
                LOG_DEBUG(std::cout << "[motive2d] Subtitle overlay blended onto rectangle image" << std::endl);
            }
        }

        // Point overlay info at the single, final frame that has received every overlay pass.
        playbackState.overlay.info.overlay.view = playbackState.overlay.image.view;
        playbackState.overlay.info.overlay.sampler = playbackState.overlay.sampler;
        playbackState.overlay.info.extent = {fbWidth, fbHeight};
        playbackState.overlay.info.offset = {0, 0};
        playbackState.overlay.info.enabled = true;

        const ColorAdjustments *adjustmentsPtr = gradingPreviewEnabled ? &adjustments : nullptr;
        if (gradingPreviewEnabled)
        {
            LOG_DEBUG(std::cout << "[motive2d] Applying grading: exposure=" << adjustments.exposure
                                << " contrast=" << adjustments.contrast
                                << " saturation=" << adjustments.saturation
                                << " preview=" << gradingPreviewEnabled << std::endl);
        }
        else
        {
            LOG_DEBUG(std::cout << "[motive2d] Grading preview OFF - using neutral values" << std::endl);
        }
        const OverlayImageInfo &fpsOverlayInfoToUse = playbackState.fpsOverlay.info;
        if (inputWindow)
        {
            inputWindow->renderFrame(playbackState.video.descriptors,
                                     playbackState.overlay.info,
                                     fpsOverlayInfoToUse,
                                     playbackState.colorInfo,
                                     scrubProgressUi,
                                     playing ? 1.0f : 0.0f,
                                     nullptr,
                                     adjustmentsPtr);
        }

        if (regionWindow)
        {
            OverlayImageInfo disabledOverlay{};
            OverlayImageInfo disabledFps{};
            regionWindow->renderFrame(playbackState.video.descriptors,
                                      disabledOverlay,
                                      disabledFps,
                                      playbackState.colorInfo,
                                      0.0f,
                                      0.0f,
                                      &regionOverrides,
                                      adjustmentsPtr);
        }

        if (gradingWindow)
        {
            RenderOverrides gradingOverrides;
            OverlayImageInfo disabledFps{};
            gradingWindow->renderFrame(blackVideo,
                                       gradingOverlayInfo,
                                       disabledFps,
                                       playbackState.colorInfo,
                                       0.0f,
                                       0.0f,
                                       &gradingOverrides,
                                       adjustmentsPtr);
        }

        if (inputWindow == nullptr)
        {
            shouldExit = true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

}
Motive2D::~Motive2D()
{
    destroyImageResource(&engine, gradingOverlayImage);
    destroySubtitleResources(&engine, subtitleResources);
    destroyImageResource(&engine, blackLuma);
    destroyImageResource(&engine, blackChroma);
    if (blackSampler != VK_NULL_HANDLE && blackSampler != playbackState.overlay.sampler)
    {
        vkDestroySampler(engine.logicalDevice, blackSampler, nullptr);
    }

    engine.shutdown();
    return 0;
}

void Motive2D::renderFrame()
{
    if (renderDebugEnabled())
    {
        std::cout << "[Display2D] renderFrame: early return - invalid resources" << std::endl;
        std::cout << "[Display2D]   swapchainImages.empty(): " << swapchainImages.empty() << std::endl;
        std::cout << "[Display2D]   videoImages.luma.view: " << videoImages.luma.view << std::endl;
        std::cout << "[Display2D]   videoImages.luma.sampler: " << videoImages.luma.sampler << std::endl;
        std::cout << "[Display2D]   videoImages.width: " << videoImages.width << " height: " << videoImages.height << std::endl;
        std::cout << "[Display2D]   gradingImages.size(): " << colorGrading.gradingImages.size() << " swapchainImages.size(): " << swapchainImages.size() << std::endl;
    }
    return;
    colorGrading.applyCurve();
    vkWaitForFences(engine->logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(engine->logicalDevice,
                                             swapchain,
                                             UINT64_MAX,
                                             imageAvailableSemaphores[currentFrame],
                                             VK_NULL_HANDLE,
                                             &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
        return;
    }

    vkResetFences(engine->logicalDevice, 1, &inFlightFences[currentFrame]);
    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImage swapImage = swapchainImages[imageIndex];
    VkImage gradingImage = colorGrading.gradingImages[imageIndex];
    VkImageLayout currentLayout = colorGrading.gradingImageLayouts[imageIndex];
    if (currentLayout != VK_IMAGE_LAYOUT_GENERAL)
    {
        LayoutInfo prevInfo = layoutInfoFor(currentLayout);
        VkImageMemoryBarrier gradingBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        gradingBarrier.oldLayout = currentLayout;
        gradingBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        gradingBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        gradingBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        gradingBarrier.image = gradingImage;
        gradingBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        gradingBarrier.subresourceRange.baseMipLevel = 0;
        gradingBarrier.subresourceRange.levelCount = 1;
        gradingBarrier.subresourceRange.baseArrayLayer = 0;
        gradingBarrier.subresourceRange.layerCount = 1;
        gradingBarrier.srcAccessMask = prevInfo.accessMask;
        gradingBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             prevInfo.stage,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &gradingBarrier);
        colorGrading.gradingImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_GENERAL;
    }

    auto ensureSwapImageGeneral = [&]()
    {
        VkImageLayout swapLayout = swapchainImageLayouts[imageIndex];
        if (swapLayout == VK_IMAGE_LAYOUT_GENERAL)
        {
            return;
        }
        LayoutInfo swapInfo = layoutInfoFor(swapLayout);
        VkImageMemoryBarrier swapBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        swapBarrier.oldLayout = swapLayout;
        swapBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapBarrier.image = swapImage;
        swapBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        swapBarrier.subresourceRange.baseMipLevel = 0;
        swapBarrier.subresourceRange.levelCount = 1;
        swapBarrier.subresourceRange.baseArrayLayer = 0;
        swapBarrier.subresourceRange.layerCount = 1;
        swapBarrier.srcAccessMask = swapInfo.accessMask;
        swapBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             swapInfo.stage,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &swapBarrier);
        swapchainImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_GENERAL;
    };

    // Update descriptor set for this image
    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageInfo.imageView = colorGrading.gradingImageViews[imageIndex];

    VkDescriptorImageInfo overlayImageInfo{};
    overlayImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (overlayInfo.enabled && overlayInfo.overlay.view != VK_NULL_HANDLE && overlayInfo.overlay.sampler != VK_NULL_HANDLE)
    {
        overlayImageInfo.imageView = overlayInfo.overlay.view;
        overlayImageInfo.sampler = overlayInfo.overlay.sampler;
    }
    else
    {
        overlayImageInfo.imageView = videoImages.luma.view;
        overlayImageInfo.sampler = videoImages.luma.sampler;
    }
    VkDescriptorImageInfo fpsOverlayImageInfo{};
    fpsOverlayImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    fpsOverlayImageInfo.imageView = (fpsOverlayInfo.overlay.view != VK_NULL_HANDLE) ? fpsOverlayInfo.overlay.view : overlayImageInfo.imageView;
    fpsOverlayImageInfo.sampler = (fpsOverlayInfo.overlay.sampler != VK_NULL_HANDLE) ? fpsOverlayInfo.overlay.sampler : overlayImageInfo.sampler;

    VkDescriptorImageInfo lumaInfo{};
    lumaInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    lumaInfo.imageView = videoImages.luma.view;
    lumaInfo.sampler = videoImages.luma.sampler;

    VkDescriptorImageInfo chromaInfo{};
    chromaInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    chromaInfo.imageView = videoImages.chroma.view ? videoImages.chroma.view : videoImages.luma.view;
    chromaInfo.sampler = videoImages.chroma.sampler ? videoImages.chroma.sampler : videoImages.luma.sampler;

    VkDescriptorBufferInfo curveBufferInfo{};
    curveBufferInfo.buffer = colorGrading.curveUBO();
    curveBufferInfo.offset = 0;
    curveBufferInfo.range = colorGrading.curveUBOSize();

    std::array<VkWriteDescriptorSet, 6> writes{};
    // 0: storage (swapchain)
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptorSets[imageIndex];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &storageInfo;

    // 1: overlay
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSets[imageIndex];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &overlayImageInfo;

    // 2: fps overlay
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptorSets[imageIndex];
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &fpsOverlayImageInfo;

    // 3: luma
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = descriptorSets[imageIndex];
    writes[3].dstBinding = 3;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].descriptorCount = 1;
    writes[3].pImageInfo = &lumaInfo;

    // 4: chroma
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = descriptorSets[imageIndex];
    writes[4].dstBinding = 4;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].descriptorCount = 1;
    writes[4].pImageInfo = &chromaInfo;

    // 5: curve UBO
    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = descriptorSets[imageIndex];
    writes[5].dstBinding = 5;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[5].descriptorCount = 1;
    writes[5].pBufferInfo = &curveBufferInfo;

    vkUpdateDescriptorSets(engine->logicalDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const float reservedHeight = kScrubberHeight + kScrubberMargin * 2.0f;
    const float availableHeight = std::max(1.0f, static_cast<float>(swapchainExtent.height) - reservedHeight);
    const float outputAspect = static_cast<float>(swapchainExtent.width) / availableHeight;
    const float videoAspect = videoImages.height > 0 ? static_cast<float>(videoImages.width) / static_cast<float>(videoImages.height) : 1.0f;
    float targetWidth = static_cast<float>(swapchainExtent.width);
    float targetHeight = availableHeight;
    if (!overrides || !overrides->useTargetOverride)
    {
        if (videoAspect > outputAspect)
        {
            targetHeight = targetWidth / videoAspect;
        }
        else
        {
            targetWidth = targetHeight * videoAspect;
        }
    }
    else
    {
        targetWidth = overrides->targetSize.x;
        targetHeight = overrides->targetSize.y;
    }
    const float originX = (overrides && overrides->useTargetOverride)
                              ? overrides->targetOrigin.x
                              : (static_cast<float>(swapchainExtent.width) - targetWidth) * 0.5f;
    const float originY = (overrides && overrides->useTargetOverride)
                              ? overrides->targetOrigin.y
                              : (kScrubberMargin + (availableHeight - targetHeight) * 0.5f);

    const uint32_t groupX = (swapchainExtent.width + 15) / 16;
    const uint32_t groupY = (swapchainExtent.height + 15) / 16;
    if (videoPassEnabled)
    {
        colorGrading.dispatch(commandBuffer,
                              pipelineLayout,
                              descriptorSets[imageIndex],
                              cropPushConstants,
                              groupX,
                              groupY);
        if (renderDebugEnabled())
        {
            LOG_DEBUG(std::cout << "[Display2D] Video pass (blit) pipeline bound and ready for dispatch" << std::endl);
            LOG_DEBUG(std::cout << "[Display2D] Video pass dispatched" << std::endl);
        }
    }

    // Copy the decoded video into the swapchain so overlay/scrub can draw on top
    ensureSwapImageGeneral();
    LayoutInfo gradingInfo = layoutInfoFor(colorGrading.gradingImageLayouts[imageIndex]);
    VkImageMemoryBarrier gradingToTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    gradingToTransfer.oldLayout = colorGrading.gradingImageLayouts[imageIndex];
    gradingToTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    gradingToTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    gradingToTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    gradingToTransfer.image = gradingImage;
    gradingToTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    gradingToTransfer.subresourceRange.baseMipLevel = 0;
    gradingToTransfer.subresourceRange.levelCount = 1;
    gradingToTransfer.subresourceRange.baseArrayLayer = 0;
    gradingToTransfer.subresourceRange.layerCount = 1;
    gradingToTransfer.srcAccessMask = gradingInfo.accessMask;
    gradingToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    LayoutInfo swapInfo = layoutInfoFor(swapchainImageLayouts[imageIndex]);
    VkImageMemoryBarrier swapToTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    swapToTransfer.oldLayout = swapchainImageLayouts[imageIndex];
    swapToTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapToTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapToTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapToTransfer.image = swapImage;
    swapToTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapToTransfer.subresourceRange.baseMipLevel = 0;
    swapToTransfer.subresourceRange.levelCount = 1;
    swapToTransfer.subresourceRange.baseArrayLayer = 0;
    swapToTransfer.subresourceRange.layerCount = 1;
    swapToTransfer.srcAccessMask = swapInfo.accessMask;
    swapToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer,
                         gradingInfo.stage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &gradingToTransfer);
    vkCmdPipelineBarrier(commandBuffer,
                         swapInfo.stage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &swapToTransfer);
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.dstSubresource = copyRegion.srcSubresource;
    copyRegion.extent.width = swapchainExtent.width;
    copyRegion.extent.height = swapchainExtent.height;
    copyRegion.extent.depth = 1;
    vkCmdCopyImage(commandBuffer,
                   gradingImage,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   swapImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1,
                   &copyRegion);
    VkImageMemoryBarrier swapBack{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    swapBack.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapBack.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    swapBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapBack.image = swapImage;
    swapBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapBack.subresourceRange.baseMipLevel = 0;
    swapBack.subresourceRange.levelCount = 1;
    swapBack.subresourceRange.baseArrayLayer = 0;
    swapBack.subresourceRange.layerCount = 1;
    swapBack.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapBack.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &swapBack);
    swapchainImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_GENERAL;
    colorGrading.gradingImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo swapStorageInfo{};
    swapStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    swapStorageInfo.imageView = swapchainImageViews[imageIndex];
    swapStorageInfo.sampler = VK_NULL_HANDLE;
    VkWriteDescriptorSet swapStorageWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    swapStorageWrite.dstSet = descriptorSets[imageIndex];
    swapStorageWrite.dstBinding = 0;
    swapStorageWrite.dstArrayElement = 0;
    swapStorageWrite.descriptorCount = 1;
    swapStorageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swapStorageWrite.pImageInfo = &swapStorageInfo;
    vkUpdateDescriptorSets(engine->logicalDevice, 1, &swapStorageWrite, 0, nullptr);

    LayoutInfo swapToPresentInfo = layoutInfoFor(swapchainImageLayouts[imageIndex]);
    VkImageMemoryBarrier swapToPresent{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    swapToPresent.oldLayout = swapchainImageLayouts[imageIndex];
    swapToPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapToPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapToPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapToPresent.image = swapImage;
    swapToPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapToPresent.subresourceRange.baseMipLevel = 0;
    swapToPresent.subresourceRange.levelCount = 1;
    swapToPresent.subresourceRange.baseArrayLayer = 0;
    swapToPresent.subresourceRange.layerCount = 1;
    swapToPresent.srcAccessMask = swapToPresentInfo.accessMask;
    swapToPresent.dstAccessMask = 0;

    vkCmdPipelineBarrier(commandBuffer,
                         swapToPresentInfo.stage,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &swapToPresent);
    swapchainImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkEndCommandBuffer(commandBuffer);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to submit compute work for Display2D");
    }

    VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;

    VkResult presentResult = vkQueuePresentKHR(graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
    {
        recreateSwapchain();
    }

    currentFrame = (currentFrame + 1) % kMaxFramesInFlight;

    if (!initialized || !videoLoaded || windows.empty())
    {
        return false;
    }

    for (auto& window : windows)
    {
        if (window)
        {
            window->pollEvents();
            if (window->shouldClose())
            {
                return false;
            }
        }
    }

    double playbackSeconds = advancePlayback(playbackState, playing);
    currentTime = static_cast<float>(playbackSeconds);

    updateFpsOverlay();

    ColorAdjustments adjustments{};
    applyGrading(adjustments);
    const ColorAdjustments* activeAdjustments = gradingSettings ? &adjustments : nullptr;

    RenderOverrides overrides = buildRenderOverrides();
    float scrubProgress = duration > 0.0f ? currentTime / duration : 0.0f;
    float scrubPlaying = playing ? 1.0f : 0.0f;

    for (auto& window : windows)
    {
        if (window)
        {
            window->renderFrame(playbackState.video.descriptors,
                                playbackState.overlay.info,
                                playbackState.fpsOverlay.info,
                                playbackState.colorInfo,
                                scrubProgress,
                                scrubPlaying,
                                &overrides,
                                activeAdjustments);
        }
    }
}

