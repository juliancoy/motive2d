#include "engine2d.h"

#include <glm/glm.hpp>
#include <glm/vec2.hpp>

#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>

namespace
{
const std::filesystem::path kDefaultVideoPath("P1090533_main8_hevc_fast.mkv");

struct CliOptions
{
    std::filesystem::path videoPath = kDefaultVideoPath;
    std::optional<bool> swapUV;
    bool showInput = true;
    bool showRegion = true;
    bool showGrading = true;
};

CliOptions parseCliOptions(int argc, char** argv)
{
    CliOptions opts{};
    bool windowsSpecified = false;
    bool parsedInput = false;
    bool parsedRegion = false;
    bool parsedGrading = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i] ? argv[i] : "");
        if (arg.empty())
        {
            continue;
        }
        if (arg == "--video" && i + 1 < argc)
        {
            std::string nextArg(argv[i + 1] ? argv[i + 1] : "");
            if (!nextArg.empty() && nextArg[0] != '-')
            {
                opts.videoPath = std::filesystem::path(nextArg);
                ++i;
            }
            continue;
        }
        if (arg.rfind("--video=", 0) == 0)
        {
            opts.videoPath = std::filesystem::path(arg.substr(std::string("--video=").size()));
            continue;
        }
        if (arg == "--swapUV")
        {
            opts.swapUV = true;
            continue;
        }
        if (arg == "--noSwapUV")
        {
            opts.swapUV = false;
            continue;
        }
        if (arg.rfind("--windows", 0) == 0)
        {
            std::string list;
            if (arg == "--windows" && i + 1 < argc)
            {
                std::string nextArg(argv[i + 1] ? argv[i + 1] : "");
                if (!nextArg.empty() && nextArg[0] != '-')
                {
                    list = nextArg;
                    ++i;
                }
            }
            else if (arg.rfind("--windows=", 0) == 0)
            {
                list = arg.substr(std::string("--windows=").size());
            }

            if (!list.empty())
            {
                windowsSpecified = true;
                parsedInput = false;
                parsedRegion = false;
                parsedGrading = false;
                std::stringstream ss(list);
                std::string token;
                while (std::getline(ss, token, ','))
                {
                    if (token == "none")
                    {
                        parsedInput = parsedRegion = parsedGrading = false;
                        continue;
                    }
                    if (token == "input")
                    {
                        parsedInput = true;
                        continue;
                    }
                    if (token == "region")
                    {
                        parsedRegion = true;
                        continue;
                    }
                    if (token == "grading")
                    {
                        parsedGrading = true;
                    }
                }
            }
        }
        else if (arg[0] != '-')
        {
            opts.videoPath = std::filesystem::path(arg);
        }
    }

    if (windowsSpecified)
    {
        opts.showInput = parsedInput;
        opts.showRegion = parsedRegion;
        opts.showGrading = parsedGrading;
    }

    return opts;
}

double g_scrollDelta = 0.0;
static void onScroll(GLFWwindow*, double, double yoffset)
{
    g_scrollDelta += yoffset;
}

struct ScrubberUi
{
    double left;
    double top;
    double right;
    double bottom;
    double iconLeft;
    double iconTop;
    double iconRight;
    double iconBottom;
};

ScrubberUi computeScrubberUi(int windowWidth, int windowHeight)
{
    const double kScrubberMargin = 20.0;
    const double kScrubberHeight = 64.0;
    const double kScrubberMinWidth = 200.0;
    const double kPlayIconSize = 28.0;

    ScrubberUi ui{};
    const double availableWidth = static_cast<double>(windowWidth);
    const double scrubberWidth =
        std::max(kScrubberMinWidth,
                 availableWidth - (kPlayIconSize + kScrubberMargin * 3.0));
    const double scrubberHeight = kScrubberHeight;
    ui.iconLeft = kScrubberMargin;
    ui.iconRight = ui.iconLeft + kPlayIconSize;
    ui.top = static_cast<double>(windowHeight) - scrubberHeight - kScrubberMargin;
    ui.bottom = ui.top + scrubberHeight;
    ui.iconTop = ui.top + (scrubberHeight - kPlayIconSize) * 0.5;
    ui.iconBottom = ui.iconTop + kPlayIconSize;

    ui.left = ui.iconRight + kScrubberMargin;
    ui.right = ui.left + scrubberWidth;
    return ui;
}

bool cursorInScrubber(double x, double y, int windowWidth, int windowHeight)
{
    const ScrubberUi ui = computeScrubberUi(windowWidth, windowHeight);
    return x >= ui.left && x <= ui.right && y >= ui.top && y <= ui.bottom;
}

bool cursorInPlayButton(double x, double y, int windowWidth, int windowHeight)
{
    const ScrubberUi ui = computeScrubberUi(windowWidth, windowHeight);
    return x >= ui.iconLeft && x <= ui.iconRight && y >= ui.iconTop && y <= ui.iconBottom;
}
} // namespace

int main(int argc, char** argv)
{
    CliOptions cli = parseCliOptions(argc, argv);

    Engine2D engine;
    if (!engine.initialize())
    {
        return 1;
    }

    if (!cli.showInput && !cli.showRegion && !cli.showGrading)
    {
        std::cout << "[motive2d] No windows enabled (--windows=none). Add --windows=input to open the main view.\n";
        return 1;
    }

    Display2D* inputWindow = nullptr;
    Display2D* regionWindow = nullptr;
    Display2D* gradingWindow = nullptr;

    if (cli.showInput)
    {
        inputWindow = engine.createWindow(1280, 720, "Motive Video 2D");
        if (!inputWindow)
        {
            std::cerr << "[motive2d] Failed to create main window.\n";
            return 1;
        }
        glfwSetScrollCallback(inputWindow->window, onScroll);
    }
    if (cli.showRegion)
    {
        regionWindow = engine.createWindow(360, 640, "Region View");
        if (!regionWindow)
        {
            std::cerr << "[motive2d] Failed to create region window.\n";
            return 1;
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
    }

    if (!engine.loadVideo(cli.videoPath, cli.swapUV))
    {
        engine.shutdown();
        return 1;
    }

    auto& playbackState = engine.getPlaybackState();
    auto& overlayCompute = engine.getOverlayCompute();

    overlay::ImageResource gradingOverlayImage;
    OverlayImageInfo gradingOverlayInfo{};
    overlay::ImageResource blackLuma;
    overlay::ImageResource blackChroma;
    VkSampler blackSampler = VK_NULL_HANDLE;
    VideoImageSet blackVideo{};

    {
        uint8_t luma = 0;
        uint8_t chroma[2] = {128, 128};
        overlay::uploadImageData(&engine, blackLuma, &luma, sizeof(luma), 1, 1, VK_FORMAT_R8_UNORM);
        overlay::uploadImageData(&engine, blackChroma, chroma, sizeof(chroma), 1, 1, VK_FORMAT_R8G8_UNORM);
        try
        {
            blackSampler = createLinearClampSampler(&engine);
        }
        catch (const std::exception&)
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
    }

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
    std::array<float, kCurveLutSize> curveLut{};
    bool curveDirty = true;
    bool gradingOverlayDirty = true;
    uint32_t gradingFbWidth = 0;
    uint32_t gradingFbHeight = 0;
    bool gradingMouseHeld = false;
    bool gradingRightHeld = false;
    grading::SliderLayout gradingLayout{};
    bool gradingPreviewEnabled = true;
    bool detectionEnabled = false;

    auto runGradingClick = [&](bool rightClick) {
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

    bool shouldExit = false;
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

        if (inputWindow)
        {
            int mouseState = glfwGetMouseButton(inputWindow->window, GLFW_MOUSE_BUTTON_LEFT);
            if (mouseState == GLFW_PRESS && !mouseHeld)
            {
                double cursorX = 0.0;
                double cursorY = 0.0;
                glfwGetCursorPos(inputWindow->window, &cursorX, &cursorY);
                if (cursorInPlayButton(cursorX, cursorY, inputWindow->width, inputWindow->height))
                {
                    playing = !playing;
                }
                else if (cursorInScrubber(cursorX, cursorY, inputWindow->width, inputWindow->height))
                {
                    scrubDragging = true;
                    scrubDragStartX = cursorX;
                    scrubDragStartProgress = scrubProgressUi;
                    mouseHeld = true;
                    playing = false;
                }
                else
                {
                    glm::vec2 scale(static_cast<float>(windowWidth / inputWindow->width),
                                    static_cast<float>(windowHeight / inputWindow->height));
                    rectCenter = glm::vec2(static_cast<float>(cursorX) * scale.x,
                                           static_cast<float>(cursorY) * scale.y);
                }
            }
            else if (mouseState == GLFW_RELEASE)
            {
                if (scrubDragging)
                {
                    const ScrubberUi ui = computeScrubberUi(inputWindow->width, inputWindow->height);
                    double x = 0.0;
                    glfwGetCursorPos(inputWindow->window, &x, nullptr);
                    double deltaX = x - scrubDragStartX;
                    double progressDelta = deltaX / (ui.right - ui.left);
                    scrubProgressUi = std::clamp(scrubDragStartProgress + static_cast<float>(progressDelta), 0.0f, 1.0f);
                    scrubDragging = false;
                    playing = true;
                }
                mouseHeld = false;
            }
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

        double playbackSeconds = advancePlayback(playbackState, playing && !scrubDragging);
        engine.setCurrentTime(static_cast<float>(playbackSeconds));
        const float totalDuration = engine.getDuration();
        if (totalDuration > 0.0f)
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

        runOverlayCompute(&engine,
                          overlayCompute,
                          playbackState.overlay.image,
                          fbWidth,
                          fbHeight,
                          rectCenter,
                          glm::vec2(rectWidth, rectHeight),
                          3.0f,
                          3.0f,
                          detectionEnabled ? 1.0f : 0.0f,
                          rectCenter + glm::vec2(rectWidth * 0.2f, rectHeight * 0.2f),
                          glm::vec2(rectWidth * 0.6f, rectHeight * 0.6f));
        playbackState.overlay.info.overlay.view = playbackState.overlay.image.view;
        playbackState.overlay.info.overlay.sampler = playbackState.overlay.sampler;
        playbackState.overlay.info.extent = {fbWidth, fbHeight};
        playbackState.overlay.info.offset = {0, 0};
        playbackState.overlay.info.enabled = true;

        ColorAdjustments adjustments{};
        if (gradingPreviewEnabled)
        {
            adjustments.exposure = gradingSettings.exposure;
            adjustments.contrast = gradingSettings.contrast;
            adjustments.saturation = gradingSettings.saturation;
            adjustments.shadows = gradingSettings.shadows;
            adjustments.midtones = gradingSettings.midtones;
            adjustments.highlights = gradingSettings.highlights;
            if (curveDirty)
            {
                grading::buildCurveLut(gradingSettings, curveLut);
                curveDirty = false;
            }
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
                regionOverrides.hideScrubber = true;
            }
        }

        bool rebuildGradingOverlay = gradingOverlayDirty;
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
                rebuildGradingOverlay = true;
            }
        }

        if (gradingWindow && rebuildGradingOverlay)
        {
            gradingOverlayDirty = grading::buildGradingOverlay(&engine,
                                                               gradingSettings,
                                                               gradingOverlayImage,
                                                               gradingOverlayInfo,
                                                               gradingFbWidth,
                                                               gradingFbHeight,
                                                               gradingLayout,
                                                               gradingPreviewEnabled,
                                                               detectionEnabled);
            gradingOverlayInfo.overlay.sampler = playbackState.overlay.sampler;
        }

        engine.refreshFpsOverlay();

        const ColorAdjustments* adjustmentsPtr = gradingPreviewEnabled ? &adjustments : nullptr;
        if (inputWindow)
        {
            inputWindow->renderFrame(playbackState.video.descriptors,
                                     playbackState.overlay.info,
                                     playbackState.fpsOverlay.info,
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
            gradingOverrides.hideScrubber = true;
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

    overlay::destroyImageResource(&engine, gradingOverlayImage);
    overlay::destroyImageResource(&engine, blackLuma);
    overlay::destroyImageResource(&engine, blackChroma);
    if (blackSampler != VK_NULL_HANDLE && blackSampler != playbackState.overlay.sampler)
    {
        vkDestroySampler(engine.logicalDevice, blackSampler, nullptr);
    }

    engine.shutdown();
    return 0;
}
