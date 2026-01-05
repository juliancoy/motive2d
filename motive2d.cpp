#include "motive2d.h"

#include "crop.h"
#include "decoder.h"
#include "engine2d.h"
#include "fps.h"
#include "pose_overlay.h"
#include "rect_overlay.h"
#include "scrubber.h"
#include "subtitle.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>


std::mutex g_stageMutex;
std::condition_variable g_stageCv;

Motive2D::Motive2D(CliOptions cliOptions)
{
    engine = new Engine2D();
    if (!engine || !engine->initialize())
    {
        throw std::runtime_error("Failed to initialize Vulkan engine");
    }

    const std::filesystem::path subtitlePath =
        cliOptions.videoPath.parent_path() / (cliOptions.videoPath.stem().string() + ".json");
    subtitle = new Subtitle(subtitlePath, engine);
    rectOverlay = new RectOverlay(engine);
    poseOverlay = new PoseOverlay(engine);
    crop = new Crop();
    blackSampler = VK_NULL_HANDLE;
    scrubber = new Scrubber(engine);
    fpsOverlay = new FpsOverlay(engine);
    colorGrading = new ColorGrading(engine);
    colorGradingUi = new ColorGradingUi(engine);
    DecoderInitParams params{};
    params.requireGraphicsQueue = true;
    params.debugLogging = cliOptions.debugLogging;
    decoder = new Decoder(cliOptions.videoPath, params);
}

Motive2D::~Motive2D()
{
    delete subtitle;
    delete rectOverlay;
    delete poseOverlay;
    delete crop;
    delete scrubber;
    delete fpsOverlay;
    delete decoder;
    delete engine;
}

void Motive2D::run()
{
    decoder->run();
    colorGrading->run();
    colorGradingUi->run();
    rectOverlay->run();
    poseOverlay->run();
    crop->run();
    scrubber->run();
    subtitle->run();
    fpsOverlay->run();
}
