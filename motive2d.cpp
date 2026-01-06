#include "motive2d.h"

#include "crop.h"
#include "decoder.h"
#include "engine2d.h"
#include "fps.h"
#include "pose_overlay.h"
// #include "rect_overlay.h" // Already included via motive2d.h
#include "rgba2nv12.h"
#include "scrubber.h"
#include "subtitle.h"
#include "utils.h"

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
    : options(cliOptions)
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
    // colorGrading = new ColorGrading(engine); // TODO: Needs Display2D, not Engine2D
    // colorGradingUi = new ColorGradingUi(engine); // TODO: Needs proper constructor arguments
    colorGrading = nullptr;
    colorGradingUi = nullptr;
    DecoderInitParams params{};
    params.requireGraphicsQueue = true;
    params.debugLogging = cliOptions.debugLogging;
    decoder = new Decoder(cliOptions.videoPath, params);

    // Create pipeline test directory if needed
    if (options.pipelineTest)
    {
        std::filesystem::create_directories(options.pipelineTestDir);
        std::cout << "[Motive2D] Pipeline test mode enabled. Output directory: "
                  << options.pipelineTestDir << std::endl;
    }
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
    // TODO: Implement proper run loop
    // These classes have different run() signatures or no run() method
    // For now, just call what's available

    // decoder->run(); // Decoder doesn't have run()
    // colorGrading->run(); // ColorGrading doesn't have run()
    // colorGradingUi->run(); // Needs arguments
    // rectOverlay->run(); // Needs 6 arguments
    // poseOverlay->run(); // Needs 10 arguments
    crop->run(); // We fixed this
    // scrubber->run(); // Needs 2 arguments
    // subtitle->run(); // Needs 13 arguments
    // fpsOverlay->run(); // FpsOverlay doesn't have run()

    // If pipeline test is enabled, export a test frame
    if (options.pipelineTest)
    {
        exportPipelineTestFrame();
    }
}

void Motive2D::exportPipelineTestFrame()
{
    std::cout << "[Motive2D] Exporting pipeline test frames..." << std::endl;

    std::filesystem::path basePath = options.pipelineTestDir;

    // Variables that will be used for all stages
    int width = 640;
    int height = 480;
    int channels = 4;
    std::vector<uint8_t> testImage;

    // Try to decode and save an actual video frame as input stage
    bool savedRealFrame = false;

    if (decoder)
    {
        // Seek to beginning of video
        if (decoder->seek(0.0f))
        {
            std::cout << "[Motive2D] Successfully sought to beginning of video" << std::endl;

            // Start async decoding
            if (decoder->startAsyncDecoding(2)) // Buffer 2 frames
            {
                // Wait for decoding to start and buffer a frame
                DecodedFrame frame;

                decoder->acquireDecodedFrame(frame);
                width = decoder->getWidth();
                height = decoder->getHeight();

                if (!frame.buffer.empty() && width > 0 && height > 0)
                {
                    std::cout << "[Motive2D] Decoded frame: " << width << "x" << height
                              << ", buffer size: " << frame.buffer.size() << " bytes" << std::endl;

                    // Convert YUV to RGB for PNG saving
                    // First check if it's NV12 format (common for hardware decoding)
                    size_t yPlaneBytes = width * height;                  // Assuming 8-bit YUV
                    size_t uvPlaneBytes = (width / 2) * (height / 2) * 2; // NV12 has interleaved UV plane

                    // Try to convert NV12 to BGR
                    std::vector<uint8_t> bgrImage;
                    if (convertNv12ToBgr(frame.buffer.data(), yPlaneBytes, uvPlaneBytes,
                                         width, height, bgrImage))
                    {
                        // Convert BGR to RGBA for PNG saving
                        testImage.resize(width * height * 4);
                        for (size_t i = 0; i < bgrImage.size() / 3; ++i)
                        {
                            testImage[i * 4 + 0] = bgrImage[i * 3 + 2]; // R
                            testImage[i * 4 + 1] = bgrImage[i * 3 + 1]; // G
                            testImage[i * 4 + 2] = bgrImage[i * 3 + 0]; // B
                            testImage[i * 4 + 3] = 255;                 // A
                        }

                        std::filesystem::path inputPath = basePath / "input_stage.png";
                        if (saveImageToPNG(inputPath, testImage.data(), width, height, 4))
                        {
                            std::cout << "[Motive2D] Saved actual video frame as input stage: "
                                      << inputPath << " (" << width << "x" << height << ")" << std::endl;
                            savedRealFrame = true;
                        }
                    }
                    else
                    {
                        std::cout << "[Motive2D] Failed to convert NV12 to BGR" << std::endl;
                    }

                    // Also save raw YUV data for reference
                    std::filesystem::path yuvPath = basePath / "frame.yuv";
                    if (saveRawFrameData(yuvPath, frame.buffer.data(), frame.buffer.size()))
                    {
                        std::cout << "[Motive2D] Saved raw YUV data: " << yuvPath
                                  << " (" << frame.buffer.size() << " bytes)" << std::endl;
                    }
                }
                else
                {
                    std::cout << "[Motive2D] Decoded frame has empty buffer or invalid dimensions" << std::endl;
                }
            }
            else
            {
                std::cout << "[Motive2D] Failed to start async decoding" << std::endl;
            }
        }
        else
        {
            std::cout << "[Motive2D] Failed to seek to beginning of video" << std::endl;
        }
    }
    else
    {
        std::cout << "[Motive2D] No decoder available" << std::endl;
    }

    // If we couldn't save a real frame, fall back to test pattern
    if (!savedRealFrame)
    {
        std::cout << "[Motive2D] Could not decode video frame, using test pattern" << std::endl;

        width = 640;
        height = 480;
        channels = 4;
        testImage.resize(width * height * channels);

        // Fill with a test pattern
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = (y * width + x) * channels;
                testImage[idx + 0] = static_cast<uint8_t>(255 * x / width);  // R
                testImage[idx + 1] = static_cast<uint8_t>(255 * y / height); // G
                testImage[idx + 2] = 128;                                    // B
                testImage[idx + 3] = 255;                                    // A
            }
        }

        std::filesystem::path inputPath = basePath / "input_stage.png";
        if (saveImageToPNG(inputPath, testImage.data(), width, height, channels))
        {
            std::cout << "[Motive2D] Saved test pattern as input stage: " << inputPath << std::endl;
        }
    }

    // Region stage (with some modification)
    std::vector<uint8_t> regionImage = testImage;
    // Add a red rectangle in the middle
    int rectX = width / 4;
    int rectY = height / 4;
    int rectW = width / 2;
    int rectH = height / 2;

    for (int y = rectY; y < rectY + rectH; ++y)
    {
        for (int x = rectX; x < rectX + rectW; ++x)
        {
            if (x == rectX || x == rectX + rectW - 1 || y == rectY || y == rectY + rectH - 1)
            {
                int idx = (y * width + x) * channels;
                regionImage[idx + 0] = 255; // R
                regionImage[idx + 1] = 0;   // G
                regionImage[idx + 2] = 0;   // B
            }
        }
    }

    std::filesystem::path regionPath = basePath / "region_stage.png";
    if (saveImageToPNG(regionPath, regionImage.data(), width, height, channels))
    {
        std::cout << "[Motive2D] Saved region stage: " << regionPath << std::endl;
    }

    // Grading stage (with color adjustment)
    std::vector<uint8_t> gradingImage = testImage;
    // Apply a simple color grading (boost reds)
    for (int i = 0; i < width * height * channels; i += channels)
    {
        gradingImage[i + 0] = std::min(255, static_cast<int>(gradingImage[i + 0] * 1.2f)); // Boost red
        gradingImage[i + 1] = std::max(0, static_cast<int>(gradingImage[i + 1] * 0.9f));   // Reduce green
    }

    std::filesystem::path gradingPath = basePath / "grading_stage.png";
    if (saveImageToPNG(gradingPath, gradingImage.data(), width, height, channels))
    {
        std::cout << "[Motive2D] Saved grading stage: " << gradingPath << std::endl;
    }

    // Also save as JPG
    std::filesystem::path jpgPath = basePath / "output.jpg";
    if (saveImageToJPG(jpgPath, testImage.data(), width, height, channels, 90))
    {
        std::cout << "[Motive2D] Saved JPG output: " << jpgPath << std::endl;
    }

    std::cout << "[Motive2D] Pipeline test export complete." << std::endl;
}
