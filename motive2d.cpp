#include "engine2d.h"
#include "pose_overlay.hpp"
#include "video.h"

#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define NCNN_VULKAN 1
#define NCNN_SIMPLEVK 0
#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <ncnn/layer.h>

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
    bool poseEnabled = false;
    std::filesystem::path poseModelBase = "yolov8n_pose";
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
        else if (arg == "--pose")
        {
            opts.poseEnabled = true;
            if (i + 1 < argc)
            {
                std::string nextArg(argv[i + 1] ? argv[i + 1] : "");
                if (!nextArg.empty() && nextArg[0] != '-')
                {
                    opts.poseModelBase = std::filesystem::path(nextArg);
                    ++i;
                }
            }
        }
        else if (arg.rfind("--pose=", 0) == 0)
        {
            opts.poseEnabled = true;
            opts.poseModelBase = std::filesystem::path(arg.substr(std::string("--pose=").size()));
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

struct KeyPoint
{
    float x = 0.0f;
    float y = 0.0f;
    float prob = 0.0f;
};

struct PoseObject
{
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float score = 0.0f;
    std::vector<KeyPoint> keypoints;
};

static inline float intersection_area(const PoseObject& a, const PoseObject& b)
{
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    const float width = std::max(0.0f, right - left);
    const float height = std::max(0.0f, bottom - top);
    return width * height;
}

static void nms_sorted_bboxes(const std::vector<PoseObject>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    if (objects.empty())
    {
        return;
    }

    const size_t n = objects.size();
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i)
    {
        areas[i] = objects[i].width * objects[i].height;
    }

    for (size_t i = 0; i < n; ++i)
    {
        const PoseObject& a = objects[i];
        bool keep = true;
        for (int idx : picked)
        {
            const PoseObject& b = objects[idx];
            const float inter_area = intersection_area(a, b);
            const float union_area = areas[i] + areas[idx] - inter_area;
            if (union_area > 0.0f && inter_area / union_area > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            picked.push_back(static_cast<int>(i));
        }
    }
}

static inline float sigmoid(float v)
{
    return 1.0f / (1.0f + std::exp(-v));
}

static void generate_proposals(const ncnn::Mat& pred,
                               const ncnn::Mat& pred_points,
                               const std::vector<int>& strides,
                               const ncnn::Mat& in_pad,
                               float prob_threshold,
                               std::vector<PoseObject>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;
    constexpr int reg_max_1 = 16;
    const int num_points = pred_points.w / 3;
    int pred_row_offset = 0;

    for (int stride : strides)
    {
        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        for (int y = 0; y < num_grid_y; ++y)
        {
            for (int x = 0; x < num_grid_x; ++x)
            {
                const int idx = pred_row_offset + y * num_grid_x + x;
                const ncnn::Mat pred_grid = pred.row_range(idx, 1);
                const ncnn::Mat pred_points_grid = pred_points.row_range(idx, 1).reshape(3, num_points);

                const float score = sigmoid(pred_grid[reg_max_1 * 4]);
                if (score < prob_threshold)
                {
                    continue;
                }

                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4).clone();

                ncnn::Layer* softmax = ncnn::create_layer("Softmax");
                ncnn::ParamDict pd;
                pd.set(0, 1);
                pd.set(1, 1);
                softmax->load_param(pd);
                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;
                softmax->create_pipeline(opt);
                softmax->forward_inplace(pred_bbox, opt);
                softmax->destroy_pipeline(opt);
                delete softmax;

                float pred_ltrb[4];
                for (int k = 0; k < 4; ++k)
                {
                    float dis = 0.0f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; ++l)
                    {
                        dis += l * dis_after_sm[l];
                    }
                    pred_ltrb[k] = dis * stride;
                }

                const float pb_cx = (x + 0.5f) * stride;
                const float pb_cy = (y + 0.5f) * stride;
                const float x0 = pb_cx - pred_ltrb[0];
                const float y0 = pb_cy - pred_ltrb[1];
                const float x1 = pb_cx + pred_ltrb[2];
                const float y1 = pb_cy + pred_ltrb[3];

                PoseObject obj;
                obj.x = x0;
                obj.y = y0;
                obj.width = x1 - x0;
                obj.height = y1 - y0;
                obj.score = score;
                obj.keypoints.reserve(num_points);
                for (int k = 0; k < num_points; ++k)
                {
                    KeyPoint keypoint;
                    const float* pointRow = pred_points_grid.row(k);
                    keypoint.x = (x + pointRow[0] * 2.0f) * stride;
                    keypoint.y = (y + pointRow[1] * 2.0f) * stride;
                    keypoint.prob = sigmoid(pointRow[2]);
                    obj.keypoints.push_back(keypoint);
                }

                objects.push_back(std::move(obj));
            }
        }

        pred_row_offset += num_grid;
    }
}

class PoseDetector
{
public:
    PoseDetector()
        : strides{8, 16, 32}
    {
    }

    bool initialize(const std::filesystem::path& modelBase, std::string& error)
    {
        std::filesystem::path paramPath = modelBase;
        std::filesystem::path binPath = modelBase;
        if (paramPath.extension() == ".bin")
        {
            binPath = paramPath;
            paramPath.replace_extension(".param");
        }
        else if (paramPath.extension() == ".param")
        {
            binPath = paramPath;
            binPath.replace_extension(".bin");
        }
        else
        {
            paramPath = modelBase;
            paramPath += ".ncnn.param";
            binPath = modelBase;
            binPath += ".ncnn.bin";
        }

        if (!std::filesystem::exists(paramPath) || !std::filesystem::exists(binPath))
        {
            error = "Pose model files not found: " + paramPath.string() + " / " + binPath.string();
            return false;
        }

        if (net.load_param(paramPath.string().c_str()) != 0)
        {
            error = "Failed to load " + paramPath.string();
            return false;
        }
        if (net.load_model(binPath.string().c_str()) != 0)
        {
            error = "Failed to load " + binPath.string();
            return false;
        }

        net.opt.use_vulkan_compute = true;
        initialized = true;
        return true;
    }

    bool detect(const uint8_t* bgrData, int imgWidth, int imgHeight, std::vector<PoseObject>& objects) const
    {
        if (!initialized || !bgrData || imgWidth <= 0 || imgHeight <= 0)
        {
            return false;
        }

        float scale = 1.0f;
        int targetW = imgWidth;
        int targetH = imgHeight;
        if (targetW > targetH)
        {
            scale = static_cast<float>(targetSize) / targetW;
            targetW = targetSize;
            targetH = static_cast<int>(targetH * scale);
        }
        else
        {
            scale = static_cast<float>(targetSize) / targetH;
            targetH = targetSize;
            targetW = static_cast<int>(targetW * scale);
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgrData,
                                                     ncnn::Mat::PIXEL_BGR2RGB,
                                                     imgWidth,
                                                     imgHeight,
                                                     targetW,
                                                     targetH);

        constexpr int max_stride = 32;
        const int wpad = ((targetW + max_stride - 1) / max_stride) * max_stride - targetW;
        const int hpad = ((targetH + max_stride - 1) / max_stride) * max_stride - targetH;

        ncnn::Mat in_pad;
        ncnn::copy_make_border(in,
                               in_pad,
                               hpad / 2,
                               hpad - hpad / 2,
                               wpad / 2,
                               wpad - wpad / 2,
                               ncnn::BORDER_CONSTANT,
                               114.0f);

        const float norm_vals[3] = {1/255.0f, 1/255.0f, 1/255.0f};
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in_pad);

        ncnn::Mat out;
        ncnn::Mat out_points;
        ex.extract("out0", out);
        ex.extract("out1", out_points);

        std::vector<PoseObject> proposals;
        generate_proposals(out, out_points, strides, in_pad, probThreshold, proposals);
        if (proposals.empty())
        {
            objects.clear();
            return true;
        }

        std::sort(proposals.begin(), proposals.end(), [](const PoseObject& a, const PoseObject& b) {
            return a.score > b.score;
        });

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nmsThreshold);
        objects.clear();

        const float padW = wpad / 2.0f;
        const float padH = hpad / 2.0f;
        for (int index : picked)
        {
            PoseObject obj = proposals[index];
            float x0 = (obj.x - padW) / scale;
            float y0 = (obj.y - padH) / scale;
            float x1 = (obj.x + obj.width - padW) / scale;
            float y1 = (obj.y + obj.height - padH) / scale;

            x0 = std::clamp(x0, 0.0f, static_cast<float>(imgWidth - 1));
            y0 = std::clamp(y0, 0.0f, static_cast<float>(imgHeight - 1));
            x1 = std::clamp(x1, 0.0f, static_cast<float>(imgWidth - 1));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(imgHeight - 1));

            obj.x = x0;
            obj.y = y0;
            obj.width = std::max(1.0f, x1 - x0);
            obj.height = std::max(1.0f, y1 - y0);

            for (auto& kp : obj.keypoints)
            {
                float kpX = (kp.x - padW) / scale;
                float kpY = (kp.y - padH) / scale;
                kp.x = std::clamp(kpX, 0.0f, static_cast<float>(imgWidth - 1));
                kp.y = std::clamp(kpY, 0.0f, static_cast<float>(imgHeight - 1));
            }

            objects.push_back(std::move(obj));
        }

        return true;
    }

    bool isValid() const
    {
        return initialized;
    }

private:
    ncnn::Net net;
    std::vector<int> strides;
    const int targetSize = 640;
    const float probThreshold = 0.25f;
    const float nmsThreshold = 0.45f;
    bool initialized = false;
};

bool convertNv12ToBgr(const uint8_t* nv12,
                      size_t yBytes,
                      size_t uvBytes,
                      int width,
                      int height,
                      std::vector<uint8_t>& bgr)
{
    if (!nv12 || width <= 0 || height <= 0 || yBytes == 0 || uvBytes == 0)
    {
        return false;
    }

    const size_t framePixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (bgr.size() != framePixels * 3)
    {
        bgr.resize(framePixels * 3);
    }

    const uint8_t* yPlane = nv12;
    const uint8_t* uvPlane = nv12 + yBytes;

    for (int row = 0; row < height; ++row)
    {
        const uint8_t* yRow = yPlane + static_cast<size_t>(row) * static_cast<size_t>(width);
        const uint8_t* uvRow = uvPlane + static_cast<size_t>(row / 2) * static_cast<size_t>(width);
        for (int col = 0; col < width; ++col)
        {
            const float Y = static_cast<float>(yRow[col]);
            const float U = static_cast<float>(uvRow[(col / 2) * 2]) - 128.0f;
            const float V = static_cast<float>(uvRow[(col / 2) * 2 + 1]) - 128.0f;

            const float y = std::max(0.0f, (Y - 16.0f)) * 1.164383f;
            const float r = y + 1.596027f * V;
            const float g = y - 0.391762f * U - 0.812968f * V;
            const float b = y + 2.017232f * U;

            const size_t dstIndex = (static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)) * 3;
            bgr[dstIndex + 0] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
            bgr[dstIndex + 1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
            bgr[dstIndex + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
        }
    }

    return true;
}

void buildDetectionEntries(const std::vector<PoseObject>& objects,
                           int frameWidth,
                           int frameHeight,
                           std::vector<overlay::DetectionEntry>& output)
{
    output.clear();
    if (frameWidth <= 0 || frameHeight <= 0)
    {
        return;
    }

    constexpr size_t kMaxDetections = 6;
    constexpr float kKeypointDiameterPx = 12.0f;
    const float widthInv = 1.0f / static_cast<float>(frameWidth);
    const float heightInv = 1.0f / static_cast<float>(frameHeight);
    const float keypointSizeNorm = kKeypointDiameterPx * std::max(widthInv, heightInv);
    static const std::array<glm::vec4, 4> palette = {
        glm::vec4(1.0f, 0.4f, 0.3f, 1.0f),
        glm::vec4(0.3f, 1.0f, 0.5f, 1.0f),
        glm::vec4(0.4f, 0.6f, 1.0f, 1.0f),
        glm::vec4(1.0f, 0.8f, 0.3f, 1.0f),
    };

    const size_t maxCount = std::min(objects.size(), kMaxDetections);
    output.reserve(maxCount * 18);

    for (size_t idx = 0; idx < maxCount; ++idx)
    {
        const PoseObject& obj = objects[idx];
        glm::vec4 bbox;
        bbox.x = std::clamp(obj.x * widthInv, 0.0f, 1.0f);
        bbox.y = std::clamp(obj.y * heightInv, 0.0f, 1.0f);
        bbox.z = std::clamp(obj.width * widthInv, 0.0f, 1.0f - bbox.x);
        bbox.w = std::clamp(obj.height * heightInv, 0.0f, 1.0f - bbox.y);

        overlay::DetectionEntry boxEntry{};
        boxEntry.bbox = bbox;
        boxEntry.color = palette[idx % palette.size()];
        boxEntry.confidence = obj.score;
        boxEntry.classId = 0;
        output.push_back(boxEntry);

        const glm::vec4 keyColor = boxEntry.color;
        for (size_t keyIndex = 0; keyIndex < obj.keypoints.size(); ++keyIndex)
        {
            const KeyPoint& kp = obj.keypoints[keyIndex];
            float kpCenterX = std::clamp(kp.x * widthInv, 0.0f, 1.0f);
            float kpCenterY = std::clamp(kp.y * heightInv, 0.0f, 1.0f);
            float kpWidth = std::min(keypointSizeNorm, 1.0f);
            float kpHeight = std::min(keypointSizeNorm, 1.0f);
            float kpLeft = std::clamp(kpCenterX - kpWidth * 0.5f, 0.0f, 1.0f - kpWidth);
            float kpTop = std::clamp(kpCenterY - kpHeight * 0.5f, 0.0f, 1.0f - kpHeight);

            overlay::DetectionEntry keyEntry{};
            keyEntry.bbox = glm::vec4(kpLeft, kpTop, kpWidth, kpHeight);
            keyEntry.color = keyColor;
            keyEntry.confidence = kp.prob;
            keyEntry.classId = static_cast<int>(100 + keyIndex);
            output.push_back(keyEntry);
        }
    }
}
} // namespace

int main(int argc, char** argv)
{
    CliOptions cli = parseCliOptions(argc, argv);
    PoseOverlay poseOverlay(cli.videoPath);

    std::unique_ptr<PoseDetector> poseDetector;
    if (cli.poseEnabled)
    {
        poseDetector = std::make_unique<PoseDetector>();
        std::string detectorError;
        if (!poseDetector->initialize(cli.poseModelBase, detectorError))
        {
            std::cerr << "[motive2d] Pose detection unavailable: " << detectorError << "\n";
            poseDetector.reset();
        }
    }
    std::vector<PoseObject> poseObjects;
    std::vector<uint8_t> poseBgrBuffer;
    std::vector<overlay::DetectionEntry> detectionEntries;
    detectionEntries.reserve(64);
    poseObjects.reserve(8);
    double lastDetectedPosePts = -1.0;
    bool detectionTogglePrev = false;

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
    auto& rectOverlayCompute = engine.getRectOverlayCompute();
    auto& poseOverlayCompute = engine.getPoseOverlayCompute();

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
                    double progress = (x - ui.left) / (ui.right - ui.left);
                    const float seekTime = scrubProgressUi * engine.getDuration();
                    engine.seek(seekTime);
                    playbackState.lastDisplayedSeconds = seekTime;
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

        const bool poseReady = poseDetector && poseDetector->isValid();
        if (poseReady && detectionEnabled && !detectionTogglePrev)
        {
            lastDetectedPosePts = -1.0;
        }

        if (poseReady && detectionEnabled)
        {
            const auto& decoder = playbackState.decoder;
            if (decoder.outputFormat == PrimitiveYuvFormat::NV12 &&
                decoder.bytesPerComponent == 1 &&
                !playbackState.pendingFrames.empty())
            {
                const auto& frame = playbackState.pendingFrames.front();
                const size_t requiredBytes = static_cast<size_t>(decoder.yPlaneBytes) + static_cast<size_t>(decoder.uvPlaneBytes);
                if (frame.buffer.size() >= requiredBytes &&
                    frame.ptsSeconds != lastDetectedPosePts)
                {
                    lastDetectedPosePts = frame.ptsSeconds;
                    if (convertNv12ToBgr(frame.buffer.data(),
                                         decoder.yPlaneBytes,
                                         decoder.uvPlaneBytes,
                                         decoder.width,
                                         decoder.height,
                                         poseBgrBuffer))
                    {
                        if (poseDetector->detect(poseBgrBuffer.data(),
                                                 decoder.width,
                                                 decoder.height,
                                                 poseObjects))
                        {
                            buildDetectionEntries(poseObjects,
                                                  decoder.width,
                                                  decoder.height,
                                                  detectionEntries);
                        }
                        else
                        {
                            detectionEntries.clear();
                        }
                    }
                    else
                    {
                        detectionEntries.clear();
                    }
                }
            }
        }
        detectionTogglePrev = detectionEnabled;

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

        const uint32_t currentFrameIndex = getFrameIndex(playbackSeconds, playbackState.decoder.fps);
        const auto& savedOverlayEntries = poseOverlay.entriesForFrame(currentFrameIndex);
        const bool hasSavedOverlay = !savedOverlayEntries.empty();
        const overlay::DetectionEntry* savedOverlayData = hasSavedOverlay ? savedOverlayEntries.data() : nullptr;

        glm::vec2 overlayRectCenter = rectCenter;
        glm::vec2 overlayRectSize(rectWidth, rectHeight);
        float overlayOuterThickness = 3.0f;
        float overlayInnerThickness = 3.0f;
        bool overlayActive = detectionEnabled || hasSavedOverlay || !detectionEntries.empty();


        uint32_t overlayCount = 0;
        const overlay::DetectionEntry* overlaySource = nullptr;
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
        overlay::runPoseOverlayCompute(&engine,
                                       poseOverlayCompute,
                                       playbackState.poseOverlayImage,
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
        overlay::runRectOverlayCompute(&engine,
                                       rectOverlayCompute,
                                       playbackState.poseOverlayImage,
                                       playbackState.overlay.image,
                                       fbWidth,
                                       fbHeight,
                                       overlayRectCenter,
                                       overlayRectSize,
                                       overlayOuterThickness,
                                       overlayInnerThickness,
                                       rectDetectionFlag,
                                       overlayActive ? 1.0f : 0.0f);
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
