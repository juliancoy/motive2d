#include "engine2d.h"
#include "overlay.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

//
#include <glm/vec4.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define NCNN_VULKAN 1
#define NCNN_SIMPLEVK 0
#include <ncnn/layer.h>
#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <ncnn/option.h>

namespace yolo11l
{

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

static void nms_sorted_bboxes(const std::vector<PoseObject>& objects, std::vector<int>& picked, float threshold)
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
        const PoseObject& obj = objects[i];
        areas[i] = obj.width * obj.height;
    }

    for (size_t i = 0; i < n; ++i)
    {
        const PoseObject& a = objects[i];
        bool keep = true;
        for (int idx : picked)
        {
            const PoseObject& b = objects[idx];
            const float inter = intersection_area(a, b);
            const float uni = areas[i] + areas[idx] - inter;
            if (uni > 0.0f && inter / uni > threshold)
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
                obj.width = std::max(1.0f, x1 - x0);
                obj.height = std::max(1.0f, y1 - y0);
                obj.score = score;
                obj.keypoints.reserve(static_cast<size_t>(num_points));

                ncnn::Mat pred_points_clone = pred_points_grid.clone();
                for (int k = 0; k < num_points; ++k)
                {
                    const float* point_row = pred_points_clone.row(k);
                    KeyPoint kp;
                    kp.x = point_row[0];
                    kp.y = point_row[1];
                    kp.prob = sigmoid(point_row[2]);
                    obj.keypoints.push_back(kp);
                }

                objects.push_back(std::move(obj));
            }
        }

        pred_row_offset += num_grid;
    }
}

static bool convertNv12ToBgr(const uint8_t* nv12,
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
    const size_t requiredSize = framePixels * 3;
    if (bgr.size() < requiredSize)
    {
        bgr.resize(requiredSize);
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

static void buildDetectionEntries(const std::vector<PoseObject>& objects,
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

class Yolo11LPosePipeline
{
public:
    Yolo11LPosePipeline() = default;
    ~Yolo11LPosePipeline()
    {
        shutdown();
    }

    bool initialize(Engine2D* enginePtr, const std::filesystem::path& modelBase)
    {
        if (!enginePtr)
        {
            std::cerr << "[Yolo11LPose] Invalid Engine2D reference\n";
            return false;
        }
        engine = enginePtr;

        auto [paramPath, binPath] = deriveModelPaths(modelBase);
        if (paramPath.empty() || binPath.empty())
        {
            std::cerr << "[Yolo11LPose] Cannot deduce .param/.bin from " << modelBase << "\n";
            return false;
        }

        if (!std::filesystem::exists(paramPath) || !std::filesystem::exists(binPath))
        {
            std::cerr << "[Yolo11LPose] Missing model files: " << paramPath << " / " << binPath << "\n";
            return false;
        }

        if (net.load_param(paramPath.string().c_str()) != 0)
        {
            std::cerr << "[Yolo11LPose] Failed to load " << paramPath << "\n";
            return false;
        }
        if (net.load_model(binPath.string().c_str()) != 0)
        {
            std::cerr << "[Yolo11LPose] Failed to load " << binPath << "\n";
            return false;
        }

        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.set_vulkan_device(0);
        modelLoaded = true;

        architectureDescription = readArchitecture(modelBase);
        if (!architectureDescription.empty())
        {
            std::istringstream iss(architectureDescription);
            std::string line;
            int lines = 0;
            while (lines < 4 && std::getline(iss, line))
            {
                std::cout << "[Yolo11LPose] Architecture: " << line << "\n";
                ++lines;
            }
        }

        if (!createTimelineSemaphore())
        {
            std::cerr << "[Yolo11LPose] Failed to create timeline semaphore\n";
            return false;
        }

        return true;
    }

    void shutdown()
    {
        if (semaphore != VK_NULL_HANDLE && engine)
        {
            vkDestroySemaphore(engine->logicalDevice, semaphore, nullptr);
            semaphore = VK_NULL_HANDLE;
        }
        modelLoaded = false;
    }

    bool detectFrame(const uint8_t* nv12,
                     size_t yBytes,
                     size_t uvBytes,
                     int width,
                     int height,
                     std::vector<overlay::DetectionEntry>& output)
    {
        if (!modelLoaded || !engine || !nv12 || width <= 0 || height <= 0)
        {
            output.clear();
            return false;
        }

        if (!convertNv12ToBgr(nv12, yBytes, uvBytes, width, height, bgr))
        {
            std::cerr << "[Yolo11LPose] Failed to convert NV12 frame\n";
            output.clear();
            return false;
        }

        return processDetection(width, height, output);
    }

    bool detectFromBgr(const uint8_t* bgrData,
                       int width,
                       int height,
                       std::vector<overlay::DetectionEntry>& output)
    {
        if (!modelLoaded || !engine || !bgrData || width <= 0 || height <= 0)
        {
            output.clear();
            return false;
        }

        const size_t frameBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
        bgr.resize(frameBytes);
        std::memcpy(bgr.data(), bgrData, frameBytes);
        return processDetection(width, height, output);
    }

private:
    bool processDetection(int width,
                          int height,
                          std::vector<overlay::DetectionEntry>& output)
    {
        if (width <= 0 || height <= 0 || bgr.empty())
        {
            output.clear();
            return false;
        }

        int targetW = width;
        int targetH = height;
        float scale = 1.0f;
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

        constexpr int max_stride = 32;
        const int wpad = ((targetW + max_stride - 1) / max_stride) * max_stride - targetW;
        const int hpad = ((targetH + max_stride - 1) / max_stride) * max_stride - targetH;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data(),
                                                     ncnn::Mat::PIXEL_BGR2RGB,
                                                     width,
                                                     height,
                                                     targetW,
                                                     targetH);

        ncnn::Mat in_pad;
        ncnn::copy_make_border(in,
                               in_pad,
                               hpad / 2,
                               hpad - hpad / 2,
                               wpad / 2,
                               wpad - wpad / 2,
                               ncnn::BORDER_CONSTANT,
                               114.0f);

        const float norm_vals[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
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
            output.clear();
            signalCompletion();
            return true;
        }

        std::sort(proposals.begin(), proposals.end(), [](const PoseObject& a, const PoseObject& b) {
            return a.score > b.score;
        });

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nmsThreshold);

        const float padW = wpad / 2.0f;
        const float padH = hpad / 2.0f;
        std::vector<PoseObject> finalObjects;
        finalObjects.reserve(picked.size());

        for (int idx : picked)
        {
            PoseObject obj = proposals[idx];
            float x0 = (obj.x - padW) / scale;
            float y0 = (obj.y - padH) / scale;
            float x1 = (obj.x + obj.width - padW) / scale;
            float y1 = (obj.y + obj.height - padH) / scale;

            x0 = std::clamp(x0, 0.0f, static_cast<float>(width - 1));
            y0 = std::clamp(y0, 0.0f, static_cast<float>(height - 1));
            x1 = std::clamp(x1, 0.0f, static_cast<float>(width - 1));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(height - 1));

            obj.x = x0;
            obj.y = y0;
            obj.width = std::max(1.0f, x1 - x0);
            obj.height = std::max(1.0f, y1 - y0);

            for (auto& kp : obj.keypoints)
            {
                float kpX = (kp.x - padW) / scale;
                float kpY = (kp.y - padH) / scale;
                kp.x = std::clamp(kpX, 0.0f, static_cast<float>(width - 1));
                kp.y = std::clamp(kpY, 0.0f, static_cast<float>(height - 1));
            }

            finalObjects.push_back(std::move(obj));
        }

        buildDetectionEntries(finalObjects, width, height, output);
        signalCompletion();
        return true;
    }

    VkSemaphore getSemaphore() const
    {
        return semaphore;
    }

    uint64_t latestValue() const
    {
        return timelineValue;
    }

private:
    static std::pair<std::filesystem::path, std::filesystem::path> deriveModelPaths(const std::filesystem::path& base)
    {
        std::filesystem::path param;
        std::filesystem::path bin;
        if (base.has_extension())
        {
            if (base.extension() == ".param")
            {
                param = base;
                bin = base;
                bin.replace_extension(".bin");
            }
            else if (base.extension() == ".bin")
            {
                bin = base;
                param = base;
                param.replace_extension(".param");
            }
        }

        if (param.empty() || bin.empty())
        {
            param = base;
            param += ".ncnn.param";
            bin = base;
            bin += ".ncnn.bin";
        }

        return {param, bin};
    }

    static std::string findArchitectureName(const std::filesystem::path& base)
    {
        if (!base.has_filename())
        {
            return {};
        }
        return base.stem().string() + ".torchscript.architecture";
    }

    static std::string readArchitecture(const std::filesystem::path& base)
    {
        const auto fileName = findArchitectureName(base);
        if (fileName.empty())
        {
            return {};
        }
        std::filesystem::path archPath = base.parent_path() / fileName;
        if (!std::filesystem::exists(archPath))
        {
            archPath = base;
            archPath += ".torchscript.architecture";
            if (!std::filesystem::exists(archPath))
            {
                return {};
            }
        }

        std::ifstream file(archPath, std::ios::in);
        if (!file)
        {
            return {};
        }
        std::ostringstream oss;
        oss << file.rdbuf();
        return oss.str();
    }

    bool createTimelineSemaphore()
    {
        if (!engine)
        {
            return false;
        }

        VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineInfo.initialValue = timelineValue;

        VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        createInfo.pNext = &timelineInfo;

        if (vkCreateSemaphore(engine->logicalDevice, &createInfo, nullptr, &semaphore) != VK_SUCCESS)
        {
            semaphore = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    void signalCompletion()
    {
        if (!engine || semaphore == VK_NULL_HANDLE)
        {
            return;
        }
        timelineValue += 1;
        VkSemaphoreSignalInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signalInfo.semaphore = semaphore;
        signalInfo.value = timelineValue;
        vkSignalSemaphore(engine->logicalDevice, &signalInfo);
    }

    Engine2D* engine = nullptr;
    ncnn::Net net;
    std::vector<uint8_t> bgr;
    std::vector<int> strides{8, 16, 32};
    const int targetSize = 640;
    const float probThreshold = 0.25f;
    const float nmsThreshold = 0.45f;
    bool modelLoaded = false;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    uint64_t timelineValue = 0;
    std::string architectureDescription;
};

} // namespace yolo11l

namespace
{
void writePixel(uint8_t* data, int width, int height, int x, int y, const glm::vec4& color)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        return;
    }
    const std::array<uint8_t, 3> packed = {
        static_cast<uint8_t>(std::clamp(color.z * 255.0f, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(color.y * 255.0f, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(color.x * 255.0f, 0.0f, 255.0f)),
    };
    const size_t index = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
    data[index + 0] = packed[0];
    data[index + 1] = packed[1];
    data[index + 2] = packed[2];
}

void drawRectangle(uint8_t* data,
                   int width,
                   int height,
                   int x0,
                   int y0,
                   int x1,
                   int y1,
                   int thickness,
                   const glm::vec4& color)
{
    for (int t = 0; t < thickness; ++t)
    {
        const int top = y0 + t;
        const int bottom = y1 - t;
        const int left = x0 + t;
        const int right = x1 - t;

        for (int dx = left; dx <= right; ++dx)
        {
            writePixel(data, width, height, dx, top, color);
            writePixel(data, width, height, dx, bottom, color);
        }
        for (int dy = top; dy <= bottom; ++dy)
        {
            writePixel(data, width, height, left, dy, color);
            writePixel(data, width, height, right, dy, color);
        }
    }
}

void drawCircle(uint8_t* data,
                int width,
                int height,
                int centerX,
                int centerY,
                int radius,
                const glm::vec4& color)
{
    const int sqRadius = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            if (dx * dx + dy * dy <= sqRadius)
            {
                writePixel(data, width, height, centerX + dx, centerY + dy, color);
            }
        }
    }
}

void applyOverlay(uint8_t* data, int width, int height, const std::vector<overlay::DetectionEntry>& entries)
{
    constexpr int boxThickness = 3;
    constexpr int keypointRadius = 4;
    for (const auto& entry : entries)
    {
        const int x0 = static_cast<int>(entry.bbox.x * width);
        const int y0 = static_cast<int>(entry.bbox.y * height);
        const int x1 = static_cast<int>((entry.bbox.x + entry.bbox.z) * width);
        const int y1 = static_cast<int>((entry.bbox.y + entry.bbox.w) * height);

        if (entry.classId >= 100)
        {
            const int centerX = (x0 + x1) / 2;
            const int centerY = (y0 + y1) / 2;
            drawCircle(data, width, height, centerX, centerY, keypointRadius, entry.color);
        }
        else
        {
            drawRectangle(data, width, height, x0, y0, x1, y1, boxThickness, entry.color);
        }
    }
}

void convertBgrToRgb(const uint8_t* bgr, uint8_t* rgb, int pixelCount)
{
    for (int i = 0; i < pixelCount; ++i)
    {
        rgb[i * 3 + 0] = bgr[i * 3 + 2];
        rgb[i * 3 + 1] = bgr[i * 3 + 1];
        rgb[i * 3 + 2] = bgr[i * 3 + 0];
    }
}
} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path modelBase("models/yolo11l-pose");
    std::filesystem::path imagePath("camrichard.jpg");
    if (argc > 1)
    {
        imagePath = argv[1];
    }

    Engine2D engine;
    if (!engine.initialize())
    {
        std::cerr << "[yolo11l-pose] Failed to initialize Engine2D.\n";
        return 1;
    }

    yolo11l::Yolo11LPosePipeline pipeline;
    if (!pipeline.initialize(&engine, modelBase))
    {
        std::cerr << "[yolo11l-pose] Pipeline initialization failed.\n";
        return 1;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = stbi_load(imagePath.string().c_str(), &width, &height, &channels, 3);
    if (!pixels)
    {
        std::cerr << "[yolo11l-pose] Failed to load image " << imagePath << "\n";
        return 1;
    }

    std::vector<overlay::DetectionEntry> entries;
    if (!pipeline.detectFromBgr(pixels, width, height, entries))
    {
        std::cerr << "[yolo11l-pose] Pose detection failed.\n";
        stbi_image_free(pixels);
        return 1;
    }

    applyOverlay(pixels, width, height, entries);

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> rgb(pixelCount * 3);
    convertBgrToRgb(pixels, rgb.data(), static_cast<int>(pixelCount));
    stbi_image_free(pixels);

    std::filesystem::path outputPath = imagePath.parent_path();
    const std::string stem = imagePath.stem().string() + "_pose.jpg";
    outputPath /= stem;
    if (!stbi_write_jpg(outputPath.string().c_str(), width, height, 3, rgb.data(), 95))
    {
        std::cerr << "[yolo11l-pose] Failed to write overlay image " << outputPath << "\n";
        return 1;
    }

    std::cout << "[yolo11l-pose] Image " << outputPath << " written (" << width << "x" << height << ").\n";
    std::cout << "[yolo11l-pose] Detection entries: " << entries.size() << "\n";
    for (size_t i = 0; i < entries.size(); ++i)
    {
        const auto& entry = entries[i];
        std::cout << "  Entry " << i << ": classId=" << entry.classId
                  << " confidence=" << entry.confidence
                  << " bbox=(" << entry.bbox.x << "," << entry.bbox.y
                  << " " << entry.bbox.z << "x" << entry.bbox.w << ")\n";
    }

    return 0;
}
