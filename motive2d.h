
#include <fstream>
#include <optional>
#include <filesystem>
#include "pose_overlay.h"
#include "rect_overlay.h"

const std::filesystem::path kDefaultVideoPath("P1090533_main8_hevc_fast.mkv");


struct CliOptions
{
    std::filesystem::path videoPath = kDefaultVideoPath;
    std::optional<bool> swapUV;
    bool showInput = true;
    bool showRegion = true;
    bool showGrading = true;
    bool poseEnabled = false;
    bool overlaysEnabled = true;
    bool scrubberEnabled = true;
    bool debugLogging = false;
    std::filesystem::path poseModelBase = "yolov8n_pose";
    bool debugDecode = false;
    bool inputOnly = false;
    bool skipBlit = false;
    bool singleFrame = false;
    std::filesystem::path outputImagePath = "frame.png";
    bool subtitleBackground = true;
};


class Motive2D {   
public:
    void renderFrame();
    Motive2D(CliOptions cliOptions);
    ~Motive2D();
    std::vector<std::unique_ptr<Display2D>> windows;
};


static inline float intersection_area(const PoseObject &a, const PoseObject &b)
{
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    const float width = std::max(0.0f, right - left);
    const float height = std::max(0.0f, bottom - top);
    return width * height;
}
