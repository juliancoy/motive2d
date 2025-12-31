#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "overlay.hpp"

class PoseOverlay
{
public:
    explicit PoseOverlay(const std::filesystem::path& videoPath);

    bool hasData() const { return valid_; }
    const std::vector<overlay::DetectionEntry>& entriesForFrame(uint32_t frameIndex);

private:
    static constexpr size_t kKeypointCount = 17;
    static constexpr float kKeypointBoxSize = 0.018f;

    struct FramePose
    {
        std::array<glm::vec2, kKeypointCount> keypoints{};
        std::array<bool, kKeypointCount> valid{{}};
        uint32_t validCount = 0;
        glm::vec4 color{0.9f, 0.4f, 0.7f, 1.0f};
    };

    static std::filesystem::path poseCoordsPath(const std::filesystem::path& videoPath);
    bool loadCoordsFile(const std::filesystem::path& coordsPath);
    bool parseJson(const std::string& text);
    void storeFrame(int frame, const std::vector<float>& coords);

    std::unordered_map<int, std::vector<FramePose>> frameData_;
    std::vector<overlay::DetectionEntry> entriesCache_;
    int cachedFrameIndex_ = -1;
    glm::vec4 keypointColor_{0.9f, 0.4f, 0.7f, 1.0f};
    bool valid_ = false;
};
