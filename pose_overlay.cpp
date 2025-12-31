#include "pose_overlay.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr std::array<glm::vec4, 4> kLabelPalette = {
    glm::vec4(0.99f, 0.49f, 0.18f, 1.0f),
    glm::vec4(0.36f, 0.72f, 0.84f, 1.0f),
    glm::vec4(0.42f, 0.88f, 0.46f, 1.0f),
    glm::vec4(0.94f, 0.73f, 0.25f, 1.0f),
};

void skipWhitespace(const char*& ptr, const char* end)
{
    while (ptr < end && std::isspace(static_cast<unsigned char>(*ptr)))
    {
        ++ptr;
    }
}

bool expectChar(const char*& ptr, const char* end, char expected)
{
    skipWhitespace(ptr, end);
    if (ptr >= end || *ptr != expected)
    {
        return false;
    }
    ++ptr;
    return true;
}

bool parseNumber(const char*& ptr, const char* end, double& out)
{
    skipWhitespace(ptr, end);
    if (ptr >= end)
    {
        return false;
    }
    char* parseEnd = nullptr;
    errno = 0;
    double value = std::strtod(ptr, &parseEnd);
    if (parseEnd == ptr)
    {
        return false;
    }
    ptr = parseEnd;
    out = value;
    return true;
}

bool parseJsonString(const char*& ptr, const char* end, std::string& out)
{
    if (!expectChar(ptr, end, '"'))
    {
        return false;
    }
    const char* start = ptr;
    while (ptr < end && *ptr != '"')
    {
        if (*ptr == '\\' && ptr + 1 < end)
        {
            ptr += 2;
        }
        else
        {
            ++ptr;
        }
    }
    if (ptr >= end)
    {
        return false;
    }
    out.assign(start, ptr);
    ++ptr;
    return true;
}

bool parseJsonNumberArray(const char*& ptr, const char* end, std::vector<float>& out)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
    out.clear();
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == ']')
        {
            ++ptr;
            break;
        }
        double value = 0.0;
        if (!parseNumber(ptr, end, value))
        {
            return false;
        }
        out.push_back(static_cast<float>(value));
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        if (*ptr == ']')
        {
            ++ptr;
            break;
        }
        return false;
    }
    return true;
}

bool parseLegacyEntry(const char*& ptr, const char* end, double& frameValue, std::vector<float>& coords)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
    skipWhitespace(ptr, end);
    if (!parseNumber(ptr, end, frameValue))
    {
        return false;
    }
    if (!expectChar(ptr, end, ','))
    {
        return false;
    }
    skipWhitespace(ptr, end);
    if (!parseJsonNumberArray(ptr, end, coords))
    {
        return false;
    }
    if (!expectChar(ptr, end, ']'))
    {
        return false;
    }
    return true;
}

bool parsePoseObject(const char*& ptr, const char* end, double& frameValue, std::vector<float>& coords)
{
    if (!expectChar(ptr, end, '{'))
    {
        return false;
    }
    bool frameSet = false;
    bool poseSet = false;
    std::string key;
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == '}')
        {
            ++ptr;
            break;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        if (!parseJsonString(ptr, end, key))
        {
            return false;
        }
        if (!expectChar(ptr, end, ':'))
        {
            return false;
        }
        skipWhitespace(ptr, end);
        if (key == "frame")
        {
            double value = 0.0;
            if (!parseNumber(ptr, end, value))
            {
                return false;
            }
            frameValue = value;
            frameSet = true;
        }
        else if (key == "pose" || key == "coords")
        {
            if (!parseJsonNumberArray(ptr, end, coords))
            {
                return false;
            }
            poseSet = true;
        }
        else
        {
            return false;
        }
    }

    return frameSet && poseSet;
}

bool parsePoseJsonLine(const std::string& line, int& frameIndex, std::vector<float>& coords)
{
    const char* ptr = line.c_str();
    const char* end = ptr + line.size();
    skipWhitespace(ptr, end);
    double frameValue = 0.0;
    if (!parsePoseObject(ptr, end, frameValue, coords))
    {
        return false;
    }
    skipWhitespace(ptr, end);
    if (ptr != end)
    {
        return false;
    }
    frameIndex = static_cast<int>(frameValue);
    return true;
}
} // namespace

PoseOverlay::PoseOverlay(const std::filesystem::path& videoPath)
{
    const std::filesystem::path coordsPath = poseCoordsPath(videoPath);
    if (coordsPath.empty())
    {
        std::cout << "[pose] No coords file found for " << videoPath << "\n";
        return;
    }
    if (!coordsPath.empty() && loadCoordsFile(coordsPath))
    {
        valid_ = !frameData_.empty() || !detectionData_.empty();
        if (valid_)
        {
            std::cout << "[pose] Loaded overlay for " << coordsPath << " (" << frameData_.size()
                      << " frames)\n";
        }
    }
}

std::filesystem::path PoseOverlay::poseCoordsPath(const std::filesystem::path& videoPath)
{
    if (videoPath.empty())
    {
        return {};
    }
    const std::string stem = videoPath.stem().string();
    if (stem.empty())
    {
        return {};
    }
    std::filesystem::path jsonPath = videoPath.parent_path() / (stem + "_pose_coords.json");
    if (std::filesystem::exists(jsonPath))
    {
        return jsonPath;
    }
    std::filesystem::path txtPath = videoPath.parent_path() / (stem + "_pose_coords.txt");
    if (std::filesystem::exists(txtPath))
    {
        return txtPath;
    }
    return {};
}

bool PoseOverlay::loadCoordsFile(const std::filesystem::path& coordsPath)
{
    if (coordsPath.empty() || !std::filesystem::exists(coordsPath))
    {
        return false;
    }

    std::ifstream file(coordsPath, std::ios::binary);
    if (!file)
    {
        return false;
    }

    frameData_.clear();
    detectionData_.clear();
    labelColors_.clear();

    const std::string extension = coordsPath.extension().string();
    bool parsed = false;
    if (extension == ".txt")
    {
        parsed = parseTxt(file);
    }
    else
    {
        std::string contents;
        contents.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (contents.empty())
        {
            return false;
        }
        parsed = parseJson(contents);
    }
    valid_ = parsed && (!frameData_.empty() || !detectionData_.empty());
    if (!valid_)
    {
        std::cout << "[pose] Failed to parse coords file: " << coordsPath << "\n";
    }
    return valid_;
}

bool PoseOverlay::parseJson(const std::string& text)
{
    frameData_.clear();
    const char* ptr = text.data();
    const char* end = ptr + text.size();
    skipWhitespace(ptr, end);
    if (ptr >= end || *ptr != '[')
    {
        return false;
    }
    ++ptr;

    std::vector<float> coords;
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == ']')
        {
            ++ptr;
            break;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }

        double frameValue = 0.0;
        bool parsedEntry = false;
        if (*ptr == '[')
        {
            parsedEntry = parseLegacyEntry(ptr, end, frameValue, coords);
        }
        else if (*ptr == '{')
        {
            parsedEntry = parsePoseObject(ptr, end, frameValue, coords);
        }
        if (!parsedEntry)
        {
            return false;
        }

        int frame = static_cast<int>(frameValue);
        if (frame >= 0)
        {
            storeFrame(frame, coords);
        }

        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            break;
        }
        if (*ptr == ',')
        {
            ++ptr;
        }
    }

    skipWhitespace(ptr, end);
    return true;
}

bool PoseOverlay::parseTxt(std::istream& stream)
{
    detectionData_.clear();
    struct RawEntry
    {
        int frame = 0;
        std::string label;
        float confidence = 0.0f;
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
    };

    std::vector<RawEntry> entries;
    float maxX = 0.0f;
    float maxY = 0.0f;
    std::string line;
    std::vector<float> coords;
    bool hasPoseData = false;
    while (std::getline(stream, line))
    {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            continue;
        }
        if (line[first] == '#')
        {
            continue;
        }
        if (line[first] == '{')
        {
            int frame = 0;
            if (parsePoseJsonLine(line.c_str() + first, frame, coords))
            {
                if (frame >= 0)
                {
                    storeFrame(frame, coords);
                    hasPoseData = true;
                }
            }
            continue;
        }
        std::istringstream lineStream(line);
        RawEntry entry;
        if (!(lineStream >> entry.frame >> entry.label >> entry.confidence >> entry.x1 >> entry.y1 >> entry.x2 >> entry.y2))
        {
            continue;
        }
        maxX = std::max(maxX, std::max(entry.x1, entry.x2));
        maxY = std::max(maxY, std::max(entry.y1, entry.y2));
        entries.push_back(entry);
    }

    bool detectionValid = false;
    if (!entries.empty())
    {
        const float width = std::max(1.0f, maxX);
        const float height = std::max(1.0f, maxY);
        for (const RawEntry& entry : entries)
        {
            const float minX = std::min(entry.x1, entry.x2);
            const float minY = std::min(entry.y1, entry.y2);
            const float boxW = std::abs(entry.x2 - entry.x1);
            const float boxH = std::abs(entry.y2 - entry.y1);
            if (boxW <= 0.0f || boxH <= 0.0f)
            {
                continue;
            }
            overlay::DetectionEntry det{};
            det.bbox = glm::vec4(minX / width, minY / height, boxW / width, boxH / height);
            det.color = colorForLabel(entry.label);
            det.confidence = entry.confidence;
            det.classId = 0;
            detectionData_[entry.frame].push_back(det);
        }
        detectionValid = !detectionData_.empty();
    }

    return hasPoseData || detectionValid;
}

void PoseOverlay::storeFrame(int frame, const std::vector<float>& coords)
{
    constexpr size_t expectedSize = 5 + 3 * kKeypointCount;
    if (coords.size() < expectedSize)
    {
        return;
    }

    FramePose pose{};
    pose.valid.fill(false);
    pose.validCount = 0;

    for (size_t idx = 0; idx < kKeypointCount; ++idx)
    {
        const size_t baseIndex = 5 + idx * 3;
        if (baseIndex + 1 >= coords.size())
        {
            break;
        }
        const float x = coords[baseIndex];
        const float y = coords[baseIndex + 1];
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            continue;
        }
        if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f)
        {
            continue;
        }
        pose.keypoints[idx] = glm::vec2(x, y);
        pose.valid[idx] = true;
        ++pose.validCount;
    }

    if (pose.validCount == 0)
    {
        return;
    }

    static const std::array<glm::vec4, 3> instanceColors = {
        glm::vec4(0.99f, 0.49f, 0.18f, 1.0f),
        glm::vec4(0.36f, 0.72f, 0.84f, 1.0f),
        glm::vec4(0.42f, 0.88f, 0.46f, 1.0f),
    };
    auto& list = frameData_[frame];
    const size_t colorIndex = list.size() % instanceColors.size();
    pose.color = instanceColors[colorIndex];
    list.push_back(pose);
}

glm::vec4 PoseOverlay::colorForLabel(const std::string& label)
{
    if (label.empty())
    {
        return kLabelPalette[0];
    }
    auto it = labelColors_.find(label);
    if (it != labelColors_.end())
    {
        return it->second;
    }
    const size_t index = labelColors_.size() % kLabelPalette.size();
    const glm::vec4 color = kLabelPalette[index];
    labelColors_.emplace(label, color);
    return color;
}

const std::vector<overlay::DetectionEntry>& PoseOverlay::entriesForFrame(uint32_t frameIndex)
{
    if (!valid_)
    {
        entriesCache_.clear();
        return entriesCache_;
    }

    if (static_cast<int>(frameIndex) == cachedFrameIndex_)
    {
        return entriesCache_;
    }

    cachedFrameIndex_ = static_cast<int>(frameIndex);
    entriesCache_.clear();

    auto detectionIt = detectionData_.find(static_cast<int>(frameIndex));
    if (detectionIt != detectionData_.end())
    {
        entriesCache_ = detectionIt->second;
        return entriesCache_;
    }

    const auto it = frameData_.find(cachedFrameIndex_);
    if (it == frameData_.end())
    {
        return entriesCache_;
    }

    for (const FramePose& pose : it->second)
    {
        for (size_t idx = 0; idx < kKeypointCount; ++idx)
        {
            if (!pose.valid[idx])
            {
                continue;
            }

            glm::vec4 bbox;
            bbox.z = kKeypointBoxSize;
            bbox.w = kKeypointBoxSize;
            bbox.x = std::clamp(pose.keypoints[idx].x - bbox.z * 0.5f, 0.0f, 1.0f - bbox.z);
            bbox.y = std::clamp(pose.keypoints[idx].y - bbox.w * 0.5f, 0.0f, 1.0f - bbox.w);

            overlay::DetectionEntry entry{};
            entry.bbox = bbox;
            entry.color = pose.color;
            entry.confidence = 1.0f;
            entry.classId = static_cast<int>(100 + idx);
            entriesCache_.push_back(entry);
        }
    }

    return entriesCache_;
}
