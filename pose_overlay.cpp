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
#include <vector>

namespace
{
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

} // namespace

PoseOverlay::PoseOverlay(const std::filesystem::path& videoPath)
{
    const std::filesystem::path coordsPath = poseCoordsPath(videoPath);
    if (!coordsPath.empty() && loadCoordsFile(coordsPath))
    {
        valid_ = !frameData_.empty();
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
    return videoPath.parent_path() / (stem + "_pose_coords.json");
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

    std::string contents;
    contents.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (contents.empty())
    {
        return false;
    }

    const bool parsed = parseJson(contents);
    valid_ = parsed && !frameData_.empty();
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
        if (!expectChar(ptr, end, '['))
        {
            return false;
        }

        skipWhitespace(ptr, end);
        double frameValue = 0.0;
        if (!parseNumber(ptr, end, frameValue))
        {
            return false;
        }
        skipWhitespace(ptr, end);
        if (!expectChar(ptr, end, ','))
        {
            return false;
        }
        skipWhitespace(ptr, end);
        if (!expectChar(ptr, end, '['))
        {
            return false;
        }

        coords.clear();
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
            coords.push_back(static_cast<float>(value));
            skipWhitespace(ptr, end);
            if (ptr < end && *ptr == ',')
            {
                ++ptr;
            }
        }

        skipWhitespace(ptr, end);
        if (!expectChar(ptr, end, ']'))
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
            continue;
        }
        if (*ptr == ']')
        {
            ++ptr;
            break;
        }

        // Unexpected character between entries
        return false;
    }

    skipWhitespace(ptr, end);
    return true;
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
