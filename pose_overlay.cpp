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

PosePoseOverlay(const std::filesystem::path& videoPath)
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

std::filesystem::path PoseposeCoordsPath(const std::filesystem::path& videoPath)
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

bool PoseloadCoordsFile(const std::filesystem::path& coordsPath)
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

bool PoseparseJson(const std::string& text)
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

bool PoseparseTxt(std::istream& stream)
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
            DetectionEntry det{};
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

void PosestoreFrame(int frame, const std::vector<float>& coords)
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

glm::vec4 PosecolorForLabel(const std::string& label)
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

const std::vector<DetectionEntry>& PoseentriesForFrame(uint32_t frameIndex)
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

            DetectionEntry entry{};
            entry.bbox = bbox;
            entry.color = pose.color;
            entry.confidence = 1.0f;
            entry.classId = static_cast<int>(100 + idx);
            entriesCache_.push_back(entry);
        }
    }

    return entriesCache_;
}
#include "debug_logging.h"
#include "fps.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "engine2d.h"
#include "utils.h"

namespace overlay
{
namespace
{
bool ensureDetectionBuffer(Engine2D* engine, PoseOverlayCompute& comp, VkDeviceSize requestedSize)
{
    const VkDeviceSize entrySize = sizeof(DetectionEntry);
    VkDeviceSize requiredSize = std::max(entrySize, requestedSize);

    if (comp.detectionBuffer != VK_NULL_HANDLE && comp.detectionBufferSize >= requiredSize)
    {
        return true;
    }

    if (comp.detectionBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(comp.device, comp.detectionBuffer, nullptr);
        comp.detectionBuffer = VK_NULL_HANDLE;
    }
    if (comp.detectionBufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(comp.device, comp.detectionBufferMemory, nullptr);
        comp.detectionBufferMemory = VK_NULL_HANDLE;
    }

    engine->createBuffer(requiredSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         comp.detectionBuffer,
                         comp.detectionBufferMemory);
    comp.detectionBufferSize = requiredSize;
    return true;
}

struct PoseOverlayPush
{
    glm::vec2 outputSize;
    glm::vec2 rectCenter;
    glm::vec2 rectSize;
    float outerThickness;
    float innerThickness;
    float detectionEnabled;
    uint32_t detectionCount;
};
} // namespace

void destroyPoseOverlayCompute(PoseOverlayCompute& comp)
{
    if (comp.fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(comp.device, comp.fence, nullptr);
        comp.fence = VK_NULL_HANDLE;
    }
    if (comp.commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(comp.device, comp.commandPool, nullptr);
        comp.commandPool = VK_NULL_HANDLE;
    }
    if (comp.descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(comp.device, comp.descriptorPool, nullptr);
        comp.descriptorPool = VK_NULL_HANDLE;
    }
    if (comp.pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(comp.device, comp.pipeline, nullptr);
        comp.pipeline = VK_NULL_HANDLE;
    }
    if (comp.pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        comp.pipelineLayout = VK_NULL_HANDLE;
    }
    if (comp.descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        comp.descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (comp.detectionBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(comp.device, comp.detectionBuffer, nullptr);
        comp.detectionBuffer = VK_NULL_HANDLE;
    }
    if (comp.detectionBufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(comp.device, comp.detectionBufferMemory, nullptr);
        comp.detectionBufferMemory = VK_NULL_HANDLE;
    }
    comp.detectionBufferSize = 0;
}

void runPoseOverlayCompute(Engine2D* engine,
                           PoseOverlayCompute& comp,
                           ImageResource& target,
                           uint32_t width,
                           uint32_t height,
                           const glm::vec2& rectCenter,
                           const glm::vec2& rectSize,
                           float outerThickness,
                           float innerThickness,
                           float detectionEnabled,
                           const DetectionEntry* detections,
                           uint32_t detectionCount)
{
    LOG_DEBUG(std::cout << "[PoseOverlay] Starting pose overlay compute on image: " << target.image 
              << " (view: " << target.view << ")" << std::endl);
    LOG_DEBUG(std::cout << "[PoseOverlay] Target dimensions: " << width << "x" << height << std::endl);
    LOG_DEBUG(std::cout << "[PoseOverlay] Detection count: " << detectionCount 
              << ", detection enabled: " << detectionEnabled << std::endl);
    
    bool recreated = false;
    if (!ensureImageResource(engine, target, width, height, VK_FORMAT_R8G8B8A8_UNORM, recreated,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    {
        LOG_DEBUG(std::cout << "[PoseOverlay] Failed to ensure image resource" << std::endl);
        return;
    }

    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageInfo.imageView = target.view;

    const VkDeviceSize entrySize = sizeof(DetectionEntry);
    VkDeviceSize desiredBufferSize = entrySize;
    if (detectionCount > 0)
    {
        desiredBufferSize = entrySize * static_cast<VkDeviceSize>(detectionCount);
    }
    if (!ensureDetectionBuffer(engine, comp, desiredBufferSize))
    {
        return;
    }

    VkDeviceSize copySize = std::max(entrySize, desiredBufferSize);
    if (comp.detectionBuffer != VK_NULL_HANDLE && comp.detectionBufferMemory != VK_NULL_HANDLE)
    {
        void* mapped = nullptr;
        vkMapMemory(comp.device, comp.detectionBufferMemory, 0, copySize, 0, &mapped);
        if (mapped)
        {
            if (detectionCount > 0 && detections)
            {
                std::memcpy(mapped, detections, entrySize * detectionCount);
            }
            else
            {
                std::memset(mapped, 0, static_cast<size_t>(entrySize));
            }
            vkUnmapMemory(comp.device, comp.detectionBufferMemory);
        }
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = comp.detectionBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = copySize;

    VkWriteDescriptorSet imageWrite{};
    imageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    imageWrite.dstSet = comp.descriptorSet;
    imageWrite.dstBinding = 0;
    imageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageWrite.descriptorCount = 1;
    imageWrite.pImageInfo = &storageInfo;

    VkWriteDescriptorSet bufferWrite{};
    bufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    bufferWrite.dstSet = comp.descriptorSet;
    bufferWrite.dstBinding = 1;
    bufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bufferWrite.descriptorCount = 1;
    bufferWrite.pBufferInfo = &bufferInfo;

    std::array<VkWriteDescriptorSet, 2> writes = {imageWrite, bufferWrite};
    vkUpdateDescriptorSets(comp.device,
                           static_cast<uint32_t>(writes.size()),
                           writes.data(),
                           0,
                           nullptr);

    vkResetCommandBuffer(comp.commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(comp.commandBuffer, &beginInfo);

    PoseOverlayPush push{glm::vec2(static_cast<float>(width), static_cast<float>(height)),
                         rectCenter,
                         rectSize,
                         outerThickness,
                         innerThickness,
                         detectionEnabled,
                         detectionCount};

    const VkImageLayout initialLayout = recreated ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageMemoryBarrier toGeneralBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toGeneralBarrier.oldLayout = initialLayout;
    toGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.image = target.image;
    toGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneralBarrier.subresourceRange.baseMipLevel = 0;
    toGeneralBarrier.subresourceRange.levelCount = 1;
    toGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    toGeneralBarrier.subresourceRange.layerCount = 1;
    toGeneralBarrier.srcAccessMask = (initialLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                         ? VK_ACCESS_SHADER_READ_BIT
                                         : 0;
    toGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    VkPipelineStageFlags srcStage = (initialLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                        ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                        : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    vkCmdPipelineBarrier(comp.commandBuffer,
                         srcStage,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toGeneralBarrier);

    vkCmdBindPipeline(comp.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, comp.pipeline);
    vkCmdBindDescriptorSets(comp.commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            comp.pipelineLayout,
                            0,
                            1,
                            &comp.descriptorSet,
                            0,
                            nullptr);
    vkCmdPushConstants(comp.commandBuffer,
                       comp.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PoseOverlayPush),
                       &push);

    const uint32_t groupX = (width + 15) / 16;
    const uint32_t groupY = (height + 15) / 16;
    vkCmdDispatch(comp.commandBuffer, groupX, groupY, 1);

    VkImageMemoryBarrier toReadBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toReadBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadBarrier.image = target.image;
    toReadBarrier.subresourceRange = toGeneralBarrier.subresourceRange;
    toReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(comp.commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toReadBarrier);

    vkEndCommandBuffer(comp.commandBuffer);

    vkResetFences(comp.device, 1, &comp.fence);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &comp.commandBuffer;
    vkQueueSubmit(engine->graphicsQueue, 1, &submitInfo, comp.fence);
    vkWaitForFences(comp.device, 1, &comp.fence, VK_TRUE, UINT64_MAX);
}

bool initializePoseOverlayCompute(Engine2D* engine, PoseOverlayCompute& comp)
{
    comp.device = engine->logicalDevice;
    comp.queue = engine->graphicsQueue;

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(comp.device, &layoutInfo, nullptr, &comp.descriptorSetLayout) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create pose overlay descriptor set layout" << std::endl;
        return false;
    }

    std::vector<char> shaderCode;
    try
    {
        shaderCode = readSPIRVFile("shaders/overlay_pose.comp.spv");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Video2D] " << ex.what() << std::endl;
        destroyPoseOverlayCompute(comp);
        return false;
    }

    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PoseOverlayPush);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &comp.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(comp.device, &pipelineLayoutInfo, nullptr, &comp.pipelineLayout) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create pose overlay pipeline layout" << std::endl;
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
        destroyPoseOverlayCompute(comp);
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = comp.pipelineLayout;

    if (vkCreateComputePipelines(comp.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &comp.pipeline) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create pose overlay compute pipeline" << std::endl;
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
        destroyPoseOverlayCompute(comp);
        return false;
    }

    vkDestroyShaderModule(comp.device, shaderModule, nullptr);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(comp.device, &poolInfo, nullptr, &comp.descriptorPool) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create pose overlay descriptor pool" << std::endl;
        destroyPoseOverlayCompute(comp);
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = comp.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &comp.descriptorSetLayout;

    if (vkAllocateDescriptorSets(comp.device, &allocInfo, &comp.descriptorSet) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to allocate pose overlay descriptor set" << std::endl;
        destroyPoseOverlayCompute(comp);
        return false;
    }

    VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCreateInfo.queueFamilyIndex = engine->graphicsQueueFamilyIndex;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(comp.device, &poolCreateInfo, nullptr, &comp.commandPool) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create pose overlay command pool" << std::endl;
        destroyPoseOverlayCompute(comp);
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = comp.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(comp.device, &cmdAllocInfo, &comp.commandBuffer) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to allocate pose overlay command buffer" << std::endl;
        destroyPoseOverlayCompute(comp);
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(comp.device, &fenceInfo, nullptr, &comp.fence) != VK_SUCCESS)
    {
        std::cerr << "[Video2D] Failed to create pose overlay fence" << std::endl;
        destroyPoseOverlayCompute(comp);
        return false;
    }

    return true;
}

} // namespace overlay
