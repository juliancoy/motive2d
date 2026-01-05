#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "fps.h"

class PoseOverlay
{
public:
    explicit PoseOverlay(const std::filesystem::path& videoPath);

    bool hasData() const { return valid_; }
    const std::vector<DetectionEntry>& entriesForFrame(uint32_t frameIndex);

    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDeviceSize detectionBufferSize = 0;
    VkBuffer detectionBuffer = VK_NULL_HANDLE;
    VkDeviceMemory detectionBufferMemory = VK_NULL_HANDLE;

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
    bool parseTxt(std::istream& lines);
    void storeFrame(int frame, const std::vector<float>& coords);
    glm::vec4 colorForLabel(const std::string& label);

    std::unordered_map<int, std::vector<FramePose>> frameData_;
    std::unordered_map<int, std::vector<DetectionEntry>> detectionData_;
    std::unordered_map<std::string, glm::vec4> labelColors_;
    std::vector<DetectionEntry> entriesCache_;
    int cachedFrameIndex_ = -1;
    glm::vec4 keypointColor_{0.9f, 0.4f, 0.7f, 1.0f};
    bool valid_ = false;
};
