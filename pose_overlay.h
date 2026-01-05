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

#include <vulkan/vulkan.h>

#include "image_resource.h"
#include "utils.h"

struct DetectionEntry;
class Engine2D;
namespace nlohmann {
    class json;
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

class PoseOverlay
{
public:
    explicit PoseOverlay(Engine2D *engine);
    ~PoseOverlay();
    
    static std::filesystem::path poseCoordsPath(const std::filesystem::path &videoPath);
    bool loadCoordsFile(const std::filesystem::path &coordsPath);
    void run(ImageResource& target,
             uint32_t width,
             uint32_t height,
             const glm::vec2& rectCenter,
             const glm::vec2& rectSize,
             float outerThickness,
             float innerThickness,
             float detectionEnabled,
             const DetectionEntry* detections,
             uint32_t detectionCount);
    
    bool hasData() const { return valid_; }
    const std::vector<DetectionEntry> &entriesForFrame(uint32_t frameIndex);
    
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

    // Helper functions
    static bool extractCoordsArray(const nlohmann::json& jsonArray, std::vector<float>& out);
    static bool parsePoseEntry(const nlohmann::json& entry, int& frameIndex, std::vector<float>& coords);
    static bool parsePoseJsonLine(const std::string& line, int& frameIndex, std::vector<float>& coords);
    
    bool parseJson(const std::string &text);
    bool parseTxt(std::istream &lines);
    void storeFrame(int frame, const std::vector<float> &coords);
    glm::vec4 colorForLabel(const std::string &label);
    bool ensureDetectionBuffer(VkDeviceSize requestedSize);
    bool ensureImageResource(ImageResource& target, 
                            uint32_t width, 
                            uint32_t height, 
                            VkFormat format, 
                            bool& recreated,
                            VkImageUsageFlags usage);

    std::unordered_map<int, std::vector<FramePose>> frameData_;
    std::unordered_map<int, std::vector<DetectionEntry>> detectionData_;
    std::unordered_map<std::string, glm::vec4> labelColors_;
    std::vector<DetectionEntry> entriesCache_;
    int cachedFrameIndex_ = -1;
    glm::vec4 keypointColor_{0.9f, 0.4f, 0.7f, 1.0f};
    bool valid_ = false;
    Engine2D* engine_ = nullptr;
};
