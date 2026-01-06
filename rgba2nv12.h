#pragma once

#include <vulkan/vulkan.h>
#include <glm/vec2.hpp>
#include <vector>
#include <cstdint>
#include <filesystem>

class Engine2D;

struct RgbaToNv12PushConstants
{
    glm::ivec2 rgbaSize;
    glm::ivec2 uvSize;
    int colorSpace;
    int colorRange;
};

class rgba2nv12
{
    public:
    Engine2D* engine = nullptr;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    RgbaToNv12PushConstants pushConstants{};
    uint32_t groupX = 0;
    uint32_t groupY = 0;

    rgba2nv12(Engine2D* engine,
                       uint32_t groupX,
                       uint32_t groupY);
    ~rgba2nv12();
    bool run();

};
// YUV conversion utilities
bool convertNv12ToBgr(const uint8_t* nv12,
                      size_t yBytes,
                      size_t uvBytes,
                      int width,
                      int height,
                      std::vector<uint8_t>& bgr);

bool saveRawFrameData(const std::filesystem::path& path, const uint8_t* data, size_t size);
