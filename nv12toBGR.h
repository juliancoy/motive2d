#pragma once

#include <vulkan/vulkan.h>
#include <glm/vec2.hpp>
#include <vector>
#include <cstdint>
#include <filesystem>

class Engine2D;

struct nv12toBGRPushConstants
{
    glm::ivec2 rgbaSize;
    glm::ivec2 uvSize;
    int colorSpace;
    int colorRange;
};

class nv12toBGR
{
    public:
    Engine2D* engine = nullptr;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    nv12toBGRPushConstants pushConstants{};
    uint32_t groupX = 0;
    uint32_t groupY = 0;

    nv12toBGR(Engine2D* engine,
                       uint32_t groupX,
                       uint32_t groupY);
    ~nv12toBGR();
    void createPipeline();
    void run();

    // Get descriptor set layout for allocating descriptor sets
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }

    // YUV conversion utilities
    bool convertNv12ToBgr(const uint8_t* nv12,
                        size_t yBytes,
                        size_t uvBytes,
                        int width,
                        int height,
                        std::vector<uint8_t>& bgr);
};
