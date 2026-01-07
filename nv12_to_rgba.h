// nv12toBGR.h  (NV12 -> RGBA, pass owns output)
#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

class Engine2D;

// Push constants must match your compute shader push constant block.
struct nv12toBGRPushConstants
{
    glm::ivec2 rgbaSize{0, 0};
    glm::ivec2 uvSize{0, 0};
    uint32_t colorSpace = 0; // 0=BT.601, 1=BT.709, 2=BT.2020 (match your convention)
    uint32_t colorRange = 1; // 1=full, 0=limited (match your convention)
};

// Minimal output bundle for downstream (present or sampling).
struct PresentInput
{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    VkExtent2D extent{0, 0};
    VkFormat format = VK_FORMAT_UNDEFINED;
};

class Nv12ToRgbaPass
{
public:
    // IMPORTANT:
    // If your GLSL uses:
    //   layout(set=0,binding=0) uniform sampler2D yTex;
    //   layout(set=0,binding=1) uniform sampler2D uvTex;
    // then the descriptor type MUST be VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER.
    Nv12ToRgbaPass(Engine2D* engine,
                   uint32_t framesInFlight,
                   int width,
                   int height,
                   VkDescriptorType inputDescriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    ~Nv12ToRgbaPass();

    Nv12ToRgbaPass(const Nv12ToRgbaPass&) = delete;
    Nv12ToRgbaPass& operator=(const Nv12ToRgbaPass&) = delete;

    // Build pipeline + outputs + descriptors.
    void initialize();

    // If you ever change video dimensions.
    void resize(int width, int height);

    // Inputs must be valid at dispatch time.
    // For VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER you MUST provide samplers.
    void setInputNV12(VkImageView yView,
                      VkImageView uvView,
                      VkSampler ySampler = VK_NULL_HANDLE,
                      VkSampler uvSampler = VK_NULL_HANDLE);

    // Records bind+dispatch into cmd for this in-flight slot.
    // Does NOT begin/end the command buffer.
    void dispatch(VkCommandBuffer cmd, uint32_t frameIndex);

    // Output the pass produced for this slot.
    PresentInput output(uint32_t frameIndex) const;

    // For downstream sampling (ColorGrading samples this output).
    VkImageView outputView(uint32_t frameIndex) const;
    VkImage outputImage(uint32_t frameIndex) const;
    VkSampler outputSampler() const { return outputSampler_; } // linear clamp sampler created by pass

    // Push constants (set per frame)
    nv12toBGRPushConstants pushConstants{};

    uint32_t framesInFlight() const { return framesInFlight_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    void createPipeline_();
    void destroyPipeline_();

    void createOutputs_();
    void destroyOutputs_();

    void createDescriptors_();
    void destroyDescriptors_();
    void rebuildDescriptorSets_();

    void createOutputSampler_();
    void destroyOutputSampler_();

private:
    Engine2D* engine_ = nullptr;

    uint32_t framesInFlight_ = 0;
    int width_ = 0;
    int height_ = 0;

    // For sampler2D NV12 inputs this should be VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER.
    VkDescriptorType inputDescriptorType_ = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    // Inputs (not owned)
    VkImageView yView_ = VK_NULL_HANDLE;
    VkImageView uvView_ = VK_NULL_HANDLE;
    VkSampler ySampler_ = VK_NULL_HANDLE;
    VkSampler uvSampler_ = VK_NULL_HANDLE;

    // Outputs (owned)
    VkFormat outFormat_ = VK_FORMAT_R8G8B8A8_UNORM;
    std::vector<VkImage> outImages_;
    std::vector<VkDeviceMemory> outMem_;
    std::vector<VkImageView> outViews_;
    std::vector<VkImageLayout> outLayouts_;

    // Descriptor infra (owned)
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    // Pipeline
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    // Sampler used when *downstream* wants to sample our RGBA output (ColorGrading)
    VkSampler outputSampler_ = VK_NULL_HANDLE;

    bool initialized_ = false;
};
