// color_grading_pass.h
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

class Engine2D;


constexpr size_t kCurveLutSize = 256;

struct ColorAdjustments
{
    float exposure = 0.0f;
    float contrast = 1.0f;
    float saturation = 1.0f;
    glm::vec3 shadows{1.0f};
    glm::vec3 midtones{1.0f};
    glm::vec3 highlights{1.0f};
    std::array<float, kCurveLutSize> curveLut{};
    bool curveEnabled = false;
};


class ColorGrading
{
public:
    struct Output
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkExtent2D extent{0, 0};
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    // You decide framesInFlight (typically swapchain images count).
    ColorGrading(Engine2D* engine, uint32_t framesInFlight);
    ~ColorGrading();

    ColorGrading(const ColorGrading&) = delete;
    ColorGrading& operator=(const ColorGrading&) = delete;

    // Call when the output size/format should change (usually on swapchain recreate).
    void resize(VkExtent2D extent, VkFormat format);

    // Input is an RGBA image view + sampler (from prior pass).
    // The input layout that the shader expects should be consistent (typically SHADER_READ_ONLY_OPTIMAL).
    void setInputRGBA(VkImageView rgbaView, VkSampler rgbaSampler);

    // Optional: if you want the pass itself to handle transitioning the input layout
    // you could add setInputImage(VkImage image, VkImageView view, VkImageLayout* trackedLayout) etc.
    // But simplest is: caller transitions input and passes view+sampler.

    // Per-frame dispatch. Produces output in a known layout (see outputLayout()).
    void dispatch(VkCommandBuffer cmd, uint32_t frameIndex);

    // Query output for a given frame slot.
    Output output(uint32_t frameIndex) const;

    // If you want, expose what layout the output will be in after dispatch.
    // (e.g. SHADER_READ_ONLY_OPTIMAL if you transition it at end)
    VkImageLayout outputLayout(uint32_t frameIndex) const;

    // Adjustments storage (same as you have today)
    ColorAdjustments* adjustments = nullptr;
    
    VkExtent2D outputExtent_{0, 0};
    VkFormat outputFormat_ = VK_FORMAT_UNDEFINED;


private:
    void createPipeline_();
    void destroyPipeline_();

    void createCurveResources_();
    void destroyCurveResources_();
    void applyCurve();
    void uploadCurveData_(const std::array<float, /*kCurveLutSize*/ 256>& curveData);

    void createOutputs_();
    void destroyOutputs_();

    void createDescriptors_();
    void destroyDescriptors_();
    void rebuildDescriptorSets_();

private:
    Engine2D* engine = nullptr;
    uint32_t framesInFlight_ = 0;

    // Input
    VkImageView rgbaView_ = VK_NULL_HANDLE;
    VkSampler rgbaSampler_ = VK_NULL_HANDLE;

    // Pipeline
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;

    // Outputs per frame
    std::vector<VkImage> outImages_;
    std::vector<VkDeviceMemory> outMem_;
    std::vector<VkImageView> outViews_;
    std::vector<VkImageLayout> outLayouts_;

    // Descriptors per frame
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    // Curve UBO
    VkBuffer curveUBO_ = VK_NULL_HANDLE;
    VkDeviceMemory curveUBOMemory_ = VK_NULL_HANDLE;
    void* curveUBOMapped_ = nullptr;
    size_t curveUBOSize_ = 0;

    // Cache/dirty tracking for curve upload (same idea as your current code)
    bool curveUploaded_ = false;
    bool lastCurveEnabled_ = false;
    std::array<float, 256> lastCurveLut_{};
};
