#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "color_adjustments.h"

class Display2D;
class Engine2D;

constexpr int kMaxFramesInFlight = 2;
constexpr float kScrubberMargin = 20.0f;
constexpr float kScrubberHeight = 64.0f;
constexpr float kScrubberMinWidth = 200.0f;
constexpr float kPlayIconSize = 28.0f;

inline const std::array<float, kCurveLutSize>& identityCurveLut()
{
    static std::array<float, kCurveLutSize> lut = [] {
        std::array<float, kCurveLutSize> arr{};
        for (size_t i = 0; i < kCurveLutSize; ++i)
        {
            arr[i] = static_cast<float>(i) / static_cast<float>(kCurveLutSize - 1);
        }
        return arr;
    }();
    return lut;
}

class ColorGrading
{
public:
    explicit ColorGrading(Display2D* display);
    ~ColorGrading();

    void createPipeline(VkPipelineLayout pipelineLayout);
    void destroyPipeline();

    void createCurveResources();
    void destroyCurveResources();

    void createGradingImages();
    void destroyGradingImages();

    void applyCurve(const ColorAdjustments* adjustments);

    void dispatch(VkCommandBuffer commandBuffer,
                  VkPipelineLayout pipelineLayout,
                  VkDescriptorSet descriptorSet,
                  uint32_t groupX,
                  uint32_t groupY);

    VkPipeline pipeline() const { return pipeline_; }
    VkBuffer curveUBO() const { return curveUBO_; }
    VkDeviceSize curveUBOSize() const { return curveUBOSize_; }

    std::vector<VkImage> gradingImages;
    std::vector<VkDeviceMemory> gradingImageMemories;
    std::vector<VkImageView> gradingImageViews;
    std::vector<VkImageLayout> gradingImageLayouts;
    ColorAdjustments* adjustments = nullptr;

    Display2D* display_ = nullptr;
    Engine2D* engine_ = nullptr;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkBuffer curveUBO_ = VK_NULL_HANDLE;
    VkDeviceMemory curveUBOMemory_ = VK_NULL_HANDLE;
    void* curveUBOMapped_ = nullptr;
    VkDeviceSize curveUBOSize_ = sizeof(glm::vec4) * 64;
    std::array<float, kCurveLutSize> lastCurveLut_{};
    bool lastCurveEnabled_ = false;
    bool curveUploaded_ = false;

    void uploadCurveData(const std::array<float, kCurveLutSize>& curveData);
};
