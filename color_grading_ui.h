#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <vulkan/vulkan.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "image_resource.h"
#include "color_adjustments.h"
#include "display2d.h"
#include "engine2d.h"
#include "fps.h"

class Engine2D;

enum DrawCommandType : uint32_t
{
    CMD_RECT = 0u,
    CMD_CIRCLE = 1u,
    CMD_LINE = 2u,
    CMD_GRID = 3u,
    CMD_BITMAP = 4u,
};

struct DrawCommand
{
    uint32_t type = 0;
    glm::vec4 color = glm::vec4(0.0f);
    glm::vec4 params = glm::vec4(0.0f);
    glm::vec4 params2 = glm::vec4(0.0f);
};

struct GradingUiCompute
{
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
    VkBuffer commandBufferStorage = VK_NULL_HANDLE;
    VkDeviceMemory commandBufferMemory = VK_NULL_HANDLE;
    VkDeviceSize commandBufferSize = 0;
    VkBuffer glyphDataBuffer = VK_NULL_HANDLE;
    VkDeviceMemory glyphDataMemory = VK_NULL_HANDLE;
    VkDeviceSize glyphDataSize = 0;
};

struct GradingSettings
{
    float exposure = 0.0f;   // stops
    float contrast = 1.0f;   // multiplier
    float saturation = 1.0f; // multiplier
    glm::vec3 shadows = glm::vec3(1.0f);
    glm::vec3 midtones = glm::vec3(1.0f);
    glm::vec3 highlights = glm::vec3(1.0f);
    std::vector<glm::vec2> curves{
        glm::vec2(0.0f, 0.0f),
        glm::vec2(0.33f, 0.33f),
        glm::vec2(0.66f, 0.66f),
        glm::vec2(1.0f, 1.0f)};
    std::array<float, 256> curveLut{};
    bool curveEnabled = false;
};

namespace grading
{

struct SliderLayout
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t margin = 20;
    uint32_t barHeight = 20;
    uint32_t barYStart = 32;
    uint32_t rowSpacing = 12;
    uint32_t handleRadius = 8;
    uint32_t curvesPadding = 16;
    uint32_t curvesHeight = 180;
    uint32_t curvesX0 = 0;
    uint32_t curvesY0 = 0;
    uint32_t curvesX1 = 0;
    uint32_t curvesY1 = 0;
    VkOffset2D offset{0, 0};
    std::vector<std::string> labels;
    // Reset button bounds (overlay-local coords)
    uint32_t resetX0 = 0;
    uint32_t resetY0 = 0;
    uint32_t resetX1 = 0;
    uint32_t resetY1 = 0;
    uint32_t resetWidth = 140;
    uint32_t resetHeight = 36;
    uint32_t loadX0 = 0;
    uint32_t loadY0 = 0;
    uint32_t loadX1 = 0;
    uint32_t loadY1 = 0;
    uint32_t loadWidth = 140;
    uint32_t loadHeight = 36;
    uint32_t saveX0 = 0;
    uint32_t saveY0 = 0;
    uint32_t saveX1 = 0;
    uint32_t saveY1 = 0;
    uint32_t saveWidth = 140;
    uint32_t saveHeight = 36;
    uint32_t previewX0 = 0;
    uint32_t previewY0 = 0;
    uint32_t previewX1 = 0;
    uint32_t previewY1 = 0;
    uint32_t previewWidth = 160;
    uint32_t previewHeight = 36;
    // Detection overlay button bounds
    uint32_t detectionX0 = 0;
    uint32_t detectionY0 = 0;
    uint32_t detectionX1 = 0;
    uint32_t detectionY1 = 0;
    uint32_t detectionWidth = 180;
    uint32_t detectionHeight = 36;
};

bool initializeGradingUiCompute(Engine2D* engine, GradingUiCompute& comp);
void destroyGradingUiCompute(GradingUiCompute& comp);
bool runGradingUiCompute(Engine2D* engine,
                         GradingUiCompute& comp,
                         ImageResource& target,
                         uint32_t width,
                         uint32_t height,
                         const std::vector<DrawCommand>& commands,
                         const std::vector<uint32_t>& glyphData);

// Build/update the grading overlay texture based on current settings; places it centered at bottom.
bool buildGradingOverlay(Engine2D* engine,
                         const GradingSettings& settings,
                         ImageResource& image,
                         OverlayImageInfo& info,
                         uint32_t fbWidth,
                         uint32_t fbHeight,
                         SliderLayout& layout,
                         bool previewEnabled,
                         bool detectionEnabled = false);

// Map a click within the overlay to one of the slider values. Returns true if a value changed.
bool handleOverlayClick(const SliderLayout& layout,
                        double cursorX,
                        double cursorY,
                        GradingSettings& settings,
                        bool doubleClick = false,
                        bool rightClick = false,
                        bool* loadRequested = nullptr,
                        bool* saveRequested = nullptr,
                        bool* previewToggleRequested = nullptr,
                        bool* detectionToggleRequested = nullptr);

void setGradingDefaults(GradingSettings& settings);
bool loadGradingSettings(const std::filesystem::path& path, GradingSettings& settings);
bool saveGradingSettings(const std::filesystem::path& path, const GradingSettings& settings);
void buildCurveLut(const GradingSettings& settings, std::array<float, kCurveLutSize>& outLut);

} // namespace grading
