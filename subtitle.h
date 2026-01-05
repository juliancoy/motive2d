#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <vulkan/vulkan.h>

#include "image_resource.h"
#include "display2d.h"
#include "engine2d.h"
#include "fps.h"
#include "text.h"

class Engine2D;

struct SubtitleLineResource
{
    ImageResource image;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct SubtitleOverlayResources
{
    SubtitleLineResource lines[2];
    bool active = false;
};

fonts::FontBitmap prepareLineBitmap(const std::string &text, uint32_t fontSize)
{
    return fonts::renderText(text, fontSize);
}

struct stringDescriptor
{
    glm::ivec2 origin{0, 0};
    glm::ivec2 size{0, 0};
};

struct SubtitlePushConstants
{
    glm::ivec2 imageSize{0, 0};
    glm::vec4 backgroundColor{0.0f, 0.0f, 0.0f, 0.55f};
    glm::vec4 textColor{1.0f};
    glm::ivec2 lineOrigin[2];
    glm::ivec2 lineSize[2];
    int32_t lineCount = 0;
};

struct Line
{
    double start = 0.0;
    double end = 0.0;
    std::string text;
};


struct SubtitleWord
{
    double start = 0.0;
    double end = 0.0;
    std::string text;
};

struct SubtitleSegment
{
    double start = 0.0;
    double end = 0.0;
    std::string text;
    std::vector<SubtitleWord> words;
};


class Subtitle
{
public:
    Subtitle(const std::filesystem::path &path, Engine2D* engine);
    ~Subtitle();
    bool run(
                const std::stringDescriptor* lineDescriptors,
                uint32_t lineCount,
                VkSampler glyphSampler,
                bool enableBackground);
    Engine2D* engine = nullptr;

    bool load(const std::filesystem::path &path);
    bool hasData() const { return !lines_.empty(); }
    std::vector<const Line *> activeLines(double currentTime, size_t maxLines = 2) const;

    std::vector<Line> lines_;
    mutable size_t lastIndex_ = 0;

    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    bool updateSubtitleOverlay(Engine2D *engine,
                               SubtitleOverlayResources &resources,
                               const Subtitle &overlay,
                               double currentTime,
                               uint32_t fbWidth,
                               uint32_t fbHeight,
                               glm::vec2 overlayCenter,
                               glm::vec2 overlaySize,
                               ImageResource &overlayTarget,
                               VkSampler overlaySampler,
                               VkSampler fallbackSampler,
                               size_t maxLines = 2,
                               bool enableBackground = true);

    void destroySubtitleOverlayResources(Engine2D *engine, SubtitleOverlayResources &resources);
};
class Engine2D;
