#include "subtitle_overlay.hpp"

#include "engine2d.h"
#include "fonts.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"
#include <glm/vec4.hpp>
namespace
{
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

bool parseJsonString(const char*& ptr, const char* end, std::string& out)
{
    if (!expectChar(ptr, end, '"'))
    {
        return false;
    }
    const char* start = ptr;
    std::string result;
    while (ptr < end)
    {
        char ch = *ptr;
        if (ch == '\\' && ptr + 1 < end)
        {
            ++ptr;
            char escaped = *ptr;
            switch (escaped)
            {
                case '\\':
                case '/':
                case '"':
                    result += escaped;
                    break;
                case 'b':
                    result += '\b';
                    break;
                case 'f':
                    result += '\f';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 'r':
                    result += '\r';
                    break;
                case 't':
                    result += '\t';
                    break;
                default:
                    result += escaped;
                    break;
            }
            ++ptr;
            continue;
        }
        if (ch == '"')
        {
            break;
        }
        result += ch;
        ++ptr;
    }
    if (ptr >= end || *ptr != '"')
    {
        return false;
    }
    ++ptr;
    out = std::move(result);
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

bool skipValue(const char*& ptr, const char* end);

bool skipObject(const char*& ptr, const char* end)
{
    if (!expectChar(ptr, end, '{'))
    {
        return false;
    }
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
            return true;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        std::string key;
        if (!parseJsonString(ptr, end, key))
        {
            return false;
        }
        if (!expectChar(ptr, end, ':'))
        {
            return false;
        }
        if (!skipValue(ptr, end))
        {
            return false;
        }
    }
    return true;
}

bool skipArray(const char*& ptr, const char* end)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
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
            return true;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        if (!skipValue(ptr, end))
        {
            return false;
        }
    }
    return true;
}

bool skipValue(const char*& ptr, const char* end)
{
    skipWhitespace(ptr, end);
    if (ptr >= end)
    {
        return false;
    }
    char ch = *ptr;
    if (ch == '{')
    {
        return skipObject(ptr, end);
    }
    if (ch == '[')
    {
        return skipArray(ptr, end);
    }
    if (ch == '"')
    {
        std::string dummy;
        return parseJsonString(ptr, end, dummy);
    }
    if (ch == '-' || ch == '+' || (ch >= '0' && ch <= '9'))
    {
        double dummy;
        return parseNumber(ptr, end, dummy);
    }
    if (std::strncmp(ptr, "true", 4) == 0)
    {
        ptr += 4;
        return true;
    }
    if (std::strncmp(ptr, "false", 5) == 0)
    {
        ptr += 5;
        return true;
    }
    if (std::strncmp(ptr, "null", 4) == 0)
    {
        ptr += 4;
        return true;
    }
    return false;
}

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

bool parseWordObject(const char*& ptr, const char* end, SubtitleWord& word)
{
    if (!expectChar(ptr, end, '{'))
    {
        return false;
    }
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
        std::string key;
        if (!parseJsonString(ptr, end, key))
        {
            return false;
        }
        if (!expectChar(ptr, end, ':'))
        {
            return false;
        }
        if (key == "word")
        {
            parseJsonString(ptr, end, word.text);
        }
        else if (key == "start")
        {
            parseNumber(ptr, end, word.start);
        }
        else if (key == "end")
        {
            parseNumber(ptr, end, word.end);
        }
        else
        {
            if (!skipValue(ptr, end))
            {
                return false;
            }
        }
    }
    return !word.text.empty();
}

bool parseWordsArray(const char*& ptr, const char* end, std::vector<SubtitleWord>& words)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
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
        SubtitleWord word;
        if (!parseWordObject(ptr, end, word))
        {
            skipValue(ptr, end);
            continue;
        }
        words.push_back(std::move(word));
    }
    return true;
}

bool parseSegmentObject(const char*& ptr, const char* end, SubtitleSegment& segment)
{
    if (!expectChar(ptr, end, '{'))
    {
        return false;
    }
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
        std::string key;
        if (!parseJsonString(ptr, end, key))
        {
            return false;
        }
        if (!expectChar(ptr, end, ':'))
        {
            return false;
        }
        if (key == "start")
        {
            parseNumber(ptr, end, segment.start);
        }
        else if (key == "end")
        {
            parseNumber(ptr, end, segment.end);
        }
        else if (key == "text")
        {
            parseJsonString(ptr, end, segment.text);
        }
        else if (key == "words")
        {
            segment.words.clear();
            parseWordsArray(ptr, end, segment.words);
        }
        else
        {
            if (!skipValue(ptr, end))
            {
                return false;
            }
        }
    }
    return true;
}

bool parseSegmentsArray(const char*& ptr, const char* end, std::vector<SubtitleSegment>& segments)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
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
        SubtitleSegment segment;
        if (!parseSegmentObject(ptr, end, segment))
        {
            skipValue(ptr, end);
            continue;
        }
        segments.push_back(std::move(segment));
    }
    return true;
}

bool findSegmentsArray(const std::string& text, const char*& ptr, const char* end)
{
    const char* base = text.data();
    const char* key = std::strstr(base, "\"segments\"");
    if (!key)
    {
        return false;
    }
    ptr = key + std::strlen("\"segments\"");
    skipWhitespace(ptr, end);
    if (!expectChar(ptr, end, ':'))
    {
        return false;
    }
    skipWhitespace(ptr, end);
    if (ptr >= end || *ptr != '[')
    {
        return false;
    }
    return true;
}

std::vector<SubtitleOverlay::Line> buildLines(const std::vector<SubtitleSegment>& segments)
{
    constexpr size_t kMaxWordsPerLine = 7;
    constexpr size_t kMaxCharsPerLine = 64;

    std::vector<SubtitleOverlay::Line> lines;
    lines.reserve(segments.size() * 2);

    for (const auto& segment : segments)
    {
        if (!segment.words.empty())
        {
            size_t index = 0;
            while (index < segment.words.size())
            {
                size_t endIndex = index;
                size_t chars = segment.words[endIndex].text.size();
                double startTime = segment.words[endIndex].start;
                double endTime = segment.words[endIndex].end;
                std::string text = segment.words[endIndex].text;
                ++endIndex;
                size_t words = 1;
                while (endIndex < segment.words.size())
                {
                    const auto& nextWord = segment.words[endIndex];
                    size_t nextLen = nextWord.text.size();
                    if (words >= kMaxWordsPerLine || chars + 1 + nextLen > kMaxCharsPerLine)
                    {
                        break;
                    }
                    text += " ";
                    text += nextWord.text;
                    chars += 1 + nextLen;
                    endTime = nextWord.end;
                    ++endIndex;
                    ++words;
                }
                lines.push_back({startTime, endTime, std::move(text)});
                index = endIndex;
            }
        }
        else if (!segment.text.empty())
        {
            lines.push_back({segment.start, segment.end, segment.text});
        }
    }

    std::sort(lines.begin(), lines.end(), [](const SubtitleOverlay::Line& a, const SubtitleOverlay::Line& b) {
        return a.start < b.start;
    });
    return lines;
}

fonts::FontBitmap prepareLineBitmap(const std::string& text, uint32_t fontSize)
{
    return fonts::renderText(text, fontSize);
}

struct SubtitleLineDescriptor
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

struct SubtitleOverlayCompute
{
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
};

SubtitleOverlayCompute g_subtitleCompute{};

bool initializeSubtitleOverlayCompute(Engine2D* engine, SubtitleOverlayCompute& comp)
{
    if (!engine)
    {
        return false;
    }
    if (comp.pipeline != VK_NULL_HANDLE)
    {
        return true;
    }

    comp.device = engine->logicalDevice;
    comp.queue = engine->graphicsQueue;
    comp.queueFamily = engine->getGraphicsQueueFamilyIndex();

    auto shaderCode = readSPIRVFile("shaders/subtitle_blit.comp.spv");
    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(comp.device, &layoutInfo, nullptr, &comp.descriptorSetLayout) != VK_SUCCESS)
    {
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(SubtitlePushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &comp.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(comp.device, &pipelineLayoutInfo, nullptr, &comp.pipelineLayout) != VK_SUCCESS)
    {
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
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
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(comp.device, shaderModule, nullptr);
        return false;
    }
    vkDestroyShaderModule(comp.device, shaderModule, nullptr);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    if (vkCreateDescriptorPool(comp.device, &poolInfo, nullptr, &comp.descriptorPool) != VK_SUCCESS)
    {
        vkDestroyPipeline(comp.device, comp.pipeline, nullptr);
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = comp.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &comp.descriptorSetLayout;
    if (vkAllocateDescriptorSets(comp.device, &allocInfo, &comp.descriptorSet) != VK_SUCCESS)
    {
        vkDestroyDescriptorPool(comp.device, comp.descriptorPool, nullptr);
        vkDestroyPipeline(comp.device, comp.pipeline, nullptr);
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        return false;
    }

    VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCreateInfo.queueFamilyIndex = comp.queueFamily;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(comp.device, &poolCreateInfo, nullptr, &comp.commandPool) != VK_SUCCESS)
    {
        vkDestroyDescriptorPool(comp.device, comp.descriptorPool, nullptr);
        vkDestroyPipeline(comp.device, comp.pipeline, nullptr);
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = comp.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(comp.device, &cmdAllocInfo, &comp.commandBuffer) != VK_SUCCESS)
    {
        vkDestroyCommandPool(comp.device, comp.commandPool, nullptr);
        vkDestroyDescriptorPool(comp.device, comp.descriptorPool, nullptr);
        vkDestroyPipeline(comp.device, comp.pipeline, nullptr);
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(comp.device, &fenceInfo, nullptr, &comp.fence) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(comp.device, comp.commandPool, 1, &comp.commandBuffer);
        vkDestroyCommandPool(comp.device, comp.commandPool, nullptr);
        vkDestroyDescriptorPool(comp.device, comp.descriptorPool, nullptr);
        vkDestroyPipeline(comp.device, comp.pipeline, nullptr);
        vkDestroyPipelineLayout(comp.device, comp.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(comp.device, comp.descriptorSetLayout, nullptr);
        return false;
    }

    return true;
}

void destroySubtitleOverlayCompute(SubtitleOverlayCompute& comp)
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
    comp.queue = VK_NULL_HANDLE;
    comp.queueFamily = 0;
    comp.device = VK_NULL_HANDLE;
}

bool runSubtitleOverlayCompute(Engine2D* engine,
                               SubtitleOverlayCompute& comp,
                               SubtitleOverlayResources& resources,
                               overlay::ImageResource& target,
                               const SubtitleLineDescriptor* lineDescriptors,
                               uint32_t lineCount,
                               VkSampler glyphSampler,
                               bool enableBackground)
{
    if (!engine || target.view == VK_NULL_HANDLE || target.width == 0 || target.height == 0)
    {
        return false;
    }
    if (!initializeSubtitleOverlayCompute(engine, comp))
    {
        return false;
    }

    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageView = target.view;
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkImageView fallbackView = storageInfo.imageView;
    VkImageView line0View = resources.lines[0].image.view != VK_NULL_HANDLE ? resources.lines[0].image.view : fallbackView;
    VkImageView line1View = resources.lines[1].image.view != VK_NULL_HANDLE ? resources.lines[1].image.view : line0View;

    VkDescriptorImageInfo glyphInfos[2];
    glyphInfos[0].imageView = line0View;
    glyphInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    glyphInfos[0].sampler = glyphSampler;
    glyphInfos[1].imageView = line1View;
    glyphInfos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    glyphInfos[1].sampler = glyphSampler;

    std::array<VkWriteDescriptorSet, 3> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = comp.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &storageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = comp.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &glyphInfos[0];

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = comp.descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &glyphInfos[1];

    vkUpdateDescriptorSets(comp.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(comp.commandBuffer, 0);
    vkBeginCommandBuffer(comp.commandBuffer, &beginInfo);

    VkImageMemoryBarrier preBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    preBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preBarrier.image = target.image;
    preBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    preBarrier.subresourceRange.baseMipLevel = 0;
    preBarrier.subresourceRange.levelCount = 1;
    preBarrier.subresourceRange.baseArrayLayer = 0;
    preBarrier.subresourceRange.layerCount = 1;
    preBarrier.oldLayout = target.layout;
    preBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    preBarrier.srcAccessMask = (target.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                    ? VK_ACCESS_SHADER_READ_BIT
                                    : 0;
    preBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(comp.commandBuffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &preBarrier);

    vkCmdBindPipeline(comp.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, comp.pipeline);
    vkCmdBindDescriptorSets(comp.commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            comp.pipelineLayout,
                            0,
                            1,
                            &comp.descriptorSet,
                            0,
                            nullptr);

    SubtitlePushConstants push{};
    push.backgroundColor = enableBackground ? glm::vec4(0.0f, 0.0f, 0.0f, 0.55f) : glm::vec4(0.0f);
    push.textColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    push.imageSize = glm::ivec2(target.width, target.height);
    push.lineCount = static_cast<int32_t>(lineCount);
    for (uint32_t i = 0; i < 2; ++i)
    {
        push.lineOrigin[i] = lineDescriptors[i].origin;
        push.lineSize[i] = lineDescriptors[i].size;
    }

    vkCmdPushConstants(comp.commandBuffer,
                       comp.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(SubtitlePushConstants),
                       &push);

    const uint32_t groupX = (target.width + 15) / 16;
    const uint32_t groupY = (target.height + 15) / 16;
    vkCmdDispatch(comp.commandBuffer, groupX, groupY, 1);

    VkImageMemoryBarrier postBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    postBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    postBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    postBarrier.image = target.image;
    postBarrier.subresourceRange = preBarrier.subresourceRange;
    postBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    postBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(comp.commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &postBarrier);

    vkEndCommandBuffer(comp.commandBuffer);

    vkResetFences(comp.device, 1, &comp.fence);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &comp.commandBuffer;
    if (vkQueueSubmit(comp.queue, 1, &submitInfo, comp.fence) != VK_SUCCESS)
    {
        return false;
    }
    vkWaitForFences(comp.device, 1, &comp.fence, VK_TRUE, UINT64_MAX);

    target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

} // namespace
// SubtitleOverlay

bool SubtitleOverlay::load(const std::filesystem::path& path)
{
    if (path.empty() || !std::filesystem::exists(path))
    {
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }
    std::string contents;
    contents.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (contents.empty())
    {
        return false;
    }

    const char* ptr = contents.data();
    const char* end = ptr + contents.size();
    skipWhitespace(ptr, end);
    if (!findSegmentsArray(contents, ptr, end))
    {
        return false;
    }

    std::vector<SubtitleSegment> segments;
    if (!parseSegmentsArray(ptr, end, segments))
    {
        return false;
    }

    lines_ = buildLines(segments);
    lastIndex_ = 0;
    return !lines_.empty();
}

std::vector<const SubtitleOverlay::Line*> SubtitleOverlay::activeLines(double currentTime, size_t maxLines) const
{
    std::vector<const Line*> result;
    if (lines_.empty() || maxLines == 0)
    {
        lastIndex_ = 0;
        return result;
    }

    const double lookbehind = 0.3;
    const double lookahead = 0.5;
    size_t index = lastIndex_;
    if (index >= lines_.size())
    {
        index = lines_.size() ? lines_.size() - 1 : 0;
    }
    while (index > 0 && lines_[index].start > currentTime)
    {
        --index;
    }
    while (index < lines_.size() && lines_[index].end + lookbehind < currentTime)
    {
        ++index;
    }

    lastIndex_ = index;
    for (size_t i = index; i < lines_.size(); ++i)
    {
        const Line& line = lines_[i];
        if (line.start - lookahead > currentTime)
        {
            if (result.empty())
            {
                result.push_back(&line);
            }
            break;
        }
        if (line.end + lookbehind >= currentTime)
        {
            result.push_back(&line);
            if (result.size() >= maxLines)
            {
                break;
            }
        }
    }

    if (result.empty() && index < lines_.size())
    {
        result.push_back(&lines_[index]);
    }
    if (result.size() > maxLines)
    {
        result.resize(maxLines);
    }
    return result;
}

// Subtitle overlay resources

bool updateSubtitleOverlay(Engine2D* engine,
                           SubtitleOverlayResources& resources,
                           const SubtitleOverlay& overlay,
                           double currentTime,
                           uint32_t fbWidth,
                           uint32_t fbHeight,
                           glm::vec2 overlayCenter,
                           glm::vec2 overlaySize,
                           overlay::ImageResource& overlayTarget,
                           VkSampler overlaySampler,
                           VkSampler fallbackSampler,
                           size_t maxLines,
                           bool enableBackground)
{
    resources.active = false;
    if (!engine || fbWidth == 0 || fbHeight == 0 || overlaySize.x <= 0.0f || overlaySize.y <= 0.0f ||
        overlayTarget.view == VK_NULL_HANDLE || overlayTarget.width == 0 || overlayTarget.height == 0 ||
        !overlay.hasData())
    {
        std::cout << "[Subtitle] updateSubtitleOverlay: invalid parameters" << std::endl;
        return false;
    }

    const std::vector<const SubtitleOverlay::Line*> candidateLines = overlay.activeLines(currentTime, maxLines);
    if (candidateLines.empty())
    {
        std::cout << "[Subtitle] No active lines at time " << currentTime << std::endl;
        return false;
    }

    std::vector<const SubtitleOverlay::Line*> lines;
    lines.reserve(candidateLines.size());
    std::string lastText;
    for (const SubtitleOverlay::Line* line : candidateLines)
    {
        if (!line)
        {
            continue;
        }
        std::string text = line->text;
        if (text.empty() || text == lastText)
        {
            continue;
        }
        lastText = text;
        lines.push_back(line);
        if (lines.size() >= maxLines)
        {
            break;
        }
    }

    if (lines.empty())
    {
        std::cout << "[Subtitle] No valid lines after filtering" << std::endl;
        return false;
    }

    const uint32_t overlayWidth = static_cast<uint32_t>(
        std::clamp(std::lround(overlaySize.x), 1l, static_cast<long>(overlayTarget.width)));
    const uint32_t overlayHeight = static_cast<uint32_t>(
        std::clamp(std::lround(overlaySize.y), 1l, static_cast<long>(overlayTarget.height)));
    if (overlayWidth == 0 || overlayHeight == 0)
    {
        std::cout << "[Subtitle] Invalid overlay dimensions: " << overlayWidth << "x" << overlayHeight << std::endl;
        return false;
    }

    const int32_t maxLeft = std::max<int32_t>(0, static_cast<int32_t>(overlayTarget.width) - static_cast<int32_t>(overlayWidth));
    const int32_t maxTop = std::max<int32_t>(0, static_cast<int32_t>(overlayTarget.height) - static_cast<int32_t>(overlayHeight));
    const int32_t overlayLeft =
        static_cast<int32_t>(std::clamp(overlayCenter.x - overlayWidth * 0.5f, 0.0f, static_cast<float>(maxLeft)));
    const int32_t overlayTop =
        static_cast<int32_t>(std::clamp(overlayCenter.y - overlayHeight * 0.5f, 0.0f, static_cast<float>(maxTop)));

    std::cout << "[Subtitle] Overlay bounds: " << overlayWidth << "x" << overlayHeight
              << " at (" << overlayLeft << "," << overlayTop << ")" << std::endl;

    const uint32_t padding = std::max<uint32_t>(8, overlayHeight / 20);
    const uint32_t innerWidth = overlayWidth > padding * 2 ? overlayWidth - padding * 2 : 0;
    const uint32_t innerHeight = overlayHeight > padding * 2 ? overlayHeight - padding * 2 : 0;
    if (innerWidth == 0 || innerHeight == 0)
    {
        return false;
    }

    uint32_t fontSize = std::max<uint32_t>(24, innerHeight / 4);
    fontSize = std::min(fontSize, innerHeight);

    std::vector<fonts::FontBitmap> bitmaps;
    bitmaps.reserve(lines.size());
    uint32_t finalLineSpacing = std::max<uint32_t>(4, fontSize / 4);
    uint32_t maxLineWidth = 0;
    uint32_t combinedHeight = 0;

    const uint32_t minFontSize = 16;
    while (true)
    {
        bitmaps.clear();
        maxLineWidth = 0;
        combinedHeight = 0;
        for (const SubtitleOverlay::Line* line : lines)
        {
            fonts::FontBitmap bmp = prepareLineBitmap(line->text, fontSize);
            if (bmp.width == 0 || bmp.height == 0 || bmp.pixels.empty())
            {
                continue;
            }
            maxLineWidth = std::max(maxLineWidth, bmp.width);
            combinedHeight += bmp.height;
            bitmaps.push_back(std::move(bmp));
        }

        if (bitmaps.empty())
        {
            std::cout << "[Subtitle] No bitmaps generated" << std::endl;
            return false;
        }

        combinedHeight += (bitmaps.size() > 1 ? (static_cast<uint32_t>(bitmaps.size()) - 1) * finalLineSpacing : 0);

        bool fits = combinedHeight <= innerHeight;
        if (fits && maxLineWidth <= innerWidth * 1.2f)
        {
            break;
        }

        if (fontSize <= minFontSize)
        {
            break;
        }

        --fontSize;
        finalLineSpacing = std::max<uint32_t>(4, fontSize / 4);
    }

    const uint32_t textWidth = std::min(maxLineWidth, innerWidth);
    const uint32_t textHeight = std::min(combinedHeight, innerHeight);
    if (textWidth == 0 || textHeight == 0)
    {
        std::cout << "[Subtitle] Invalid text dimensions: " << textWidth << "x" << textHeight << std::endl;
        return false;
    }
    
    std::cout << "[Subtitle] Text dimensions: " << textWidth << "x" << textHeight 
              << " (inner: " << innerWidth << "x" << innerHeight << ")" << std::endl;

    SubtitleLineDescriptor lineDescriptors[2]{};
    uint32_t yOffset = padding;
    const size_t preparedLines = std::min(bitmaps.size(), static_cast<size_t>(2));
    for (size_t idx = 0; idx < preparedLines; ++idx)
    {
        const auto& bmp = bitmaps[idx];
        if (bmp.pixels.empty())
        {
            continue;
        }

        uint32_t destX = overlayLeft + padding;
        if (bmp.width < innerWidth)
        {
            destX += (innerWidth - bmp.width) / 2;
        }

        if (!overlay::uploadImageData(engine,
                                      resources.lines[idx].image,
                                      bmp.pixels.data(),
                                      bmp.pixels.size(),
                                      bmp.width,
                                      bmp.height,
                                      VK_FORMAT_R8G8B8A8_UNORM))
        {
            std::cout << "[Subtitle] Failed to upload image data for line " << idx << std::endl;
            return false;
        }

        std::cout << "[Subtitle] Uploaded line " << idx << ": " << bmp.width << "x" << bmp.height 
                  << " (" << bmp.pixels.size() << " bytes)" << std::endl;
        resources.lines[idx].width = bmp.width;
        resources.lines[idx].height = bmp.height;

        lineDescriptors[idx].origin = glm::ivec2(static_cast<int32_t>(destX), static_cast<int32_t>(overlayTop + yOffset));
        lineDescriptors[idx].size = glm::ivec2(static_cast<int32_t>(bmp.width),
                                               static_cast<int32_t>(bmp.height));

        yOffset += bmp.height;
        if (idx + 1 < bitmaps.size())
        {
            yOffset += finalLineSpacing;
        }
    }

    VkSampler glyphSampler = overlaySampler != VK_NULL_HANDLE ? overlaySampler : fallbackSampler;
    std::cout << "[Subtitle] Running compute with " << preparedLines << " lines, sampler: " 
              << (glyphSampler != VK_NULL_HANDLE ? "valid" : "null") << std::endl;
    if (!runSubtitleOverlayCompute(engine,
                                   g_subtitleCompute,
                                   resources,
                                   overlayTarget,
                                   lineDescriptors,
                                   static_cast<uint32_t>(preparedLines),
                                   glyphSampler,
                                   enableBackground))
    {
        std::cout << "[Subtitle] Compute shader failed" << std::endl;
        return false;
    }

    std::cout << "[Subtitle] Compute shader succeeded" << std::endl;
    resources.active = preparedLines > 0;
    return resources.active;
}

void destroySubtitleOverlayResources(Engine2D* engine, SubtitleOverlayResources& resources)
{
    if (engine)
    {
        for (auto& line : resources.lines)
        {
            overlay::destroyImageResource(engine, line.image);
            line.width = 0;
            line.height = 0;
        }
        destroySubtitleOverlayCompute(g_subtitleCompute);
    }
    resources.active = false;
}
