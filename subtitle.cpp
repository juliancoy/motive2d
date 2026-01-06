#include "debug_logging.h"
#include "text.h"

#include "engine2d.h"
#include "text.h"
#include "subtitle.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"
#include <glm/vec4.hpp>

// Parse JSON subtitle file using nlohmann/json
static std::vector<SubtitleSegment> parseJSON(const std::string& jsonContent)
{
    using json = nlohmann::json;
    std::vector<SubtitleSegment> segments;
    
    try {
        auto j = json::parse(jsonContent);
        
        if (!j.contains("segments") || !j["segments"].is_array()) {
            LOG_DEBUG(std::cout << "[Subtitle] JSON missing 'segments' array" << std::endl);
            return segments;
        }
        
        for (const auto& segJson : j["segments"]) {
            SubtitleSegment segment;
            
            if (segJson.contains("start") && segJson["start"].is_number()) {
                segment.start = segJson["start"].get<double>();
            }
            if (segJson.contains("end") && segJson["end"].is_number()) {
                segment.end = segJson["end"].get<double>();
            }
            if (segJson.contains("text") && segJson["text"].is_string()) {
                segment.text = segJson["text"].get<std::string>();
            }
            
            if (segJson.contains("words") && segJson["words"].is_array()) {
                for (const auto& wordJson : segJson["words"]) {
                    SubtitleWord word;
                    
                    // Try "word" field first, fall back to "text"
                    if (wordJson.contains("word") && wordJson["word"].is_string()) {
                        word.text = wordJson["word"].get<std::string>();
                    } else if (wordJson.contains("text") && wordJson["text"].is_string()) {
                        word.text = wordJson["text"].get<std::string>();
                    }
                    
                    if (wordJson.contains("start") && wordJson["start"].is_number()) {
                        word.start = wordJson["start"].get<double>();
                    }
                    if (wordJson.contains("end") && wordJson["end"].is_number()) {
                        word.end = wordJson["end"].get<double>();
                    }
                    
                    if (!word.text.empty()) {
                        segment.words.push_back(word);
                    }
                }
            }
            
            segments.push_back(segment);
        }
    } catch (const json::parse_error& e) {
        LOG_DEBUG(std::cout << "[Subtitle] JSON parse error: " << e.what() << std::endl);
    } catch (const json::exception& e) {
        LOG_DEBUG(std::cout << "[Subtitle] JSON exception: " << e.what() << std::endl);
    }
    
    return segments;
}

std::vector<Line> buildLines(const std::vector<SubtitleSegment>& segments)
{
    constexpr size_t kMaxWordsPerLine = 7;
    constexpr size_t kMaxCharsPerLine = 64;

    std::vector<Line> lines;
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

    std::sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) {
        return a.start < b.start;
    });
    return lines;
}


Subtitle::Subtitle(const std::filesystem::path &path, Engine2D* engine)
{

    device = engine->logicalDevice;
    queue = engine->graphicsQueue;
    queueFamily = engine->getGraphicsQueueFamilyIndex();

    auto shaderCode = readSPIRVFile("shaders/subtitle.spv");
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
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(SubtitlePushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }
    vkDestroyShaderModule(device, shaderModule, nullptr);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCreateInfo.queueFamilyIndex = queueFamily;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer) != VK_SUCCESS)
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

}


bool Subtitle::load(const std::filesystem::path& path)
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
    const auto segments = parseJSON(contents);

    lines_ = buildLines(segments);
    lastIndex_ = 0;
    return !lines_.empty();
}

std::vector<const Line*> Subtitle::activeLines(double currentTime, size_t maxLines) const
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
};

// Subtitle overlay resources

Subtitle::~Subtitle()
{

    if (fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(device, fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
    if (commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    queue = VK_NULL_HANDLE;
    queueFamily = 0;
    device = VK_NULL_HANDLE;
};

bool Subtitle::run(Engine2D *engine,
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
             size_t maxLines,
             bool enableBackground)
{
    // Validate parameters
    if (!engine || fbWidth == 0 || fbHeight == 0 || overlaySize.x <= 0.0f || overlaySize.y <= 0.0f ||
        overlayTarget.view == VK_NULL_HANDLE || overlayTarget.width == 0 || overlayTarget.height == 0 ||
        !overlay.hasData())
    {
        LOG_DEBUG(std::cout << "[Subtitle] run: invalid parameters" << std::endl);
        return false;
    }

    const std::vector<const Line*> candidateLines = overlay.activeLines(currentTime, maxLines);
    if (candidateLines.empty())
    {
        LOG_DEBUG(std::cout << "[Subtitle] No active lines at time " << currentTime << std::endl);
        return false;
    }

    std::vector<const Line*> lines;
    lines.reserve(candidateLines.size());
    std::string lastText;
    for (const Line* line : candidateLines)
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
        LOG_DEBUG(std::cout << "[Subtitle] No valid lines after filtering" << std::endl);
        return false;
    }

    const uint32_t overlayWidth = static_cast<uint32_t>(
        std::clamp(std::lround(overlaySize.x), 1l, static_cast<long>(overlayTarget.width)));
    const uint32_t overlayHeight = static_cast<uint32_t>(
        std::clamp(std::lround(overlaySize.y), 1l, static_cast<long>(overlayTarget.height)));
    if (overlayWidth == 0 || overlayHeight == 0)
    {
        LOG_DEBUG(std::cout << "[Subtitle] Invalid overlay dimensions: " << overlayWidth << "x" << overlayHeight << std::endl);
        return false;
    }

    const int32_t maxLeft = std::max<int32_t>(0, static_cast<int32_t>(overlayTarget.width) - static_cast<int32_t>(overlayWidth));
    const int32_t maxTop = std::max<int32_t>(0, static_cast<int32_t>(overlayTarget.height) - static_cast<int32_t>(overlayHeight));
    const int32_t overlayLeft =
        static_cast<int32_t>(std::clamp(overlayCenter.x - overlayWidth * 0.5f, 0.0f, static_cast<float>(maxLeft)));
    const int32_t overlayTop =
        static_cast<int32_t>(std::clamp(overlayCenter.y - overlayHeight * 0.5f, 0.0f, static_cast<float>(maxTop)));

    LOG_DEBUG(std::cout << "[Subtitle] Overlay bounds: " << overlayWidth << "x" << overlayHeight
              << " at (" << overlayLeft << "," << overlayTop << ")" << std::endl);

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
        for (const Line* line : lines)
        {
            fonts::FontBitmap bmp = fonts::renderText(line->text, fontSize);
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
            LOG_DEBUG(std::cout << "[Subtitle] No bitmaps generated" << std::endl);
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
        LOG_DEBUG(std::cout << "[Subtitle] Invalid text dimensions: " << textWidth << "x" << textHeight << std::endl);
        return false;
    }
    
    LOG_DEBUG(std::cout << "[Subtitle] Text dimensions: " << textWidth << "x" << textHeight 
              << " (inner: " << innerWidth << "x" << innerHeight << ")" << std::endl);

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

        // Upload bitmap to image resource
        resources.lines[idx].image.uploadImageData(
                             bmp.pixels.data(),
                             bmp.pixels.size(),
                             bmp.width,
                             bmp.height,
                             VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        LOG_DEBUG(std::cout << "[Subtitle] Uploaded line " << idx << ": " << bmp.width << "x" << bmp.height 
                  << " (" << bmp.pixels.size() << " bytes)" << std::endl);
        resources.lines[idx].width = bmp.width;
        resources.lines[idx].height = bmp.height;

        // Store line descriptor (optional, if needed elsewhere)
        // lineDescriptors[idx].origin = glm::ivec2(static_cast<int32_t>(destX), static_cast<int32_t>(overlayTop + yOffset));
        // lineDescriptors[idx].size = glm::ivec2(static_cast<int32_t>(bmp.width), static_cast<int32_t>(bmp.height));

        yOffset += bmp.height;
        if (idx + 1 < bitmaps.size())
        {
            yOffset += finalLineSpacing;
        }
    }

    resources.active = preparedLines > 0;
    return resources.active;
}
