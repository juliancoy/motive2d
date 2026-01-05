#include "debug_logging.h"
#include "text.h"

#include "engine2d.h"
#include "text.h"
#include "subtitle.h"

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

std::vector<std::string> buildLines(const std::vector<SubtitleSegment>& segments)
{
    constexpr size_t kMaxWordsPerLine = 7;
    constexpr size_t kMaxCharsPerLine = 64;

    std::vector<std::string> lines;
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

    std::sort(lines.begin(), lines.end(), [](const std::string& a, const std::string& b) {
        return a.start < b.start;
    });
    return lines;
}


Subtitle::Subtitle(Engine2D* engine, )
{
    if (!engine)
    {
        return false;
    }
    if (pipeline != VK_NULL_HANDLE)
    {
        return true;
    }

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
        return false;
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
        return false;
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
        return false;
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
        return false;
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
        return false;
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
        return false;
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
        return false;
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
        return false;
    }

    return true;
}


bool Subtitle::Run(
                const std::stringDescriptor* lineDescriptors,
                uint32_t lineCount,
                VkSampler glyphSampler,
                bool enableBackground)
{
    if (!engine || target.view == VK_NULL_HANDLE || target.width == 0 || target.height == 0)
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
    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &storageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &glyphInfos[0];

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &glyphInfos[1];

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

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
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &preBarrier);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0,
                            1,
                            &descriptorSet,
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

    vkCmdPushConstants(commandBuffer,
                       pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(SubtitlePushConstants),
                       &push);

    const uint32_t groupX = (target.width + 15) / 16;
    const uint32_t groupY = (target.height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupX, groupY, 1);

    VkImageMemoryBarrier postBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    postBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    postBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    postBarrier.image = target.image;
    postBarrier.subresourceRange = preBarrier.subresourceRange;
    postBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    postBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &postBarrier);

    vkEndCommandBuffer(commandBuffer);

    vkResetFences(device, 1, &fence);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS)
    {
        return false;
    }
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
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

std::vector<const std::string*> Subtitle::activeLines(double currentTime, size_t maxLines) const
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

bool Subtitle::updateOverlay(
                           uint32_t fbWidth,
                           uint32_t fbHeight,
                           glm::vec2 overlayCenter,
                           glm::vec2 overlaySize,
                           ImageResource& overlayTarget,
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
        LOG_DEBUG(std::cout << "[Subtitle] updateSubtitleOverlay: invalid parameters" << std::endl);
        return false;
    }

    const std::vector<const Line*> candidateLines = activeLines(currentTime, maxLines);
    if (candidateLines.empty())
    {
        LOG_DEBUG(std::cout << "[Subtitle] No active lines at time " << currentTime << std::endl);
        return false;
    }

    std::vector<const std::string*> lines;
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
        for (const std::string* line : lines)
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

    std::string lineDescriptors[2]{};
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

        if (!uploadImageData(engine,
                                      resources.lines[idx].image,
                                      bmp.pixels.data(),
                                      bmp.pixels.size(),
                                      bmp.width,
                                      bmp.height,
                                      VK_FORMAT_R8G8B8A8_UNORM))
        {
        LOG_DEBUG(std::cout << "[Subtitle] Failed to upload image data for line " << idx << std::endl);
            return false;
        }

        LOG_DEBUG(std::cout << "[Subtitle] Uploaded line " << idx << ": " << bmp.width << "x" << bmp.height 
                  << " (" << bmp.pixels.size() << " bytes)" << std::endl);
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
    LOG_DEBUG(std::cout << "[Subtitle] Running compute with " << preparedLines << " lines, sampler: " 
              << (glyphSampler != VK_NULL_HANDLE ? "valid" : "null") << std::endl);
    if (!run(engine,
                                   g_subtitleCompute,
                                   resources,
                                   overlayTarget,
                                   lineDescriptors,
                                   static_cast<uint32_t>(preparedLines),
                                   glyphSampler,
                                   enableBackground))
    {
        LOG_DEBUG(std::cout << "[Subtitle] Compute shader failed" << std::endl);
        return false;
    }

    LOG_DEBUG(std::cout << "[Subtitle] Compute shader succeeded" << std::endl);
    resources.active = preparedLines > 0;
    return resources.active;
}

Subtitle::~Subtitle::()
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

    if (engine)
    {
        for (auto& line : resources.lines)
        {
            destroyImageResource(engine, line.image);
            line.width = 0;
            line.height = 0;
        }
        destroy(g_subtitleCompute);
    }
    resources.active = false;
}
