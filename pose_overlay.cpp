#include "pose_overlay.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "debug_logging.h"
#include "fps.h"
#include <nlohmann/json.hpp>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "engine2d.h"
#include "utils.h"

constexpr std::array<glm::vec4, 4> kLabelPalette = {
    glm::vec4(0.99f, 0.49f, 0.18f, 1.0f),
    glm::vec4(0.36f, 0.72f, 0.84f, 1.0f),
    glm::vec4(0.42f, 0.88f, 0.46f, 1.0f),
    glm::vec4(0.94f, 0.73f, 0.25f, 1.0f),
};

bool PoseOverlay::extractCoordsArray(const nlohmann::json& jsonArray, std::vector<float>& out)
{
    if (!jsonArray.is_array())
    {
        return false;
    }
    out.clear();
    out.reserve(jsonArray.size());
    for (const auto& value : jsonArray)
    {
        if (!value.is_number())
        {
            return false;
        }
        out.push_back(static_cast<float>(value.get<double>()));
    }
    return true;
}

bool PoseOverlay::parsePoseEntry(const nlohmann::json& entry, int& frameIndex, std::vector<float>& coords)
{
    std::vector<float> parsedCoords;
    int parsedFrame = -1;

    if (entry.is_array())
    {
        if (entry.size() != 2 || !entry[0].is_number())
        {
            return false;
        }
        if (!extractCoordsArray(entry[1], parsedCoords))
        {
            return false;
        }
        parsedFrame = static_cast<int>(entry[0].get<double>());
    }
    else if (entry.is_object())
    {
        bool frameSet = false;
        bool poseSet = false;
        for (const auto& item : entry.items())
        {
            const auto& key = item.key();
            const auto& value = item.value();
            if (key == "frame")
            {
                if (!value.is_number())
                {
                    return false;
                }
                parsedFrame = static_cast<int>(value.get<double>());
                frameSet = true;
            }
            else if (key == "pose" || key == "coords")
            {
                if (!extractCoordsArray(value, parsedCoords))
                {
                    return false;
                }
                poseSet = true;
            }
            else
            {
                return false;
            }
        }
        if (!frameSet || !poseSet)
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    frameIndex = parsedFrame;
    coords = std::move(parsedCoords);
    return true;
}

bool PoseOverlay::parsePoseJsonLine(const std::string& line, int& frameIndex, std::vector<float>& coords)
{
    try
    {
        const auto document = nlohmann::json::parse(line);
        return parsePoseEntry(document, frameIndex, coords);
    }
    catch (const nlohmann::json::exception&)
    {
        return false;
    }
}

std::filesystem::path PoseOverlay::poseCoordsPath(const std::filesystem::path& videoPath)
{
    if (videoPath.empty())
    {
        return {};
    }
    const std::string stem = videoPath.stem().string();
    if (stem.empty())
    {
        return {};
    }
    std::filesystem::path jsonPath = videoPath.parent_path() / (stem + "_pose_coords.json");
    if (std::filesystem::exists(jsonPath))
    {
        return jsonPath;
    }
    std::filesystem::path txtPath = videoPath.parent_path() / (stem + "_pose_coords.txt");
    if (std::filesystem::exists(txtPath))
    {
        return txtPath;
    }
    return {};
}

bool PoseOverlay::loadCoordsFile(const std::filesystem::path& coordsPath)
{
    if (coordsPath.empty() || !std::filesystem::exists(coordsPath))
    {
        return false;
    }

    std::ifstream file(coordsPath, std::ios::binary);
    if (!file)
    {
        return false;
    }

    frameData_.clear();
    detectionData_.clear();
    labelColors_.clear();

    const std::string extension = coordsPath.extension().string();
    bool parsed = false;
    if (extension == ".txt")
    {
        parsed = parseTxt(file);
    }
    else
    {
        std::string contents;
        contents.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (contents.empty())
        {
            return false;
        }
        parsed = parseJson(contents);
    }
    valid_ = parsed && (!frameData_.empty() || !detectionData_.empty());
    if (!valid_)
    {
        std::cout << "[pose] Failed to parse coords file: " << coordsPath << "\n";
    }
    return valid_;
}

bool PoseOverlay::parseJson(const std::string& text)
{
    frameData_.clear();
    try
    {
        const auto document = nlohmann::json::parse(text);
        if (!document.is_array())
        {
            return false;
        }

        std::vector<float> coords;
        for (const auto& entry : document)
        {
            int frame = -1;
            if (!parsePoseEntry(entry, frame, coords))
            {
                return false;
            }
            if (frame >= 0)
            {
                storeFrame(frame, coords);
            }
        }
        return true;
    }
    catch (const nlohmann::json::exception&)
    {
        return false;
    }
}

void PoseOverlay::storeFrame(int frame, const std::vector<float>& coords)
{
    constexpr size_t expectedSize = 5 + 3 * kKeypointCount;
    if (coords.size() < expectedSize)
    {
        return;
    }

    FramePose pose{};
    pose.valid.fill(false);
    pose.validCount = 0;

    for (size_t idx = 0; idx < kKeypointCount; ++idx)
    {
        const size_t baseIndex = 5 + idx * 3;
        if (baseIndex + 1 >= coords.size())
        {
            break;
        }
        const float x = coords[baseIndex];
        const float y = coords[baseIndex + 1];
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            continue;
        }
        if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f)
        {
            continue;
        }
        pose.keypoints[idx] = glm::vec2(x, y);
        pose.valid[idx] = true;
        ++pose.validCount;
    }

    if (pose.validCount == 0)
    {
        return;
    }

    static const std::array<glm::vec4, 3> instanceColors = {
        glm::vec4(0.99f, 0.49f, 0.18f, 1.0f),
        glm::vec4(0.36f, 0.72f, 0.84f, 1.0f),
        glm::vec4(0.42f, 0.88f, 0.46f, 1.0f),
    };
    auto& list = frameData_[frame];
    const size_t colorIndex = list.size() % instanceColors.size();
    pose.color = instanceColors[colorIndex];
    list.push_back(pose);
}

glm::vec4 PoseOverlay::colorForLabel(const std::string& label)
{
    if (label.empty())
    {
        return kLabelPalette[0];
    }
    auto it = labelColors_.find(label);
    if (it != labelColors_.end())
    {
        return it->second;
    }
    const size_t index = labelColors_.size() % kLabelPalette.size();
    const glm::vec4 color = kLabelPalette[index];
    labelColors_.emplace(label, color);
    return color;
}

void PoseOverlay::run(ImageResource& target,
                     uint32_t width,
                     uint32_t height,
                     const glm::vec2& rectCenter,
                     const glm::vec2& rectSize,
                     float outerThickness,
                     float innerThickness,
                     float detectionEnabled,
                     const DetectionEntry* detections,
                     uint32_t detectionCount)
{
    LOG_DEBUG(std::cout << "[PoseOverlay] Starting pose overlay compute on image: " << target.image 
              << " (view: " << target.view << ")" << std::endl);
    LOG_DEBUG(std::cout << "[PoseOverlay] Target dimensions: " << width << "x" << height << std::endl);
    LOG_DEBUG(std::cout << "[PoseOverlay] Detection count: " << detectionCount 
              << ", detection enabled: " << detectionEnabled << std::endl);
    
    if (!engine_)
    {
        LOG_DEBUG(std::cout << "[PoseOverlay] Engine not initialized" << std::endl);
        return;
    }
    
    bool recreated = false;
    if (!target.ensure(width, height, VK_FORMAT_R8G8B8A8_UNORM, recreated,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    {
        LOG_DEBUG(std::cout << "[PoseOverlay] Failed to ensure image resource" << std::endl);
        return;
    }

    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageInfo.imageView = target.view;

    const VkDeviceSize entrySize = sizeof(DetectionEntry);
    VkDeviceSize desiredBufferSize = entrySize;
    if (detectionCount > 0)
    {
        desiredBufferSize = entrySize * static_cast<VkDeviceSize>(detectionCount);
    }
    if (!ensureDetectionBuffer(desiredBufferSize))
    {
        return;
    }

    VkDeviceSize copySize = std::max(entrySize, desiredBufferSize);
    if (detectionBuffer != VK_NULL_HANDLE && detectionBufferMemory != VK_NULL_HANDLE)
    {
        void* mapped = nullptr;
        vkMapMemory(device, detectionBufferMemory, 0, copySize, 0, &mapped);
        if (mapped)
        {
            if (detectionCount > 0 && detections)
            {
                std::memcpy(mapped, detections, entrySize * detectionCount);
            }
            else
            {
                std::memset(mapped, 0, static_cast<size_t>(entrySize));
            }
            vkUnmapMemory(device, detectionBufferMemory);
        }
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = detectionBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = copySize;

    VkWriteDescriptorSet imageWrite{};
    imageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    imageWrite.dstSet = descriptorSet;
    imageWrite.dstBinding = 0;
    imageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageWrite.descriptorCount = 1;
    imageWrite.pImageInfo = &storageInfo;

    VkWriteDescriptorSet bufferWrite{};
    bufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    bufferWrite.dstSet = descriptorSet;
    bufferWrite.dstBinding = 1;
    bufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bufferWrite.descriptorCount = 1;
    bufferWrite.pBufferInfo = &bufferInfo;

    std::array<VkWriteDescriptorSet, 2> writes = {imageWrite, bufferWrite};
    vkUpdateDescriptorSets(device,
                           static_cast<uint32_t>(writes.size()),
                           writes.data(),
                           0,
                           nullptr);

    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    PoseOverlayPush push{glm::vec2(static_cast<float>(width), static_cast<float>(height)),
                         rectCenter,
                         rectSize,
                         outerThickness,
                         innerThickness,
                         detectionEnabled,
                         detectionCount};

    const VkImageLayout initialLayout = recreated ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageMemoryBarrier toGeneralBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toGeneralBarrier.oldLayout = initialLayout;
    toGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.image = target.image;
    toGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneralBarrier.subresourceRange.baseMipLevel = 0;
    toGeneralBarrier.subresourceRange.levelCount = 1;
    toGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    toGeneralBarrier.subresourceRange.layerCount = 1;
    toGeneralBarrier.srcAccessMask = (initialLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                         ? VK_ACCESS_SHADER_READ_BIT
                                         : 0;
    toGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    VkPipelineStageFlags srcStage = (initialLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                        ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                        : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    vkCmdPipelineBarrier(commandBuffer,
                         srcStage,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toGeneralBarrier);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PoseOverlayPush),
                       &push);

    const uint32_t groupX = (width + 15) / 16;
    const uint32_t groupY = (height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupX, groupY, 1);

    VkImageMemoryBarrier toReadBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toReadBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadBarrier.image = target.image;
    toReadBarrier.subresourceRange = toGeneralBarrier.subresourceRange;
    toReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toReadBarrier);

    vkEndCommandBuffer(commandBuffer);

    vkResetFences(device, 1, &fence);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
}

PoseOverlay::PoseOverlay(Engine2D* engine) : engine_(engine)
{
    if (!engine)
    {
        std::cerr << "[PoseOverlay] Engine is null" << std::endl;
        return;
    }
    
    device = engine->logicalDevice;
    queue = engine->graphicsQueue;

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        std::cerr << "[PoseOverlay] Failed to create descriptor set layout" << std::endl;
        return;
    }

    std::vector<char> shaderCode;
    try
    {
        shaderCode = readSPIRVFile("shaders/overlay_pose.spv");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[PoseOverlay] " << ex.what() << std::endl;
        return;
    }

    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PoseOverlayPush);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        std::cerr << "[PoseOverlay] Failed to create pipeline layout" << std::endl;
        vkDestroyShaderModule(device, shaderModule, nullptr);
        return;
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
        std::cerr << "[PoseOverlay] Failed to create compute pipeline" << std::endl;
        vkDestroyShaderModule(device, shaderModule, nullptr);
        return;
    }

    vkDestroyShaderModule(device, shaderModule, nullptr);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        std::cerr << "[PoseOverlay] Failed to create descriptor pool" << std::endl;
        return;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
    {
        std::cerr << "[PoseOverlay] Failed to allocate descriptor set" << std::endl;
        return;
    }

    VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCreateInfo.queueFamilyIndex = engine->graphicsQueueFamilyIndex;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        std::cerr << "[PoseOverlay] Failed to create command pool" << std::endl;
        return;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer) != VK_SUCCESS)
    {
        std::cerr << "[PoseOverlay] Failed to allocate command buffer" << std::endl;
        return;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
    {
        std::cerr << "[PoseOverlay] Failed to create fence" << std::endl;
        return;
    }
}

PoseOverlay::~PoseOverlay()
{
    if (device == VK_NULL_HANDLE)
        return;
        
    vkDeviceWaitIdle(device);

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
    if (detectionBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, detectionBuffer, nullptr);
        detectionBuffer = VK_NULL_HANDLE;
    }
    if (detectionBufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, detectionBufferMemory, nullptr);
        detectionBufferMemory = VK_NULL_HANDLE;
    }
    detectionBufferSize = 0;
}

bool PoseOverlay::parseTxt(std::istream& input) {
    // Simple TXT format: each line contains frame index followed by coordinates
    std::string line;
    int lineNum = 0;
    while (std::getline(input, line)) {
        ++lineNum;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int frame;
        if (!(iss >> frame)) continue;
        std::vector<float> coords;
        float val;
        while (iss >> val) {
            coords.push_back(val);
        }
        if (coords.size() >= 5 + 3 * kKeypointCount) {
            storeFrame(frame, coords);
        }
    }
    return !frameData_.empty();
}

bool PoseOverlay::ensureDetectionBuffer(VkDeviceSize requiredSize) {
    if (requiredSize == 0) requiredSize = sizeof(DetectionEntry);
    if (detectionBufferSize >= requiredSize) return true;
    
    // Destroy old buffer if exists
    if (detectionBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, detectionBuffer, nullptr);
        detectionBuffer = VK_NULL_HANDLE;
    }
    if (detectionBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, detectionBufferMemory, nullptr);
        detectionBufferMemory = VK_NULL_HANDLE;
    }
    
    // Create new buffer
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = requiredSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &detectionBuffer) != VK_SUCCESS) {
        std::cerr << "[PoseOverlay] Failed to create detection buffer" << std::endl;
        return false;
    }
    
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, detectionBuffer, &memReq);
    
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = engine_->findMemoryType(memReq.memoryTypeBits, 
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &detectionBufferMemory) != VK_SUCCESS) {
        std::cerr << "[PoseOverlay] Failed to allocate detection buffer memory" << std::endl;
        vkDestroyBuffer(device, detectionBuffer, nullptr);
        detectionBuffer = VK_NULL_HANDLE;
        return false;
    }
    
    if (vkBindBufferMemory(device, detectionBuffer, detectionBufferMemory, 0) != VK_SUCCESS) {
        std::cerr << "[PoseOverlay] Failed to bind detection buffer memory" << std::endl;
        vkDestroyBuffer(device, detectionBuffer, nullptr);
        vkFreeMemory(device, detectionBufferMemory, nullptr);
        detectionBuffer = VK_NULL_HANDLE;
        detectionBufferMemory = VK_NULL_HANDLE;
        return false;
    }
    
    detectionBufferSize = requiredSize;
    return true;
}
