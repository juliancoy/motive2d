#include "color_grading_pass.h"

#include "display2d.h"
#include "engine2d.h"
#include "utils.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <nlohmann/json.hpp>

ColorGrading::ColorGrading(Display2D* display)
    : display_(display), engine_(display ? display->engine : nullptr)
{

    // 3: luma
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 4: chroma
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 5: curve LUT UBO
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    destroyPipeline();
    if (!engine_ || pipelineLayout == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Invalid parameters while creating grading blit pipeline");
    }

    pipelineLayout_ = pipelineLayout;


    // Pipeline layout with push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(CropPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(engine->logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout for Display2D");
    }

    // Compute pipeline
    destroyPipeline();
    createPipeline(pipelineLayout);

    createCurveResources();

    auto shaderCode = readSPIRVFile("shaders/grading_pass.comp.spv");
    VkShaderModule shaderModule = engine_->createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout_;

    if (vkCreateComputePipelines(engine_->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(engine_->logicalDevice, shaderModule, nullptr);
        pipeline_ = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to create grading blit compute pipeline");
    }

    vkDestroyShaderModule(engine_->logicalDevice, shaderModule, nullptr);

}

ColorGrading::~ColorGrading()
{
    destroyPipeline();
    destroyCurveResources();
    destroyGradingImages();
}


void ColorGrading::destroyPipeline()
{
    if (engine_ && pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine_->logicalDevice, pipeline_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
}

void ColorGrading::createCurveResources()
{
    destroyCurveResources();

    if (!engine_)
    {
        return;
    }

    engine_->createBuffer(curveUBOSize_,
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          curveUBO_,
                          curveUBOMemory_);
    vkMapMemory(engine_->logicalDevice, curveUBOMemory_, 0, curveUBOSize_, 0, &curveUBOMapped_);

    if (curveUBOMapped_)
    {
        std::array<glm::vec4, 64> packed{};
        const auto& idLut = identityCurveLut();
        for (size_t i = 0; i < packed.size(); ++i)
        {
            packed[i] = glm::vec4(idLut[i * 4 + 0],
                                  idLut[i * 4 + 1],
                                  idLut[i * 4 + 2],
                                  idLut[i * 4 + 3]);
        }
        std::memcpy(curveUBOMapped_, packed.data(), curveUBOSize_);
        lastCurveLut_ = idLut;
        lastCurveEnabled_ = false;
        curveUploaded_ = true;
    }
}

void ColorGrading::destroyCurveResources()
{
    if (curveUBOMapped_)
    {
        vkUnmapMemory(engine_->logicalDevice, curveUBOMemory_);
        curveUBOMapped_ = nullptr;
    }
    if (curveUBO_ != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(engine_->logicalDevice, curveUBO_, nullptr);
        curveUBO_ = VK_NULL_HANDLE;
    }
    if (curveUBOMemory_ != VK_NULL_HANDLE)
    {
        vkFreeMemory(engine_->logicalDevice, curveUBOMemory_, nullptr);
        curveUBOMemory_ = VK_NULL_HANDLE;
    }
    curveUploaded_ = false;
}

void ColorGrading::createGradingImages()
{
    destroyGradingImages();

    if (!engine_ || !display_)
    {
        return;
    }

    const size_t count = display_->swapchainImages.size();
    if (count == 0)
    {
        return;
    }

    gradingImages.resize(count);
    gradingImageMemories.resize(count);
    gradingImageViews.resize(count);
    gradingImageLayouts.assign(count, VK_IMAGE_LAYOUT_UNDEFINED);

    for (size_t i = 0; i < count; ++i)
    {
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = display_->swapchainFormat;
        imageInfo.extent = VkExtent3D{display_->swapchainExtent.width, display_->swapchainExtent.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(engine_->logicalDevice, &imageInfo, nullptr, &gradingImages[i]) != VK_SUCCESS)
        {
            destroyGradingImages();
            throw std::runtime_error("Failed to create grading intermediate image");
        }

        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(engine_->logicalDevice, gradingImages[i], &memReq);

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = engine_->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(engine_->logicalDevice, &allocInfo, nullptr, &gradingImageMemories[i]) != VK_SUCCESS)
        {
            destroyGradingImages();
            throw std::runtime_error("Failed to allocate grading intermediate image memory");
        }

        vkBindImageMemory(engine_->logicalDevice, gradingImages[i], gradingImageMemories[i], 0);

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = gradingImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = display_->swapchainFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(engine_->logicalDevice, &viewInfo, nullptr, &gradingImageViews[i]) != VK_SUCCESS)
        {
            destroyGradingImages();
            throw std::runtime_error("Failed to create grading intermediate image view");
        }
    }
}

void ColorGrading::destroyGradingImages()
{
    if (!engine_)
    {
        return;
    }

    for (auto view : gradingImageViews)
    {
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(engine_->logicalDevice, view, nullptr);
        }
    }
    for (auto image : gradingImages)
    {
        if (image != VK_NULL_HANDLE)
        {
            vkDestroyImage(engine_->logicalDevice, image, nullptr);
        }
    }
    for (auto mem : gradingImageMemories)
    {
        if (mem != VK_NULL_HANDLE)
        {
            vkFreeMemory(engine_->logicalDevice, mem, nullptr);
        }
    }
    gradingImages.clear();
    gradingImageViews.clear();
    gradingImageMemories.clear();
    gradingImageLayouts.clear();
}

void ColorGrading::applyCurve()
{
    const bool wantCurve = adjustments && adjustments->curveEnabled;
    const std::array<float, kCurveLutSize>& curveData = wantCurve ? adjustments->curveLut : identityCurveLut();
    if (!curveUBOMapped_)
    {
        return;
    }

    const bool needsUpload = !curveUploaded_ ||
                             (wantCurve != lastCurveEnabled_) ||
                             (std::memcmp(lastCurveLut_.data(), curveData.data(), sizeof(float) * kCurveLutSize) != 0);
    if (!needsUpload)
    {
        return;
    }

    uploadCurveData(curveData);
    lastCurveLut_ = curveData;
    lastCurveEnabled_ = wantCurve;
    curveUploaded_ = true;
}

void ColorGrading::uploadCurveData(const std::array<float, kCurveLutSize>& curveData)
{
    if (!curveUBOMapped_)
    {
        return;
    }

    std::array<glm::vec4, 64> packed{};
    for (size_t i = 0; i < packed.size(); ++i)
    {
        packed[i] = glm::vec4(curveData[i * 4 + 0],
                              curveData[i * 4 + 1],
                              curveData[i * 4 + 2],
                              curveData[i * 4 + 3]);
    }
    std::memcpy(curveUBOMapped_, packed.data(), curveUBOSize_);
}

void ColorGrading::dispatch(VkCommandBuffer commandBuffer,
                            VkPipelineLayout pipelineLayout,
                            VkDescriptorSet descriptorSet,
                            uint32_t groupX,
                            uint32_t groupY)
{
    if (commandBuffer == VK_NULL_HANDLE || pipeline_ == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE)
    {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);
    vkCmdDispatch(commandBuffer, groupX, groupY, 1);
}
