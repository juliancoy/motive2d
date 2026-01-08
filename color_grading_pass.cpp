// color_grading_pass.cpp
// RGBA-in -> Graded RGBA-out (same size), NO crop/target, NO push constants
// Display2D decoupled: caller supplies output extent/format via resize() and consumes outputs via output().

#include "color_grading_pass.h"

#include "engine2d.h"
#include "utils.h"
#include "debug_logging.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <glm/glm.hpp>

static const std::array<float, kCurveLutSize>& identityCurveLut()
{
    // Identity curve: output = input for each channel.
    // Layout expectation: shader samples curve as 256 entries (often 4x 64 segments),
    // but your UBO packs 64 vec4s => 256 floats. We fill them as:
    // [0..255] = i/255.
    static const std::array<float, kCurveLutSize> lut = [] {
        std::array<float, kCurveLutSize> a{};
        for (size_t i = 0; i < kCurveLutSize; ++i)
            a[i] = static_cast<float>(i) / 255.0f;
        return a;
    }();
    return lut;
}

namespace
{
static void destroyImageAndView(VkDevice device,
                               VkImage& img,
                               VkImageView& view,
                               VkDeviceMemory& mem)
{
    if (view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    if (img != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, img, nullptr);
        img = VK_NULL_HANDLE;
    }
    if (mem != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, mem, nullptr);
        mem = VK_NULL_HANDLE;
    }
}

static VkImageMemoryBarrier makeImageBarrier(VkImage image,
                                             VkImageLayout oldLayout,
                                             VkImageLayout newLayout,
                                             VkAccessFlags srcAccess,
                                             VkAccessFlags dstAccess)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    return b;
}

static void ensureImageLayout(VkCommandBuffer cmd,
                              VkImage image,
                              VkImageLayout& trackedLayout,
                              VkImageLayout desiredLayout,
                              VkAccessFlags srcAccess,
                              VkAccessFlags dstAccess,
                              VkPipelineStageFlags srcStage,
                              VkPipelineStageFlags dstStage)
{
    if (cmd == VK_NULL_HANDLE || image == VK_NULL_HANDLE || trackedLayout == desiredLayout)
        return;

    VkImageMemoryBarrier b = makeImageBarrier(image, trackedLayout, desiredLayout, srcAccess, dstAccess);

    vkCmdPipelineBarrier(cmd,
                         srcStage,
                         dstStage,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &b);

    trackedLayout = desiredLayout;
}
} // namespace

ColorGrading::ColorGrading(Engine2D* eng, uint32_t framesInFlight)
    : engine(eng), framesInFlight_(framesInFlight)
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("ColorGrading requires a valid Engine2D");

    if (framesInFlight_ == 0)
        throw std::runtime_error("ColorGrading: framesInFlight must be > 0");

    outImages_.assign(framesInFlight_, VK_NULL_HANDLE);
    outMem_.assign(framesInFlight_, VK_NULL_HANDLE);
    outViews_.assign(framesInFlight_, VK_NULL_HANDLE);
    outLayouts_.assign(framesInFlight_, VK_IMAGE_LAYOUT_UNDEFINED);
    descriptorSets_.assign(framesInFlight_, VK_NULL_HANDLE);

    // 256 floats packed into 64 vec4s (matches your shader UBO layout).
    curveUBOSize_ = sizeof(glm::vec4) * 64;

    createPipeline_();
    createCurveResources_();
    // Outputs + descriptors are created lazily in resize().
}

ColorGrading::~ColorGrading()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(engine->logicalDevice);

    destroyDescriptors_();
    destroyOutputs_();
    destroyCurveResources_();
    destroyPipeline_();
}

void ColorGrading::resize(VkExtent2D extent, VkFormat format)
{
    if (!engine) return;

    if (extent.width == 0 || extent.height == 0 || format == VK_FORMAT_UNDEFINED)
        return;

    const bool formatChanged = (outputFormat_ != format);
    const bool extentChanged = (outputExtent_.width != extent.width || outputExtent_.height != extent.height);

    if (!formatChanged && !extentChanged)
        return;

    outputFormat_ = format;
    outputExtent_ = extent;

    createOutputs_();
    createDescriptors_();
    rebuildDescriptorSets_();
}

void ColorGrading::setInputRGBA(VkImageView rgbaView, VkSampler rgbaSampler)
{
    rgbaView_ = rgbaView;
    rgbaSampler_ = rgbaSampler;

    if (descriptorPool_ != VK_NULL_HANDLE)
        rebuildDescriptorSets_();
}

ColorGrading::Output ColorGrading::output(uint32_t frameIndex) const
{
    Output o{};
    if (framesInFlight_ == 0)
        return o;

    const uint32_t fi = frameIndex % framesInFlight_;

    o.image = (fi < outImages_.size()) ? outImages_[fi] : VK_NULL_HANDLE;
    o.view  = (fi < outViews_.size()) ? outViews_[fi] : VK_NULL_HANDLE;
    o.layout = (fi < outLayouts_.size()) ? outLayouts_[fi] : VK_IMAGE_LAYOUT_UNDEFINED;
    o.extent = outputExtent_;
    o.format = outputFormat_;
    return o;
}

VkImageLayout ColorGrading::outputLayout(uint32_t frameIndex) const
{
    if (framesInFlight_ == 0 || outLayouts_.empty())
        return VK_IMAGE_LAYOUT_UNDEFINED;
    return outLayouts_[frameIndex % framesInFlight_];
}

void ColorGrading::dispatch(VkCommandBuffer cmd, uint32_t frameIndex)
{
    if (!engine || cmd == VK_NULL_HANDLE)
        return;

    if (framesInFlight_ == 0)
        return;

    // Must have a valid output spec.
    if (outputFormat_ == VK_FORMAT_UNDEFINED || outputExtent_.width == 0 || outputExtent_.height == 0)
        return;

    const uint32_t fi = frameIndex % framesInFlight_;

    if (pipeline_ == VK_NULL_HANDLE || pipelineLayout_ == VK_NULL_HANDLE)
        return;

    if (fi >= descriptorSets_.size() || descriptorSets_[fi] == VK_NULL_HANDLE)
        return;

    if (fi >= outImages_.size() || outImages_[fi] == VK_NULL_HANDLE)
        return;

    // Require RGBA input.
    if (rgbaView_ == VK_NULL_HANDLE || rgbaSampler_ == VK_NULL_HANDLE)
        return;

    // Upload curve if needed.
    applyCurve();

    // Output must be GENERAL for imageStore().
    ensureImageLayout(cmd,
                      outImages_[fi],
                      outLayouts_[fi],
                      VK_IMAGE_LAYOUT_GENERAL,
                      0,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    if (renderDebugEnabled())
    {
        std::cout << "[ColorGrading] dispatch fi=" << fi
                  << " extent=" << outputExtent_.width << "x" << outputExtent_.height
                  << std::endl;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout_,
                            0,
                            1,
                            &descriptorSets_[fi],
                            0,
                            nullptr);

    const uint32_t groupX = (outputExtent_.width + 15u) / 16u;
    const uint32_t groupY = (outputExtent_.height + 15u) / 16u;
    vkCmdDispatch(cmd, groupX, groupY, 1);

    // Make shader writes visible.
    VkImageMemoryBarrier after = makeImageBarrier(outImages_[fi],
                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                  VK_ACCESS_SHADER_WRITE_BIT,
                                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &after);

    // If the next consumer is sampling (e.g., Display2D), end in SHADER_READ_ONLY_OPTIMAL.
    ensureImageLayout(cmd,
                      outImages_[fi],
                      outLayouts_[fi],
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void ColorGrading::applyCurve()
{
    const bool wantCurve = adjustments && adjustments->curveEnabled;
    const std::array<float, kCurveLutSize>& curveData =
        wantCurve ? adjustments->curveLut : identityCurveLut();

    if (!curveUBOMapped_)
        return;

    const bool needsUpload =
        !curveUploaded_ ||
        (wantCurve != lastCurveEnabled_) ||
        (std::memcmp(lastCurveLut_.data(), curveData.data(), sizeof(float) * kCurveLutSize) != 0);

    if (!needsUpload)
        return;

    uploadCurveData_(curveData);
    lastCurveLut_ = curveData;
    lastCurveEnabled_ = wantCurve;
    curveUploaded_ = true;
}

void ColorGrading::createPipeline_()
{
    // Bindings must match your GLSL:
    // 0 = outImage (storage)
    // 1 = texRGBA (combined sampler)
    // 5 = curveUBO (uniform buffer)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 5;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = static_cast<uint32_t>(bindings.size());
    dsl.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(engine->logicalDevice, &dsl, nullptr, &setLayout_) != VK_SUCCESS)
        throw std::runtime_error("ColorGrading: failed to create descriptor set layout");

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 128; // HACK: Guessing push constant size, max is 256

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &setLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(engine->logicalDevice, &pli, nullptr, &pipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("ColorGrading: failed to create pipeline layout");

    auto shaderCode = readSPIRVFile("shaders/color_grading_pass.spv");
    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = stage;
    cpi.layout = pipelineLayout_;

    if (vkCreateComputePipelines(engine->logicalDevice, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
        throw std::runtime_error("ColorGrading: failed to create compute pipeline");
    }

    vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
}

void ColorGrading::destroyPipeline_()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    if (pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(engine->logicalDevice, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (setLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(engine->logicalDevice, setLayout_, nullptr);
        setLayout_ = VK_NULL_HANDLE;
    }
}

void ColorGrading::createCurveResources_()
{
    destroyCurveResources_();

    if (!engine) return;

    engine->createBuffer(static_cast<VkDeviceSize>(curveUBOSize_),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         curveUBO_,
                         curveUBOMemory_);

    vkMapMemory(engine->logicalDevice, curveUBOMemory_, 0, curveUBOSize_, 0, &curveUBOMapped_);

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

void ColorGrading::destroyCurveResources_()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    if (curveUBOMapped_)
    {
        vkUnmapMemory(engine->logicalDevice, curveUBOMemory_);
        curveUBOMapped_ = nullptr;
    }
    if (curveUBO_ != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(engine->logicalDevice, curveUBO_, nullptr);
        curveUBO_ = VK_NULL_HANDLE;
    }
    if (curveUBOMemory_ != VK_NULL_HANDLE)
    {
        vkFreeMemory(engine->logicalDevice, curveUBOMemory_, nullptr);
        curveUBOMemory_ = VK_NULL_HANDLE;
    }

    curveUploaded_ = false;
}

void ColorGrading::uploadCurveData_(const std::array<float, kCurveLutSize>& curveData)
{
    if (!curveUBOMapped_)
        return;

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

void ColorGrading::createOutputs_()
{
    destroyOutputs_();

    if (!engine)
        return;

    if (outputFormat_ == VK_FORMAT_UNDEFINED || outputExtent_.width == 0 || outputExtent_.height == 0)
        return;

    outImages_.assign(framesInFlight_, VK_NULL_HANDLE);
    outMem_.assign(framesInFlight_, VK_NULL_HANDLE);
    outViews_.assign(framesInFlight_, VK_NULL_HANDLE);
    outLayouts_.assign(framesInFlight_, VK_IMAGE_LAYOUT_UNDEFINED);

    for (uint32_t i = 0; i < framesInFlight_; ++i)
    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = outputFormat_;
        ii.extent = VkExtent3D{outputExtent_.width, outputExtent_.height, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        // Must be STORAGE for compute writes; include SAMPLED for downstream sampling.
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(engine->logicalDevice, &ii, nullptr, &outImages_[i]) != VK_SUCCESS)
            throw std::runtime_error("ColorGrading: failed to create output image");

        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(engine->logicalDevice, outImages_[i], &mr);

        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = engine->findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(engine->logicalDevice, &ai, nullptr, &outMem_[i]) != VK_SUCCESS)
            throw std::runtime_error("ColorGrading: failed to allocate output image memory");

        vkBindImageMemory(engine->logicalDevice, outImages_[i], outMem_[i], 0);

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = outImages_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = outputFormat_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel = 0;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount = 1;

        if (vkCreateImageView(engine->logicalDevice, &vi, nullptr, &outViews_[i]) != VK_SUCCESS)
            throw std::runtime_error("ColorGrading: failed to create output image view");

        outLayouts_[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

void ColorGrading::destroyOutputs_()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    for (uint32_t i = 0; i < outImages_.size(); ++i)
    {
        destroyImageAndView(engine->logicalDevice, outImages_[i], outViews_[i], outMem_[i]);
        outLayouts_[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    outImages_.assign(framesInFlight_, VK_NULL_HANDLE);
    outMem_.assign(framesInFlight_, VK_NULL_HANDLE);
    outViews_.assign(framesInFlight_, VK_NULL_HANDLE);
    outLayouts_.assign(framesInFlight_, VK_IMAGE_LAYOUT_UNDEFINED);
}

void ColorGrading::createDescriptors_()
{
    destroyDescriptors_();

    if (!engine || setLayout_ == VK_NULL_HANDLE)
        return;

    descriptorSets_.assign(framesInFlight_, VK_NULL_HANDLE);

    std::array<VkDescriptorPoolSize, 3> sizes{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[0].descriptorCount = framesInFlight_;

    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = framesInFlight_;

    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[2].descriptorCount = framesInFlight_;

    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pi.pPoolSizes = sizes.data();
    pi.maxSets = framesInFlight_;

    if (vkCreateDescriptorPool(engine->logicalDevice, &pi, nullptr, &descriptorPool_) != VK_SUCCESS)
        throw std::runtime_error("ColorGrading: failed to create descriptor pool");

    std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, setLayout_);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descriptorPool_;
    ai.descriptorSetCount = framesInFlight_;
    ai.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(engine->logicalDevice, &ai, descriptorSets_.data()) != VK_SUCCESS)
        throw std::runtime_error("ColorGrading: failed to allocate descriptor sets");
}

void ColorGrading::destroyDescriptors_()
{
    if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
        return;

    if (descriptorPool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(engine->logicalDevice, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }

    descriptorSets_.assign(framesInFlight_, VK_NULL_HANDLE);
}

void ColorGrading::rebuildDescriptorSets_()
{
    if (!engine || descriptorPool_ == VK_NULL_HANDLE)
        return;

    if (descriptorSets_.size() != framesInFlight_)
        return;

    for (uint32_t i = 0; i < framesInFlight_; ++i)
    {
        if (i >= outViews_.size() || outViews_[i] == VK_NULL_HANDLE)
            return;
        if (descriptorSets_[i] == VK_NULL_HANDLE)
            return;
    }

    for (uint32_t i = 0; i < framesInFlight_; ++i)
    {
        std::array<VkWriteDescriptorSet, 3> writes{};

        // binding 0: output storage image
        VkDescriptorImageInfo outInfo{};
        outInfo.imageView = outViews_[i];
        outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &outInfo;

        // binding 1: RGBA sampled input
        VkDescriptorImageInfo rgbaInfo{};
        rgbaInfo.imageView = rgbaView_;
        rgbaInfo.sampler = rgbaSampler_;
        rgbaInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &rgbaInfo;

        // binding 5: curve UBO
        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = curveUBO_;
        bufInfo.offset = 0;
        bufInfo.range = static_cast<VkDeviceSize>(curveUBOSize_);

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptorSets_[i];
        writes[2].dstBinding = 5;
        writes[2].dstArrayElement = 0;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &bufInfo;

        vkUpdateDescriptorSets(engine->logicalDevice,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(),
                               0,
                               nullptr);
    }
}
