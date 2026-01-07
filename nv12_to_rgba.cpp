// nv12_to_rgba.cpp  (NV12 -> RGBA, pass owns output)
#include "nv12_to_rgba.h"

#include "engine2d.h"
#include "utils.h"
#include "debug_logging.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
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

static void destroyImageAndView(VkDevice device, VkImage& img, VkImageView& view, VkDeviceMemory& mem)
{
    if (view != VK_NULL_HANDLE) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
    if (img != VK_NULL_HANDLE)  { vkDestroyImage(device, img, nullptr);     img = VK_NULL_HANDLE; }
    if (mem != VK_NULL_HANDLE)  { vkFreeMemory(device, mem, nullptr);       mem = VK_NULL_HANDLE; }
}

// For sampler2D inputs, descriptor type must be COMBINED_IMAGE_SAMPLER.
static bool isSampledInputType(VkDescriptorType t)
{
    return t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
}
} // namespace

Nv12ToRgbaPass::Nv12ToRgbaPass(Engine2D* engine,
                               uint32_t framesInFlight,
                               int width,
                               int height,
                               VkDescriptorType inputDescriptorType)
    : engine_(engine),
      framesInFlight_(framesInFlight),
      width_(width),
      height_(height),
      inputDescriptorType_(inputDescriptorType)
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("Nv12ToRgbaPass: invalid engine");

    if (framesInFlight_ == 0 || framesInFlight_ > 16)
        throw std::runtime_error("Nv12ToRgbaPass: framesInFlight must be 1..16");

    if (width_ <= 0 || height_ <= 0)
        throw std::runtime_error("Nv12ToRgbaPass: invalid dimensions");

    // Your current GLSL uses sampler2D for yTex/uvTex:
    // layout(set=0,binding=0) uniform sampler2D yTex;
    // layout(set=0,binding=1) uniform sampler2D uvTex;
    //
    // That requires COMBINED_IMAGE_SAMPLER descriptors.
    if (!isSampledInputType(inputDescriptorType_))
        throw std::runtime_error("Nv12ToRgbaPass: shader uses sampler2D, so inputDescriptorType must be VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER");
}

Nv12ToRgbaPass::~Nv12ToRgbaPass()
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(engine_->logicalDevice);

    destroyDescriptors_();
    destroyOutputs_();
    destroyOutputSampler_();
    destroyPipeline_();
}

void Nv12ToRgbaPass::initialize()
{
    if (initialized_) return;

    createPipeline_();
    createOutputs_();
    createDescriptors_();
    createOutputSampler_();

    initialized_ = true;

    if (renderDebugEnabled())
        std::cout << "[Nv12ToRgbaPass] initialized framesInFlight=" << framesInFlight_
                  << " size=" << width_ << "x" << height_ << std::endl;
}

void Nv12ToRgbaPass::resize(int width, int height)
{
    if (width == width_ && height == height_)
        return;

    width_ = width;
    height_ = height;

    pushConstants.rgbaSize = glm::ivec2(width_, height_);
    pushConstants.uvSize   = glm::ivec2(width_ / 2, height_ / 2);

    destroyDescriptors_();
    destroyOutputs_();

    createOutputs_();
    createDescriptors_();
    rebuildDescriptorSets_();
}

void Nv12ToRgbaPass::setInputNV12(VkImageView yView,
                                  VkImageView uvView,
                                  VkSampler ySampler,
                                  VkSampler uvSampler)
{
    yView_ = yView;
    uvView_ = uvView;
    ySampler_ = ySampler;
    uvSampler_ = uvSampler;

    // For sampler2D, samplers must be provided.
    if (yView_ != VK_NULL_HANDLE && uvView_ != VK_NULL_HANDLE)
    {
        if (ySampler_ == VK_NULL_HANDLE || uvSampler_ == VK_NULL_HANDLE)
            throw std::runtime_error("Nv12ToRgbaPass: sampler2D inputs require non-null ySampler + uvSampler");
    }

    if (descriptorPool_ != VK_NULL_HANDLE)
        rebuildDescriptorSets_();
}

void Nv12ToRgbaPass::dispatch(VkCommandBuffer cmd, uint32_t frameIndex)
{
    if (!initialized_ || cmd == VK_NULL_HANDLE)
        return;

    if (pipeline_ == VK_NULL_HANDLE || pipelineLayout_ == VK_NULL_HANDLE)
        return;

    // Require inputs (views + samplers) because shader uses sampler2D.
    if (yView_ == VK_NULL_HANDLE || uvView_ == VK_NULL_HANDLE || ySampler_ == VK_NULL_HANDLE || uvSampler_ == VK_NULL_HANDLE)
        return;

    const uint32_t fi = frameIndex % framesInFlight_;
    if (fi >= outImages_.size() || fi >= descriptorSets_.size())
        return;

    // Output must be GENERAL for imageStore()
    if (outLayouts_[fi] != VK_IMAGE_LAYOUT_GENERAL)
    {
        VkImageMemoryBarrier b = makeImageBarrier(outImages_[fi],
                                                  outLayouts_[fi],
                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                  0,
                                                  VK_ACCESS_SHADER_WRITE_BIT);

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &b);

        outLayouts_[fi] = VK_IMAGE_LAYOUT_GENERAL;
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

    vkCmdPushConstants(cmd,
                       pipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(nv12toBGRPushConstants),
                       &pushConstants);

    const uint32_t groupX = (static_cast<uint32_t>(width_) + 15u) / 16u;
    const uint32_t groupY = (static_cast<uint32_t>(height_) + 15u) / 16u;
    vkCmdDispatch(cmd, groupX, groupY, 1);
}

PresentInput Nv12ToRgbaPass::output(uint32_t frameIndex) const
{
    const uint32_t fi = frameIndex % framesInFlight_;

    PresentInput out{};
    if (fi < outImages_.size())
    {
        out.image = outImages_[fi];
        out.view = outViews_[fi];
        out.layout = outLayouts_[fi];
        out.extent = VkExtent2D{static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};
        out.format = outFormat_;
    }
    return out;
}

VkImageView Nv12ToRgbaPass::outputView(uint32_t frameIndex) const
{
    return outViews_[frameIndex % framesInFlight_];
}

VkImage Nv12ToRgbaPass::outputImage(uint32_t frameIndex) const
{
    return outImages_[frameIndex % framesInFlight_];
}

void Nv12ToRgbaPass::createPipeline_()
{
    // bindings 0,1,2 must match SPIR-V compiled from the GLSL:
    // 0 = yTex (sampler2D)  -> COMBINED_IMAGE_SAMPLER
    // 1 = uvTex (sampler2D) -> COMBINED_IMAGE_SAMPLER
    // 2 = rgbaOutput (storage image)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = static_cast<uint32_t>(bindings.size());
    dsl.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(engine_->logicalDevice, &dsl, nullptr, &descriptorSetLayout_) != VK_SUCCESS)
        throw std::runtime_error("Nv12ToRgbaPass: failed to create descriptor set layout");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(nv12toBGRPushConstants);

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &descriptorSetLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;

    if (vkCreatePipelineLayout(engine_->logicalDevice, &pli, nullptr, &pipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("Nv12ToRgbaPass: failed to create pipeline layout");

    // Make sure this SPIR-V is compiled from the sampler2D version of the shader.
    auto shaderCode = readSPIRVFile("shaders/nv12_to_rgba.spv");
    VkShaderModule shaderModule = engine_->createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = stage;
    cpi.layout = pipelineLayout_;

    if (vkCreateComputePipelines(engine_->logicalDevice, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(engine_->logicalDevice, shaderModule, nullptr);
        throw std::runtime_error("Nv12ToRgbaPass: failed to create compute pipeline");
    }

    vkDestroyShaderModule(engine_->logicalDevice, shaderModule, nullptr);
}

void Nv12ToRgbaPass::destroyPipeline_()
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        return;

    if (pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine_->logicalDevice, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(engine_->logicalDevice, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(engine_->logicalDevice, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }
}

void Nv12ToRgbaPass::createOutputs_()
{
    destroyOutputs_();

    outImages_.resize(framesInFlight_, VK_NULL_HANDLE);
    outMem_.resize(framesInFlight_, VK_NULL_HANDLE);
    outViews_.resize(framesInFlight_, VK_NULL_HANDLE);
    outLayouts_.assign(framesInFlight_, VK_IMAGE_LAYOUT_UNDEFINED);

    for (uint32_t i = 0; i < framesInFlight_; ++i)
    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = outFormat_;
        ii.extent = VkExtent3D{static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; // written by compute, can be sampled downstream
        ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(engine_->logicalDevice, &ii, nullptr, &outImages_[i]) != VK_SUCCESS)
            throw std::runtime_error("Nv12ToRgbaPass: failed to create output image");

        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(engine_->logicalDevice, outImages_[i], &mr);

        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = engine_->findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(engine_->logicalDevice, &ai, nullptr, &outMem_[i]) != VK_SUCCESS)
            throw std::runtime_error("Nv12ToRgbaPass: failed to allocate output image memory");

        vkBindImageMemory(engine_->logicalDevice, outImages_[i], outMem_[i], 0);

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = outImages_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = outFormat_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel = 0;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount = 1;

        if (vkCreateImageView(engine_->logicalDevice, &vi, nullptr, &outViews_[i]) != VK_SUCCESS)
            throw std::runtime_error("Nv12ToRgbaPass: failed to create output image view");
    }

    // Default push constants
    pushConstants.rgbaSize = glm::ivec2(width_, height_);
    pushConstants.uvSize   = glm::ivec2(width_ / 2, height_ / 2);
}

void Nv12ToRgbaPass::destroyOutputs_()
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        return;

    for (uint32_t i = 0; i < outImages_.size(); ++i)
        destroyImageAndView(engine_->logicalDevice, outImages_[i], outViews_[i], outMem_[i]);

    outImages_.clear();
    outViews_.clear();
    outMem_.clear();
    outLayouts_.clear();
}

void Nv12ToRgbaPass::createDescriptors_()
{
    destroyDescriptors_();

    if (descriptorSetLayout_ == VK_NULL_HANDLE)
        return;

    // One set per in-flight slot.
    std::array<VkDescriptorPoolSize, 2> sizes{};

    // yTex + uvTex are COMBINED_IMAGE_SAMPLER (2 per set)
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = framesInFlight_ * 2;

    // rgbaOutput is STORAGE_IMAGE (1 per set)
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = framesInFlight_;

    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pi.pPoolSizes = sizes.data();
    pi.maxSets = framesInFlight_;

    if (vkCreateDescriptorPool(engine_->logicalDevice, &pi, nullptr, &descriptorPool_) != VK_SUCCESS)
        throw std::runtime_error("Nv12ToRgbaPass: failed to create descriptor pool");

    std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, descriptorSetLayout_);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descriptorPool_;
    ai.descriptorSetCount = framesInFlight_;
    ai.pSetLayouts = layouts.data();

    descriptorSets_.resize(framesInFlight_, VK_NULL_HANDLE);
    if (vkAllocateDescriptorSets(engine_->logicalDevice, &ai, descriptorSets_.data()) != VK_SUCCESS)
        throw std::runtime_error("Nv12ToRgbaPass: failed to allocate descriptor sets");

    rebuildDescriptorSets_();
}

void Nv12ToRgbaPass::destroyDescriptors_()
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        return;

    if (descriptorPool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(engine_->logicalDevice, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }

    descriptorSets_.clear();
}

void Nv12ToRgbaPass::rebuildDescriptorSets_()
{
    if (!engine_ || descriptorPool_ == VK_NULL_HANDLE)
        return;

    // If inputs aren't set yet, skip; we’ll rebuild when setInputNV12 is called.
    if (yView_ == VK_NULL_HANDLE || uvView_ == VK_NULL_HANDLE || ySampler_ == VK_NULL_HANDLE || uvSampler_ == VK_NULL_HANDLE)
        return;

    for (uint32_t i = 0; i < framesInFlight_; ++i)
    {
        std::array<VkWriteDescriptorSet, 3> writes{};

        // Binding 0: yTex (combined sampler)
        VkDescriptorImageInfo yInfo{};
        yInfo.imageView = yView_;
        yInfo.sampler = ySampler_;
        yInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &yInfo;

        // Binding 1: uvTex (combined sampler)
        VkDescriptorImageInfo uvInfo{};
        uvInfo.imageView = uvView_;
        uvInfo.sampler = uvSampler_;
        uvInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &uvInfo;

        // Binding 2: rgbaOutput (storage image)
        VkDescriptorImageInfo outInfo{};
        outInfo.imageView = outViews_[i];
        outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptorSets_[i];
        writes[2].dstBinding = 2;
        writes[2].dstArrayElement = 0;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &outInfo;

        vkUpdateDescriptorSets(engine_->logicalDevice,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(),
                               0,
                               nullptr);
    }
}

void Nv12ToRgbaPass::createOutputSampler_()
{
    destroyOutputSampler_();

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(engine_->logicalDevice, &si, nullptr, &outputSampler_) != VK_SUCCESS)
        throw std::runtime_error("Nv12ToRgbaPass: failed to create output sampler");
}

void Nv12ToRgbaPass::destroyOutputSampler_()
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        return;

    if (outputSampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(engine_->logicalDevice, outputSampler_, nullptr);
        outputSampler_ = VK_NULL_HANDLE;
    }
}
