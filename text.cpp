// text.cpp
// "Text" pass with the SAME BOUNDS as ColorGrading:
//
// - Constructor(engine, framesInFlight)
// - resize(extent, format) creates outputs + per-frame descriptor sets lazily
// - setAtlas(view, sampler) sets the input (borrowed resources)
// - upload(frameIndex, glyphs, spans, tileGlyphs) updates per-frame SSBOs (host visible, mapped)
// - dispatch(cmd, frameIndex) writes into this pass' owned output image for that frame
// - output(frameIndex) / outputLayout(frameIndex) expose produced image/view/layout/extent/format
//
// Shader contract (set=0):
//   binding 0: storage image outImage          (per-frame)
//   binding 1: combined sampler fontAtlas      (shared)
//   binding 2: SSBO TileSpans                 (per-frame)
//   binding 3: SSBO TileGlyphs                (per-frame)
//   binding 4: SSBO GlyphInstances            (per-frame)
//
// Push constants:
//   ivec2 imageSize
//   ivec2 tileGridSize
//   float sdfPxRange
//
// NOTE: This is a renderer; it does not do shaping/layout/atlas baking.
//       You feed glyph instances + tile culling lists.
//
// SPIR-V expected at: shaders/text_sdf.spv

#include "engine2d.h"
#include "utils.h"
#include "debug_logging.h"

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace
{
static void destroyImageAndView(VkDevice device,
                               VkImage& img,
                               VkImageView& view,
                               VkDeviceMemory& mem)
{
    if (view != VK_NULL_HANDLE) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
    if (img  != VK_NULL_HANDLE) { vkDestroyImage(device, img, nullptr); img = VK_NULL_HANDLE; }
    if (mem  != VK_NULL_HANDLE) { vkFreeMemory(device, mem, nullptr); mem = VK_NULL_HANDLE; }
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
    if (cmd == VK_NULL_HANDLE || image == VK_NULL_HANDLE) return;
    if (trackedLayout == desiredLayout) return;

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

static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static void destroyBuffer(VkDevice device, VkBuffer& buf, VkDeviceMemory& mem, void*& mapped)
{
    if (mapped)
    {
        vkUnmapMemory(device, mem);
        mapped = nullptr;
    }
    if (buf != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, buf, nullptr);
        buf = VK_NULL_HANDLE;
    }
    if (mem != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, mem, nullptr);
        mem = VK_NULL_HANDLE;
    }
}
} // namespace

class Text
{
public:
    struct Output
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkExtent2D extent{0, 0};
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    // std430-friendly CPU struct for GlyphInstance
    // GLSL layout (std430):
    //   int x0,y0,x1,y1;
    //   float u0,v0,u1,v1;
    //   vec4 textColor;
    //   vec4 bgColor;
    //   uint flags;
    // We'll pad to 16B multiple.
    struct alignas(16) GlyphInstance
    {
        int32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
        glm::vec4 textColor{1, 1, 1, 1};
        glm::vec4 bgColor{0, 0, 0, 0};
        uint32_t flags = 0;
        uint32_t _pad0 = 0, _pad1 = 0, _pad2 = 0;
    };

    struct TileSpan
    {
        uint32_t start = 0;
        uint32_t count = 0;
    };

    Text(Engine2D* eng,
         uint32_t framesInFlight,
         uint32_t maxGlyphs = 4096,
         uint32_t maxTileGlyphRefs = 65536)
        : engine(eng)
        , framesInFlight_(framesInFlight)
        , maxGlyphs_(maxGlyphs)
        , maxTileGlyphRefs_(maxTileGlyphRefs)
    {
        if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("Text requires a valid Engine2D");
        if (framesInFlight_ == 0)
            throw std::runtime_error("Text: framesInFlight must be > 0");
        if (maxGlyphs_ == 0 || maxTileGlyphRefs_ == 0)
            throw std::runtime_error("Text: maxGlyphs/maxTileGlyphRefs must be > 0");

        outImages_.assign(framesInFlight_, VK_NULL_HANDLE);
        outMem_.assign(framesInFlight_, VK_NULL_HANDLE);
        outViews_.assign(framesInFlight_, VK_NULL_HANDLE);
        outLayouts_.assign(framesInFlight_, VK_IMAGE_LAYOUT_UNDEFINED);
        descriptorSets_.assign(framesInFlight_, VK_NULL_HANDLE);

        frame_.resize(framesInFlight_);

        createPipeline_();
        // Outputs + descriptors are created lazily in resize(), like ColorGrading.
    }

    ~Text()
    {
        if (!engine || engine->logicalDevice == VK_NULL_HANDLE)
            return;

        vkDeviceWaitIdle(engine->logicalDevice);

        destroyDescriptors_();
        destroyOutputs_();
        destroyFrameBuffers_();
        destroyPipeline_();
    }

    Text(const Text&) = delete;
    Text& operator=(const Text&) = delete;

    // Same bound as ColorGrading: caller sets output spec via resize().
    // format must be a storage-compatible RGBA format for outImage (e.g. VK_FORMAT_R8G8B8A8_UNORM).
    void resize(VkExtent2D extent, VkFormat format)
    {
        if (!engine) return;
        if (extent.width == 0 || extent.height == 0 || format == VK_FORMAT_UNDEFINED) return;

        const bool fmtChanged = (outputFormat_ != format);
        const bool extChanged = (outputExtent_.width != extent.width || outputExtent_.height != extent.height);

        if (!fmtChanged && !extChanged)
            return;

        outputFormat_ = format;
        outputExtent_ = extent;

        tileGridSize_.x = int32_t(ceilDiv(extent.width, 16));
        tileGridSize_.y = int32_t(ceilDiv(extent.height, 16));

        createOutputs_();
        createFrameBuffers_();   // buffers depend on tile count (spans size)
        createDescriptors_();
        rebuildDescriptorSets_();
    }

    // Input: atlas view+sampler (borrowed), like setInputRGBA()
    void setAtlas(VkImageView atlasView, VkSampler atlasSampler)
    {
        atlasView_ = atlasView;
        atlasSampler_ = atlasSampler;

        if (descriptorPool_ != VK_NULL_HANDLE)
            rebuildDescriptorSets_();
    }

    void setSdfPxRange(float pxRange) { sdfPxRange_ = std::max(0.0001f, pxRange); }

    // Upload per-frame data (caller should use the SAME frame index discipline as other passes)
    bool upload(uint32_t frameIndex,
                const std::vector<GlyphInstance>& glyphs,
                const std::vector<TileSpan>& spans,
                const std::vector<uint32_t>& tileGlyphs)
    {
        if (framesInFlight_ == 0) return false;
        const uint32_t fi = frameIndex % framesInFlight_;

        if (!frame_[fi].glyphMapped || !frame_[fi].spanMapped || !frame_[fi].tileGlyphMapped)
            return false;

        const uint32_t tileCount = uint32_t(tileGridSize_.x * tileGridSize_.y);
        if (tileCount == 0) return false;

        if (spans.size() != tileCount)
        {
            LOG_DEBUG(std::cout << "[Text] upload: spans.size()=" << spans.size()
                                << " expected tileCount=" << tileCount << std::endl);
            return false;
        }

        if (glyphs.size() > maxGlyphs_ || tileGlyphs.size() > maxTileGlyphRefs_)
        {
            LOG_DEBUG(std::cout << "[Text] upload: too many glyphs/refs "
                                << glyphs.size() << "/" << tileGlyphs.size()
                                << " max " << maxGlyphs_ << "/" << maxTileGlyphRefs_ << std::endl);
            return false;
        }

        // Copy data into persistently-mapped buffers
        if (!glyphs.empty())
            std::memcpy(frame_[fi].glyphMapped, glyphs.data(), glyphs.size() * sizeof(GlyphInstance));
        if (!spans.empty())
            std::memcpy(frame_[fi].spanMapped, spans.data(), spans.size() * sizeof(TileSpan));
        if (!tileGlyphs.empty())
            std::memcpy(frame_[fi].tileGlyphMapped, tileGlyphs.data(), tileGlyphs.size() * sizeof(uint32_t));

        frame_[fi].glyphCount = uint32_t(glyphs.size());
        frame_[fi].tileGlyphCount = uint32_t(tileGlyphs.size());
        frame_[fi].hasData = true;
        return true;
    }

    // Same contract style as ColorGrading
    Output output(uint32_t frameIndex) const
    {
        Output o{};
        if (framesInFlight_ == 0) return o;
        const uint32_t fi = frameIndex % framesInFlight_;

        o.image  = (fi < outImages_.size()) ? outImages_[fi] : VK_NULL_HANDLE;
        o.view   = (fi < outViews_.size()) ? outViews_[fi] : VK_NULL_HANDLE;
        o.layout = (fi < outLayouts_.size()) ? outLayouts_[fi] : VK_IMAGE_LAYOUT_UNDEFINED;
        o.extent = outputExtent_;
        o.format = outputFormat_;
        return o;
    }

    VkImageLayout outputLayout(uint32_t frameIndex) const
    {
        if (framesInFlight_ == 0 || outLayouts_.empty()) return VK_IMAGE_LAYOUT_UNDEFINED;
        return outLayouts_[frameIndex % framesInFlight_];
    }

    // Records compute dispatch to produce the per-frame output.
    // Like your ColorGrading, this ends the output in SHADER_READ_ONLY_OPTIMAL for downstream sampling.
    void dispatch(VkCommandBuffer cmd, uint32_t frameIndex)
    {
        if (!engine || cmd == VK_NULL_HANDLE) return;
        if (framesInFlight_ == 0) return;

        if (outputFormat_ == VK_FORMAT_UNDEFINED || outputExtent_.width == 0 || outputExtent_.height == 0)
            return;

        const uint32_t fi = frameIndex % framesInFlight_;

        if (pipeline_ == VK_NULL_HANDLE || pipelineLayout_ == VK_NULL_HANDLE) return;
        if (fi >= descriptorSets_.size() || descriptorSets_[fi] == VK_NULL_HANDLE) return;
        if (fi >= outImages_.size() || outImages_[fi] == VK_NULL_HANDLE) return;

        // Require atlas
        if (atlasView_ == VK_NULL_HANDLE || atlasSampler_ == VK_NULL_HANDLE) return;

        // Require data (you can decide to allow empty and just do nothing)
        if (!frame_[fi].hasData) return;

        // outImage must be GENERAL for imageStore
        ensureImageLayout(cmd,
                          outImages_[fi],
                          outLayouts_[fi],
                          VK_IMAGE_LAYOUT_GENERAL,
                          0,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_,
                                0,
                                1,
                                &descriptorSets_[fi],
                                0,
                                nullptr);

        // Push constants
        Push pc{};
        pc.imageSize = glm::ivec2(int32_t(outputExtent_.width), int32_t(outputExtent_.height));
        pc.tileGridSize = tileGridSize_;
        pc.sdfPxRange = sdfPxRange_;

        vkCmdPushConstants(cmd,
                           pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(Push),
                           &pc);

        const uint32_t groupX = ceilDiv(outputExtent_.width, 16);
        const uint32_t groupY = ceilDiv(outputExtent_.height, 16);
        vkCmdDispatch(cmd, groupX, groupY, 1);

        // Make shader writes visible
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

        // End in sampled layout like your ColorGrading pass
        ensureImageLayout(cmd,
                          outImages_[fi],
                          outLayouts_[fi],
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

private:
    struct Push
    {
        glm::ivec2 imageSize{0, 0};
        glm::ivec2 tileGridSize{0, 0};
        float sdfPxRange = 8.0f;
        float _pad[3] = {0, 0, 0}; // align to 16 bytes
    };

    struct FrameBuffers
    {
        // Glyph instances
        VkBuffer glyphBuf = VK_NULL_HANDLE;
        VkDeviceMemory glyphMem = VK_NULL_HANDLE;
        void* glyphMapped = nullptr;

        // Tile spans
        VkBuffer spanBuf = VK_NULL_HANDLE;
        VkDeviceMemory spanMem = VK_NULL_HANDLE;
        void* spanMapped = nullptr;

        // Tile glyph indices
        VkBuffer tileGlyphBuf = VK_NULL_HANDLE;
        VkDeviceMemory tileGlyphMem = VK_NULL_HANDLE;
        void* tileGlyphMapped = nullptr;

        uint32_t glyphCount = 0;
        uint32_t tileGlyphCount = 0;
        bool hasData = false;
    };

private:
    void createPipeline_()
    {
        // bindings: 0 outImage, 1 atlas, 2 spans, 3 tileGlyphs, 4 glyphs
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl.bindingCount = uint32_t(bindings.size());
        dsl.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(engine->logicalDevice, &dsl, nullptr, &setLayout_) != VK_SUCCESS)
            throw std::runtime_error("Text: failed to create descriptor set layout");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(Push);

        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;

        if (vkCreatePipelineLayout(engine->logicalDevice, &pli, nullptr, &pipelineLayout_) != VK_SUCCESS)
            throw std::runtime_error("Text: failed to create pipeline layout");

        auto shaderCode = readSPIRVFile("shaders/text_sdf.spv");
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
            throw std::runtime_error("Text: failed to create compute pipeline");
        }

        vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
    }

    void destroyPipeline_()
    {
        if (!engine || engine->logicalDevice == VK_NULL_HANDLE) return;

        if (pipeline_ != VK_NULL_HANDLE) { vkDestroyPipeline(engine->logicalDevice, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE; }
        if (pipelineLayout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(engine->logicalDevice, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE; }
        if (setLayout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(engine->logicalDevice, setLayout_, nullptr); setLayout_ = VK_NULL_HANDLE; }
    }

    void createOutputs_()
    {
        destroyOutputs_();

        if (!engine) return;
        if (outputFormat_ == VK_FORMAT_UNDEFINED || outputExtent_.width == 0 || outputExtent_.height == 0) return;

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
            ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            if (vkCreateImage(engine->logicalDevice, &ii, nullptr, &outImages_[i]) != VK_SUCCESS)
                throw std::runtime_error("Text: failed to create output image");

            VkMemoryRequirements mr{};
            vkGetImageMemoryRequirements(engine->logicalDevice, outImages_[i], &mr);

            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = engine->findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            if (vkAllocateMemory(engine->logicalDevice, &ai, nullptr, &outMem_[i]) != VK_SUCCESS)
                throw std::runtime_error("Text: failed to allocate output image memory");

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
                throw std::runtime_error("Text: failed to create output image view");

            outLayouts_[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        }
    }

    void destroyOutputs_()
    {
        if (!engine || engine->logicalDevice == VK_NULL_HANDLE) return;

        for (uint32_t i = 0; i < outImages_.size(); ++i)
        {
            destroyImageAndView(engine->logicalDevice, outImages_[i], outViews_[i], outMem_[i]);
            if (i < outLayouts_.size()) outLayouts_[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        outImages_.assign(framesInFlight_, VK_NULL_HANDLE);
        outMem_.assign(framesInFlight_, VK_NULL_HANDLE);
        outViews_.assign(framesInFlight_, VK_NULL_HANDLE);
        outLayouts_.assign(framesInFlight_, VK_IMAGE_LAYOUT_UNDEFINED);
    }

    void createFrameBuffers_()
    {
        // frame buffers depend on tile count
        destroyFrameBuffers_();

        if (!engine) return;
        if (outputExtent_.width == 0 || outputExtent_.height == 0) return;
        const uint32_t tileCount = uint32_t(tileGridSize_.x * tileGridSize_.y);
        if (tileCount == 0) return;

        const VkDeviceSize glyphBytes = VkDeviceSize(sizeof(GlyphInstance)) * VkDeviceSize(maxGlyphs_);
        const VkDeviceSize spanBytes  = VkDeviceSize(sizeof(TileSpan)) * VkDeviceSize(tileCount);
        const VkDeviceSize tileGlyphBytes = VkDeviceSize(sizeof(uint32_t)) * VkDeviceSize(maxTileGlyphRefs_);

        for (uint32_t i = 0; i < framesInFlight_; ++i)
        {
            // Use your Engine2D helper to create host-visible buffers.
            // Assumes signature:
            //   createBuffer(size, usage, memProps, outBuffer, outMemory)
            engine->createBuffer(glyphBytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 frame_[i].glyphBuf,
                                 frame_[i].glyphMem);
            vkMapMemory(engine->logicalDevice, frame_[i].glyphMem, 0, glyphBytes, 0, &frame_[i].glyphMapped);

            engine->createBuffer(spanBytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 frame_[i].spanBuf,
                                 frame_[i].spanMem);
            vkMapMemory(engine->logicalDevice, frame_[i].spanMem, 0, spanBytes, 0, &frame_[i].spanMapped);

            engine->createBuffer(tileGlyphBytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 frame_[i].tileGlyphBuf,
                                 frame_[i].tileGlyphMem);
            vkMapMemory(engine->logicalDevice, frame_[i].tileGlyphMem, 0, tileGlyphBytes, 0, &frame_[i].tileGlyphMapped);

            frame_[i].glyphCount = 0;
            frame_[i].tileGlyphCount = 0;
            frame_[i].hasData = false;
        }
    }

    void destroyFrameBuffers_()
    {
        if (!engine || engine->logicalDevice == VK_NULL_HANDLE) return;

        for (auto& f : frame_)
        {
            destroyBuffer(engine->logicalDevice, f.glyphBuf, f.glyphMem, f.glyphMapped);
            destroyBuffer(engine->logicalDevice, f.spanBuf, f.spanMem, f.spanMapped);
            destroyBuffer(engine->logicalDevice, f.tileGlyphBuf, f.tileGlyphMem, f.tileGlyphMapped);
            f.glyphCount = 0;
            f.tileGlyphCount = 0;
            f.hasData = false;
        }
    }

    void createDescriptors_()
    {
        destroyDescriptors_();

        if (!engine || setLayout_ == VK_NULL_HANDLE) return;

        descriptorSets_.assign(framesInFlight_, VK_NULL_HANDLE);

        std::array<VkDescriptorPoolSize, 3> sizes{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[0].descriptorCount = framesInFlight_;
        sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_;
        sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // spans + tileGlyphs + glyphs per frame = 3 buffers per set
        sizes[2].descriptorCount = framesInFlight_ * 3;

        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.poolSizeCount = uint32_t(sizes.size());
        pi.pPoolSizes = sizes.data();
        pi.maxSets = framesInFlight_;

        if (vkCreateDescriptorPool(engine->logicalDevice, &pi, nullptr, &descriptorPool_) != VK_SUCCESS)
            throw std::runtime_error("Text: failed to create descriptor pool");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, setLayout_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts = layouts.data();

        if (vkAllocateDescriptorSets(engine->logicalDevice, &ai, descriptorSets_.data()) != VK_SUCCESS)
            throw std::runtime_error("Text: failed to allocate descriptor sets");
    }

    void destroyDescriptors_()
    {
        if (!engine || engine->logicalDevice == VK_NULL_HANDLE) return;

        if (descriptorPool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(engine->logicalDevice, descriptorPool_, nullptr);
            descriptorPool_ = VK_NULL_HANDLE;
        }
        descriptorSets_.assign(framesInFlight_, VK_NULL_HANDLE);
    }

    void rebuildDescriptorSets_()
    {
        if (!engine || descriptorPool_ == VK_NULL_HANDLE) return;
        if (descriptorSets_.size() != framesInFlight_) return;

        for (uint32_t i = 0; i < framesInFlight_; ++i)
        {
            if (outViews_.empty() || outViews_[i] == VK_NULL_HANDLE) return;
            if (descriptorSets_[i] == VK_NULL_HANDLE) return;
            if (frame_[i].glyphBuf == VK_NULL_HANDLE || frame_[i].spanBuf == VK_NULL_HANDLE || frame_[i].tileGlyphBuf == VK_NULL_HANDLE)
                return;
        }

        for (uint32_t i = 0; i < framesInFlight_; ++i)
        {
            std::array<VkWriteDescriptorSet, 5> writes{};

            VkDescriptorImageInfo outInfo{};
            outInfo.imageView = outViews_[i];
            outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = descriptorSets_[i];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[0].descriptorCount = 1;
            writes[0].pImageInfo = &outInfo;

            VkDescriptorImageInfo atlasInfo{};
            atlasInfo.imageView = atlasView_;
            atlasInfo.sampler = atlasSampler_;
            atlasInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = descriptorSets_[i];
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].descriptorCount = 1;
            writes[1].pImageInfo = &atlasInfo;

            VkDescriptorBufferInfo spanBI{};
            spanBI.buffer = frame_[i].spanBuf;
            spanBI.offset = 0;
            // NOTE: shader only reads tileCount entries; range can be VK_WHOLE_SIZE since buffers are dedicated.
            spanBI.range = VK_WHOLE_SIZE;

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = descriptorSets_[i];
            writes[2].dstBinding = 2;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2].descriptorCount = 1;
            writes[2].pBufferInfo = &spanBI;

            VkDescriptorBufferInfo tileGlyphBI{};
            tileGlyphBI.buffer = frame_[i].tileGlyphBuf;
            tileGlyphBI.offset = 0;
            tileGlyphBI.range = VK_WHOLE_SIZE;

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = descriptorSets_[i];
            writes[3].dstBinding = 3;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[3].descriptorCount = 1;
            writes[3].pBufferInfo = &tileGlyphBI;

            VkDescriptorBufferInfo glyphBI{};
            glyphBI.buffer = frame_[i].glyphBuf;
            glyphBI.offset = 0;
            glyphBI.range = VK_WHOLE_SIZE;

            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = descriptorSets_[i];
            writes[4].dstBinding = 4;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[4].descriptorCount = 1;
            writes[4].pBufferInfo = &glyphBI;

            vkUpdateDescriptorSets(engine->logicalDevice,
                                   uint32_t(writes.size()),
                                   writes.data(),
                                   0,
                                   nullptr);
        }
    }

private:
    Engine2D* engine = nullptr;
    uint32_t framesInFlight_ = 0;

    uint32_t maxGlyphs_ = 0;
    uint32_t maxTileGlyphRefs_ = 0;

    // Input atlas
    VkImageView atlasView_ = VK_NULL_HANDLE;
    VkSampler atlasSampler_ = VK_NULL_HANDLE;

    // Pipeline
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;

    // Outputs per frame
    std::vector<VkImage> outImages_;
    std::vector<VkDeviceMemory> outMem_;
    std::vector<VkImageView> outViews_;
    std::vector<VkImageLayout> outLayouts_;

    // Per-frame SSBOs
    std::vector<FrameBuffers> frame_;

    // Descriptors per frame
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

public:
    // Same “bounds” exposure style as your header showed (you said you're fine with it).
    // If you prefer stricter, move these to private with getters.
    VkExtent2D outputExtent_{0, 0};
    VkFormat outputFormat_ = VK_FORMAT_UNDEFINED;

private:
    glm::ivec2 tileGridSize_{0, 0};
    float sdfPxRange_ = 8.0f;
};
