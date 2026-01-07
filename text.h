// text.h
#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

class Engine2D;

// Text pass with the SAME BOUNDS as ColorGrading:
//
//   Text(engine, framesInFlight)
//   resize(extent, format)          -> defines/allocates outputs + per-frame buffers/descriptor sets
//   setAtlas(view, sampler)         -> sets borrowed input atlas resources
//   upload(frameIndex, ...)         -> uploads per-frame glyph/tile data
//   dispatch(cmd, frameIndex)       -> runs compute, writes into owned per-frame output
//   output(frameIndex) / outputLayout(frameIndex)
//
// Shader contract (set=0):
//   binding 0: storage image outImage          (per-frame output image view)
//   binding 1: combined sampler fontAtlas      (atlas view + sampler)
//   binding 2: SSBO TileSpans                 (per-frame)
//   binding 3: SSBO TileGlyphs                (per-frame)
//   binding 4: SSBO GlyphInstances            (per-frame)
//
// Push constants:
//   ivec2 imageSize
//   ivec2 tileGridSize
//   float sdfPxRange

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

    // std430-friendly CPU struct for the shader's GlyphInstance
    struct alignas(16) GlyphInstance
    {
        int32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0; // pixel AABB [x0,x1),[y0,y1)
        float u0 = 0, v0 = 0, u1 = 0, v1 = 0;   // atlas UV rect
        glm::vec4 textColor{1, 1, 1, 1};        // rgba
        glm::vec4 bgColor{0, 0, 0, 0};          // rgba (optional)
        uint32_t flags = 0;                     // bit0: bg enable, etc.
        uint32_t _pad0 = 0, _pad1 = 0, _pad2 = 0;
    };

    struct TileSpan
    {
        uint32_t start = 0;
        uint32_t count = 0;
    };

public:
    // framesInFlight typically equals swapchain image count.
    // maxGlyphs and maxTileGlyphRefs are per-frame capacities for uploads.
    Text(Engine2D* engine,
         uint32_t framesInFlight,
         uint32_t maxGlyphs = 4096,
         uint32_t maxTileGlyphRefs = 65536);
    ~Text();

    Text(const Text&) = delete;
    Text& operator=(const Text&) = delete;

    // Output spec (same pattern as ColorGrading::resize).
    // format must support STORAGE + SAMPLED usage (e.g. VK_FORMAT_R8G8B8A8_UNORM).
    void resize(VkExtent2D extent, VkFormat format);

    // Input atlas (borrowed).
    // Atlas should be in SHADER_READ_ONLY_OPTIMAL when sampled.
    void setAtlas(VkImageView atlasView, VkSampler atlasSampler);

    // Optional tuning constant for your SDF bake.
    void setSdfPxRange(float pxRange);

    // Upload per-frame data for the shader.
    // spans.size() MUST equal tileGridSize.x * tileGridSize.y for current output extent.
    // glyphs.size() must be <= maxGlyphs, tileGlyphs.size() <= maxTileGlyphRefs.
    bool upload(uint32_t frameIndex,
                const std::vector<GlyphInstance>& glyphs,
                const std::vector<TileSpan>& spans,
                const std::vector<uint32_t>& tileGlyphs);

    // Run the compute shader for a given frame slot.
    void dispatch(VkCommandBuffer cmd, uint32_t frameIndex);

    // Query output for a given frame slot.
    Output output(uint32_t frameIndex) const;

    // Layout after dispatch (tracked internally).
    VkImageLayout outputLayout(uint32_t frameIndex) const;

    // Same “bounds” exposure style you used in ColorGrading.
    VkExtent2D outputExtent_{0, 0};
    VkFormat outputFormat_ = VK_FORMAT_UNDEFINED;

private:
    void createPipeline_();
    void destroyPipeline_();

    void createOutputs_();
    void destroyOutputs_();

    void createFrameBuffers_();
    void destroyFrameBuffers_();

    void createDescriptors_();
    void destroyDescriptors_();
    void rebuildDescriptorSets_();

private:
    struct FrameBuffers
    {
        VkBuffer glyphBuf = VK_NULL_HANDLE;
        VkDeviceMemory glyphMem = VK_NULL_HANDLE;
        void* glyphMapped = nullptr;

        VkBuffer spanBuf = VK_NULL_HANDLE;
        VkDeviceMemory spanMem = VK_NULL_HANDLE;
        void* spanMapped = nullptr;

        VkBuffer tileGlyphBuf = VK_NULL_HANDLE;
        VkDeviceMemory tileGlyphMem = VK_NULL_HANDLE;
        void* tileGlyphMapped = nullptr;

        uint32_t glyphCount = 0;
        uint32_t tileGlyphCount = 0;
        bool hasData = false;
    };

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

    // Dispatch metadata
    glm::ivec2 tileGridSize_{0, 0};
    float sdfPxRange_ = 8.0f;
};
