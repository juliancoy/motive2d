// font.cpp
//
// Implements Font as a wrapper-pass around Text (same bounds as ColorGrading/Text):
//   - resize(extent, format) defines output spec and (re)creates Text outputs/buffers/sets
//   - loadFontFile() loads nofile.ttf via FreeType
//   - update(frameIndex) rasterizes missing glyphs, updates atlas, builds instances+tiles, calls Text::upload()
//   - dispatch(cmd, frameIndex) calls Text::dispatch()
//
// Notes / assumptions about your Engine2D:
//   - engine->logicalDevice (VkDevice) exists
//   - engine->physicalDevice (VkPhysicalDevice) exists (needed for memory types) OR engine->findMemoryType(bits, props) exists
//   - engine->graphicsQueue exists
//   - engine->getGraphicsQueueFamilyIndex() exists
//   - engine->createBuffer(size, usage, memProps, outBuf, outMem) exists
//   - engine->findMemoryType(typeBits, props) exists
//
// If your Engine2D names differ, search/replace those calls.
// This file intentionally owns its own tiny "immediate submit" command pool for atlas uploads.
//
// Atlas format:
//   - Uses VK_FORMAT_R8_UNORM (single-channel) and samples .r in shader.
//   - If you want MSDF, you’d switch to VK_FORMAT_R8G8B8A8_UNORM and change the shader sampling.
//
// Shader expectation:
//   - The Text pass uses shaders/text_sdf.spv and expects sampler2D fontAtlas.
//   - SDF/MSDF decoding is done in shader; here we’re uploading a plain grayscale coverage bitmap.
//     That will still render, but edges will not be scale-perfect like true SDF.
//     (To get true SDF, you must generate SDF bitmaps, not FT_LOAD_RENDER coverage.)

#include "font.h"

#include "engine2d.h"
#include "text.h"
#include "utils.h"
#include "debug_logging.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ft2build.h>
#include FT_FREETYPE_H

namespace
{
static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// --- font locate helper (mirrors your earlier style) ---
static std::filesystem::path locateFontFile(const std::filesystem::path& preferred)
{
    std::vector<std::filesystem::path> candidates;
    if (!preferred.empty())
        candidates.push_back(preferred);

    // project-local “nofile.ttf”
    candidates.push_back("nofile.ttf");
    candidates.push_back("../nofile.ttf");
    candidates.push_back("../../nofile.ttf");

    // common Linux fonts (fallback)
    candidates.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    candidates.push_back("/usr/share/fonts/dejavu/DejaVuSans.ttf");

    // common Windows fallback
    candidates.push_back("C:/Windows/Fonts/Arial.ttf");

    // relative font folders
    candidates.push_back("fonts/DejaVuSans.ttf");
    candidates.push_back("../fonts/DejaVuSans.ttf");
    candidates.push_back("../../fonts/DejaVuSans.ttf");

    for (const auto& c : candidates)
    {
        std::error_code ec;
        if (!c.empty() && std::filesystem::exists(c, ec))
            return std::filesystem::absolute(c, ec);
    }
    return {};
}

// --- UTF-8 to codepoints (minimal, permissive) ---
static std::vector<uint32_t> utf8ToCodepoints(const std::string& s)
{
    std::vector<uint32_t> cps;
    cps.reserve(s.size());

    const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
    size_t i = 0, n = s.size();

    while (i < n)
    {
        uint8_t c = p[i++];

        if (c < 0x80)
        {
            cps.push_back(c);
            continue;
        }

        // 2-byte
        if ((c & 0xE0) == 0xC0 && i < n)
        {
            uint8_t c2 = p[i++];
            cps.push_back(((uint32_t)(c & 0x1F) << 6) | (uint32_t)(c2 & 0x3F));
            continue;
        }

        // 3-byte
        if ((c & 0xF0) == 0xE0 && i + 1 < n)
        {
            uint8_t c2 = p[i++];
            uint8_t c3 = p[i++];
            cps.push_back(((uint32_t)(c & 0x0F) << 12) |
                          ((uint32_t)(c2 & 0x3F) << 6) |
                          (uint32_t)(c3 & 0x3F));
            continue;
        }

        // 4-byte
        if ((c & 0xF8) == 0xF0 && i + 2 < n)
        {
            uint8_t c2 = p[i++];
            uint8_t c3 = p[i++];
            uint8_t c4 = p[i++];
            cps.push_back(((uint32_t)(c & 0x07) << 18) |
                          ((uint32_t)(c2 & 0x3F) << 12) |
                          ((uint32_t)(c3 & 0x3F) << 6) |
                          (uint32_t)(c4 & 0x3F));
            continue;
        }

        // invalid byte sequence – skip
        cps.push_back((uint32_t)'?');
    }

    return cps;
}

// --- Vulkan helpers for atlas ---
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
                                             VkAccessFlags dstAccess,
                                             VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = aspectMask;
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    return b;
}

static void cmdTransitionImage(VkCommandBuffer cmd,
                               VkImage img,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout,
                               VkAccessFlags srcAccess,
                               VkAccessFlags dstAccess,
                               VkPipelineStageFlags srcStage,
                               VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier b = makeImageBarrier(img, oldLayout, newLayout, srcAccess, dstAccess);
    vkCmdPipelineBarrier(cmd,
                         srcStage, dstStage,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &b);
}

static void destroyBuffer(VkDevice device, VkBuffer& buf, VkDeviceMemory& mem, void*& mapped)
{
    if (mapped)
    {
        vkUnmapMemory(device, mem);
        mapped = nullptr;
    }
    if (buf != VK_NULL_HANDLE) { vkDestroyBuffer(device, buf, nullptr); buf = VK_NULL_HANDLE; }
    if (mem != VK_NULL_HANDLE) { vkFreeMemory(device, mem, nullptr); mem = VK_NULL_HANDLE; }
}

// Key hash for glyph cache
struct GlyphKeyHash
{
    size_t operator()(const Font::GlyphKey& k) const noexcept
    {
        // simple mix
        return (size_t(k.codepoint) * 1315423911u) ^ (size_t(k.px) + 0x9e3779b97f4a7c15ull);
    }
};

struct GlyphKeyEq
{
    bool operator()(const Font::GlyphKey& a, const Font::GlyphKey& b) const noexcept
    {
        return a.codepoint == b.codepoint && a.px == b.px;
    }
};

} // namespace

// -------------------- Font private state (glyph map) --------------------
struct FontGlyphMap
{
    std::unordered_map<Font::GlyphKey, Font::GlyphEntry, GlyphKeyHash, GlyphKeyEq> map;

    // very simple shelf packer state
    uint32_t cursorX = 1;
    uint32_t cursorY = 1;
    uint32_t rowH = 0;
};

// -------------------- Vulkan upload state --------------------
struct FontUploadContext
{
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    void* stagingMapped = nullptr;
    VkDeviceSize stagingSize = 0;

    VkImageLayout atlasLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

// -------------------- Font implementation --------------------

Font::Font(Engine2D* engine, uint32_t framesInFlight)
    : engine_(engine)
    , framesInFlight_(framesInFlight)
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("Font requires a valid Engine2D");
    if (framesInFlight_ == 0)
        throw std::runtime_error("Font: framesInFlight must be > 0");

    text_ = new Text(engine_, framesInFlight_);
    frame_.resize(framesInFlight_);

    glyphMap_ = new FontGlyphMap();

    // Create atlas GPU objects now (will be allocated on first use)
    // Create a sampler now (view/image later)
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxAnisotropy = 1.0f;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;

    if (vkCreateSampler(engine_->logicalDevice, &si, nullptr, &atlasSampler_) != VK_SUCCESS)
        throw std::runtime_error("Font: failed to create atlas sampler");

    // Create upload context (small immediate submit)
    auto* up = new FontUploadContext();
    // command pool
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = engine_->getGraphicsQueueFamilyIndex();
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(engine_->logicalDevice, &pci, nullptr, &up->pool) != VK_SUCCESS)
        throw std::runtime_error("Font: failed to create upload command pool");

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = up->pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(engine_->logicalDevice, &cai, &up->cmd) != VK_SUCCESS)
        throw std::runtime_error("Font: failed to alloc upload command buffer");

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(engine_->logicalDevice, &fci, nullptr, &up->fence) != VK_SUCCESS)
        throw std::runtime_error("Font: failed to create upload fence");

    // store in pointer slot we can reach via atlasDirty_ (we’ll just reinterpret-cast)
    // (keeps header simple)
    // We'll stash it in glyphMap_ map pointer by abusing nullptr? No—keep as member by using atlasDirty_? not ok.
    // Instead: store it in ftLibrary_ when ft not initialized? No.
    // We'll store it via glyphMap_ as a sidecar: simplest is to keep a static map from this pointer, but no.
    // We'll instead use ftLibrary_ pointer slot once initialized? Also no.
    //
    // Since header does not provide a field, we’ll keep it as a function-static map keyed by this.
    //
    // To avoid that mess, we embed upload ctx into glyphMap_ object using a trailing pointer.
    // (Still safe and private to this TU.)
    // We'll just keep a global side-table:
    //
    // (This is purely internal; if you dislike it, add a `void* uploadCtx_` member to the header.)
    static std::unordered_map<const Font*, FontUploadContext*> gUpload;
    gUpload[this] = up;
}

Font::~Font()
{
    if (!engine_ || engine_->logicalDevice == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(engine_->logicalDevice);

    // Free FreeType
    if (ftFace_)
    {
        FT_Done_Face(reinterpret_cast<FT_Face>(ftFace_));
        ftFace_ = nullptr;
    }
    if (ftLibrary_)
    {
        FT_Done_FreeType(reinterpret_cast<FT_Library>(ftLibrary_));
        ftLibrary_ = nullptr;
    }

    // Destroy atlas
    if (atlasSampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(engine_->logicalDevice, atlasSampler_, nullptr);
        atlasSampler_ = VK_NULL_HANDLE;
    }
    destroyImageAndView(engine_->logicalDevice, atlasImage_, atlasView_, atlasMem_);

    // Destroy upload context
    static std::unordered_map<const Font*, FontUploadContext*> gUpload;
    auto it = gUpload.find(this);
    if (it != gUpload.end())
    {
        FontUploadContext* up = it->second;
        if (up)
        {
            if (up->fence) vkDestroyFence(engine_->logicalDevice, up->fence, nullptr);
            if (up->cmd) vkFreeCommandBuffers(engine_->logicalDevice, up->pool, 1, &up->cmd);
            if (up->pool) vkDestroyCommandPool(engine_->logicalDevice, up->pool, nullptr);
            destroyBuffer(engine_->logicalDevice, up->staging, up->stagingMem, up->stagingMapped);
            delete up;
        }
        gUpload.erase(it);
    }

    // Destroy glyph map
    if (glyphMap_)
    {
        delete reinterpret_cast<FontGlyphMap*>(glyphMap_);
        glyphMap_ = nullptr;
    }

    // Destroy Text
    delete text_;
    text_ = nullptr;

    engine_ = nullptr;
}

void Font::resize(VkExtent2D extent, VkFormat format)
{
    outputExtent_ = extent;
    outputFormat_ = format;

    if (text_)
    {
        text_->resize(extent, format);
        text_->outputExtent_ = extent;
        text_->outputFormat_ = format;
    }
}

bool Font::ensureFreeType_()
{
    if (ftInitialized_)
        return true;

    FT_Library lib = nullptr;
    if (FT_Init_FreeType(&lib) != 0)
    {
        LOG_DEBUG(std::cout << "[Font] FT_Init_FreeType failed\n");
        return false;
    }

    ftLibrary_ = lib;
    ftInitialized_ = true;
    return true;
}

bool Font::ensureFace_()
{
    if (ftFace_)
        return true;

    if (!ensureFreeType_())
        return false;

    fontPath_ = locateFontFile(fontPath_);
    if (fontPath_.empty())
    {
        LOG_DEBUG(std::cout << "[Font] No font file found\n");
        return false;
    }

    FT_Face face = nullptr;
    if (FT_New_Face(reinterpret_cast<FT_Library>(ftLibrary_), fontPath_.string().c_str(), 0, &face) != 0)
    {
        LOG_DEBUG(std::cout << "[Font] FT_New_Face failed for " << fontPath_ << "\n");
        return false;
    }

    ftFace_ = face;

    // default pixel size
    FT_Set_Pixel_Sizes(face, 0, std::max<uint32_t>(pixelHeight_, 1));

    return true;
}

bool Font::loadFontFile(const std::filesystem::path& ttfPath)
{
    fontPath_ = ttfPath;
    // reset face so it reloads
    if (ftFace_)
    {
        FT_Done_Face(reinterpret_cast<FT_Face>(ftFace_));
        ftFace_ = nullptr;
    }

    bool ok = ensureFace_();
    if (ok)
    {
        LOG_DEBUG(std::cout << "[Font] Loaded font: " << fontPath_ << "\n");
    }
    return ok;
}

void Font::setPixelHeight(uint32_t px)
{
    pixelHeight_ = std::max<uint32_t>(1, px);
    if (ftFace_)
    {
        FT_Set_Pixel_Sizes(reinterpret_cast<FT_Face>(ftFace_), 0, pixelHeight_);
    }
    clearCache();
}

void Font::setSdfPxRange(float pxRange)
{
    sdfPxRange_ = std::max(0.0001f, pxRange);
    if (text_) text_->setSdfPxRange(sdfPxRange_);
}

void Font::setText(std::string textUtf8)
{
    textUtf8_ = std::move(textUtf8);
    lines_.clear();

    // Split on '\n'
    size_t start = 0;
    while (start <= textUtf8_.size())
    {
        size_t end = textUtf8_.find('\n', start);
        if (end == std::string::npos)
            end = textUtf8_.size();
        lines_.push_back(textUtf8_.substr(start, end - start));
        if (end == textUtf8_.size())
            break;
        start = end + 1;
    }
}

void Font::setLines(std::vector<std::string> lines)
{
    lines_ = std::move(lines);
    textUtf8_.clear();
}

void Font::setStyle(const Style& s)
{
    style_ = s;
}

void Font::clearCache()
{
    if (glyphMap_)
    {
        auto* gm = reinterpret_cast<FontGlyphMap*>(glyphMap_);
        gm->map.clear();
        gm->cursorX = 1;
        gm->cursorY = 1;
        gm->rowH = 0;
    }

    // Mark atlas dirty and also clear it to 0 (we'll reupload on next update)
    atlasDirty_ = true;
}

static void ensureStagingCapacity(Engine2D* engine, FontUploadContext* up, VkDeviceSize bytes)
{
    if (up->staging != VK_NULL_HANDLE && up->stagingSize >= bytes)
        return;

    destroyBuffer(engine->logicalDevice, up->staging, up->stagingMem, up->stagingMapped);

    engine->createBuffer(bytes,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         up->staging,
                         up->stagingMem);

    up->stagingSize = bytes;
    vkMapMemory(engine->logicalDevice, up->stagingMem, 0, bytes, 0, &up->stagingMapped);
}

void Font::rebuildAtlasIfNeeded_()
{
    if (atlasImage_ != VK_NULL_HANDLE && atlasView_ != VK_NULL_HANDLE)
        return;

    if (!engine_) return;

    // Create atlas image
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = atlasFormat_;
    ii.extent = VkExtent3D{atlasWidth_, atlasHeight_, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(engine_->logicalDevice, &ii, nullptr, &atlasImage_) != VK_SUCCESS)
        throw std::runtime_error("Font: failed to create atlas image");

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(engine_->logicalDevice, atlasImage_, &mr);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = engine_->findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(engine_->logicalDevice, &ai, nullptr, &atlasMem_) != VK_SUCCESS)
        throw std::runtime_error("Font: failed to allocate atlas memory");

    vkBindImageMemory(engine_->logicalDevice, atlasImage_, atlasMem_, 0);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = atlasImage_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = atlasFormat_;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;

    if (vkCreateImageView(engine_->logicalDevice, &vi, nullptr, &atlasView_) != VK_SUCCESS)
        throw std::runtime_error("Font: failed to create atlas view");

    // Inform Text about the atlas now (descriptor rebuild will happen there)
    if (text_)
        text_->setAtlas(atlasView_, atlasSampler_);

    atlasDirty_ = true;
}

void Font::ensureAtlasUploaded_()
{
    if (!atlasDirty_) return;
    rebuildAtlasIfNeeded_();
    if (atlasImage_ == VK_NULL_HANDLE) return;

    // Upload a full clear to 0 on first dirty (simple approach).
    // For incremental updates, you’d upload only dirty rectangles; start with full upload.
    static std::unordered_map<const Font*, FontUploadContext*> gUpload;
    FontUploadContext* up = gUpload[this];
    if (!up) return;

    const VkDeviceSize bytes = VkDeviceSize(atlasWidth_) * VkDeviceSize(atlasHeight_); // R8
    ensureStagingCapacity(engine_, up, bytes);

    std::memset(up->stagingMapped, 0, size_t(bytes));

    // record + submit
    vkResetFences(engine_->logicalDevice, 1, &up->fence);
    vkResetCommandBuffer(up->cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(up->cmd, &bi);

    cmdTransitionImage(up->cmd,
                       atlasImage_,
                       up->atlasLayout,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       0,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.bufferRowLength = 0;   // tightly packed
    copy.bufferImageHeight = 0;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = VkOffset3D{0, 0, 0};
    copy.imageExtent = VkExtent3D{atlasWidth_, atlasHeight_, 1};

    vkCmdCopyBufferToImage(up->cmd,
                           up->staging,
                           atlasImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copy);

    cmdTransitionImage(up->cmd,
                       atlasImage_,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    vkEndCommandBuffer(up->cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &up->cmd;

    vkQueueSubmit(engine_->graphicsQueue, 1, &si, up->fence);
    vkWaitForFences(engine_->logicalDevice, 1, &up->fence, VK_TRUE, UINT64_MAX);

    up->atlasLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    atlasDirty_ = false;
}

// Rasterize a glyph and place it into atlas; uploads only that glyph region (incremental).
// Returns cached GlyphEntry on success.
bool Font::rasterizeAndCacheGlyph(uint32_t codepoint, Font::GlyphEntry& out)
{
    if (!ensureFace_()) return false;

    rebuildAtlasIfNeeded_();
    if (atlasImage_ == VK_NULL_HANDLE) return false;

    auto* gm = reinterpret_cast<FontGlyphMap*>(glyphMap_);
    FT_Face face = reinterpret_cast<FT_Face>(ftFace_);

    FT_Set_Pixel_Sizes(face, 0, std::max<uint32_t>(pixelHeight_, 1));

    if (FT_Load_Char(face, codepoint, FT_LOAD_RENDER) != 0)
        return false;

    FT_GlyphSlot slot = face->glyph;
    const FT_Bitmap& bmp = slot->bitmap;

    const uint32_t gw = bmp.width;
    const uint32_t gh = bmp.rows;

    // Handle space / empty glyphs
    if (gw == 0 || gh == 0)
    {
        out.x = out.y = out.w = out.h = 0;
        out.bearingX = int16_t(slot->bitmap_left);
        out.bearingY = int16_t(slot->bitmap_top);
        out.advanceX = int16_t(slot->advance.x >> 6);
        return true;
    }

    // Simple shelf packing with 1px padding
    const uint32_t pad = 1;
    const uint32_t needW = gw + pad;
    const uint32_t needH = gh + pad;

    if (gm->cursorX + needW >= atlasWidth_)
    {
        gm->cursorX = 1;
        gm->cursorY += gm->rowH + 1;
        gm->rowH = 0;
    }
    if (gm->cursorY + needH >= atlasHeight_)
    {
        // Atlas full; simplest policy: clear cache + atlas and restart.
        // (You can implement multi-page atlases later.)
        LOG_DEBUG(std::cout << "[Font] Atlas full; clearing cache\n");
        clearCache();
        ensureAtlasUploaded_(); // clears atlas image to zero
        gm = reinterpret_cast<FontGlyphMap*>(glyphMap_);
        // retry pack at start
        gm->cursorX = 1;
        gm->cursorY = 1;
        gm->rowH = 0;

        if (gm->cursorX + needW >= atlasWidth_ || gm->cursorY + needH >= atlasHeight_)
            return false;
    }

    const uint32_t ax = gm->cursorX;
    const uint32_t ay = gm->cursorY;

    gm->cursorX += needW;
    gm->rowH = std::max(gm->rowH, needH);

    out.x = uint16_t(ax);
    out.y = uint16_t(ay);
    out.w = uint16_t(gw);
    out.h = uint16_t(gh);
    out.bearingX = int16_t(slot->bitmap_left);
    out.bearingY = int16_t(slot->bitmap_top);
    out.advanceX = int16_t(slot->advance.x >> 6);

    // Upload glyph bitmap into atlas as R8; FreeType bitmap.buffer is 8-bit coverage.
    static std::unordered_map<const Font*, FontUploadContext*> gUpload;
    FontUploadContext* up = gUpload[self];
    if (!up) return false;

    // Staging: tightly packed gw*gh bytes.
    const VkDeviceSize bytes = VkDeviceSize(gw) * VkDeviceSize(gh);
    ensureStagingCapacity(engine_, up, bytes);

    // Copy rows (bmp.pitch may differ)
    uint8_t* dst = reinterpret_cast<uint8_t*>(up->stagingMapped);
    for (uint32_t row = 0; row < gh; ++row)
    {
        const uint8_t* srcRow = bmp.buffer + row * bmp.pitch;
        std::memcpy(dst + row * gw, srcRow, gw);
    }

    vkResetFences(engine_->logicalDevice, 1, &up->fence);
    vkResetCommandBuffer(up->cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(up->cmd, &bi);

    // transition atlas to transfer dst
    cmdTransitionImage(up->cmd,
                       atlasImage_,
                       up->atlasLayout,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       (up->atlasLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ? VK_ACCESS_SHADER_READ_BIT : 0,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.bufferRowLength = 0;
    copy.bufferImageHeight = 0;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = VkOffset3D{int32_t(ax), int32_t(ay), 0};
    copy.imageExtent = VkExtent3D{gw, gh, 1};

    vkCmdCopyBufferToImage(up->cmd,
                           up->staging,
                           atlasImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copy);

    // back to shader read
    cmdTransitionImage(up->cmd,
                       atlasImage_,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    vkEndCommandBuffer(up->cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &up->cmd;

    vkQueueSubmit(engine_->graphicsQueue, 1, &si, up->fence);
    vkWaitForFences(engine_->logicalDevice, 1, &up->fence, VK_TRUE, UINT64_MAX);

    up->atlasLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    return true;
}

void Font::buildInstancesAndTiles_(uint32_t frameIndex)
{
    if (!text_) return;
    if (outputExtent_.width == 0 || outputExtent_.height == 0) return;

    const uint32_t fi = frameIndex % framesInFlight_;

    // Choose lines: if setLines used, use that; else split from textUtf8_ already done by setText
    const std::vector<std::string>& lines = lines_;

    // If no text, make empty upload
    if (lines.empty())
    {
        std::vector<Text::GlyphInstance> glyphs;
        std::vector<Text::TileSpan> spans(uint32_t(ceilDiv(outputExtent_.width, 16) * ceilDiv(outputExtent_.height, 16)));
        std::vector<uint32_t> indices;
        text_->upload(frameIndex, glyphs, spans, indices);
        return;
    }

    if (!ensureFace_()) return;
    rebuildAtlasIfNeeded_();

    auto* gm = reinterpret_cast<FontGlyphMap*>(glyphMap_);
    FT_Face face = reinterpret_cast<FT_Face>(ftFace_);
    FT_Set_Pixel_Sizes(face, 0, std::max<uint32_t>(pixelHeight_, 1));

    const int ascent  = int(face->size->metrics.ascender >> 6);
    const int descent = int(-(face->size->metrics.descender >> 6)); // make positive
    const int lineH   = std::max<int>(1, ascent + descent);

    const uint32_t tileW = ceilDiv(outputExtent_.width, 16);
    const uint32_t tileH = ceilDiv(outputExtent_.height, 16);
    const uint32_t tileCount = tileW * tileH;

    // Build glyph instances
    std::vector<Text::GlyphInstance> glyphInstances;
    glyphInstances.reserve(256);

    // For tile binning: one vector per tile
    std::vector<std::vector<uint32_t>> tileLists(tileCount);

    const int originX = style_.originPx.x;
    const int originY = style_.originPx.y;

    int penY = originY;

    for (size_t li = 0; li < lines.size(); ++li)
    {
        const std::string& line = lines[li];
        std::vector<uint32_t> cps = utf8ToCodepoints(line);

        int penX = originX;
        const int baselineY = penY + ascent;

        for (uint32_t cp : cps)
        {
            // handle tab as spaces
            if (cp == '\t') cp = ' ';

            GlyphKey key{cp, pixelHeight_};
            GlyphEntry ge{};

            auto it = gm->map.find(key);
            if (it == gm->map.end())
            {
                if (!rasterizeAndCacheGlyph(this, cp, ge))
                    continue;
                gm->map.emplace(key, ge);
            }
            else
            {
                ge = it->second;
            }

            // advance-only glyph (space)
            if (ge.w == 0 || ge.h == 0)
            {
                penX += ge.advanceX;
                continue;
            }

            const int x0 = penX + ge.bearingX;
            const int y0 = baselineY - ge.bearingY;
            const int x1 = x0 + int(ge.w);
            const int y1 = y0 + int(ge.h);

            // clip early if fully off-screen (optional)
            if (x1 <= 0 || y1 <= 0 || x0 >= int(outputExtent_.width) || y0 >= int(outputExtent_.height))
            {
                penX += ge.advanceX;
                continue;
            }

            Text::GlyphInstance inst{};
            inst.x0 = x0; inst.y0 = y0; inst.x1 = x1; inst.y1 = y1;

            const float invW = 1.0f / float(atlasWidth_);
            const float invH = 1.0f / float(atlasHeight_);

            inst.u0 = float(ge.x) * invW;
            inst.v0 = float(ge.y) * invH;
            inst.u1 = float(ge.x + ge.w) * invW;
            inst.v1 = float(ge.y + ge.h) * invH;

            inst.textColor = style_.textColor;
            inst.bgColor = style_.backgroundColor;
            inst.flags = (style_.enableBackground ? 1u : 0u);

            const uint32_t glyphIndex = uint32_t(glyphInstances.size());
            glyphInstances.push_back(inst);

            // Bin into tiles
            const int tx0 = std::max(0, x0) / 16;
            const int ty0 = std::max(0, y0) / 16;
            const int tx1 = std::min(int(outputExtent_.width - 1), x1 - 1) / 16;
            const int ty1 = std::min(int(outputExtent_.height - 1), y1 - 1) / 16;

            for (int ty = ty0; ty <= ty1; ++ty)
            {
                for (int tx = tx0; tx <= tx1; ++tx)
                {
                    const uint32_t t = uint32_t(ty) * tileW + uint32_t(tx);
                    if (t < tileCount)
                        tileLists[t].push_back(glyphIndex);
                }
            }

            penX += ge.advanceX;
        }

        penY += int(float(lineH) + style_.lineSpacingPx);
    }

    // Flatten tile lists -> spans + indices
    std::vector<Text::TileSpan> spans(tileCount);
    std::vector<uint32_t> tileGlyphIndices;
    tileGlyphIndices.reserve(glyphInstances.size() * 2);

    uint32_t cursor = 0;
    for (uint32_t t = 0; t < tileCount; ++t)
    {
        spans[t].start = cursor;
        spans[t].count = uint32_t(tileLists[t].size());
        if (!tileLists[t].empty())
        {
            tileGlyphIndices.insert(tileGlyphIndices.end(), tileLists[t].begin(), tileLists[t].end());
            cursor += uint32_t(tileLists[t].size());
        }
    }

    // Upload via Text
    text_->upload(frameIndex, glyphInstances, spans, tileGlyphIndices);
}

void Font::update(uint32_t frameIndex)
{
    // Ensure atlas exists and is cleared at least once
    rebuildAtlasIfNeeded_();
    ensureAtlasUploaded_();

    // Build instances + tiles and upload to Text
    buildInstancesAndTiles_(frameIndex);
}

void Font::dispatch(VkCommandBuffer cmd, uint32_t frameIndex)
{
    if (!text_) return;

    // Make sure Text has atlas
    if (atlasView_ != VK_NULL_HANDLE && atlasSampler_ != VK_NULL_HANDLE)
        text_->setAtlas(atlasView_, atlasSampler_);

    // Keep sdfPxRange in sync
    text_->setSdfPxRange(sdfPxRange_);

    text_->dispatch(cmd, frameIndex);
}

Font::Output Font::output(uint32_t frameIndex) const
{
    Font::Output o{};
    if (!text_) return o;
    auto to = text_->output(frameIndex);
    o.image = to.image;
    o.view = to.view;
    o.layout = to.layout;
    o.extent = to.extent;
    o.format = to.format;
    return o;
}

VkImageLayout Font::outputLayout(uint32_t frameIndex) const
{
    if (!text_) return VK_IMAGE_LAYOUT_UNDEFINED;
    return text_->outputLayout(frameIndex);
}
