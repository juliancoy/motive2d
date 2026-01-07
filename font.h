// font.h
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

class Engine2D;
class Text; // from text.h (the pass we wrapped)

class Font
{
public:

    // ---- glyph cache + packing state (opaque containers; implemented in font.cpp) ----
    struct GlyphKey { uint32_t codepoint; uint32_t px; };
    
    struct Output
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkExtent2D extent{0, 0};
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    struct Style
    {
        glm::vec4 textColor{1, 1, 1, 1};
        glm::vec4 backgroundColor{0, 0, 0, 0};     // if a>0, shader can blend background
        glm::ivec2 originPx{0, 0};                 // top-left origin in output pixel space
        float lineSpacingPx = 4.0f;                // extra spacing between lines
        bool enableBackground = false;             // per-glyph quad background (flag bit0)
    };

    struct GlyphEntry
    {
        // packing location in atlas (pixels)
        uint16_t x = 0, y = 0, w = 0, h = 0;
        // FreeType metrics (pixels)
        int16_t bearingX = 0;
        int16_t bearingY = 0;
        int16_t advanceX = 0;
    };


    // Mirrors ColorGrading/Text bounds:
    // - owns outputs (via internal Text pass)
    // - caller controls output spec via resize()
    // - caller drives per-frame dispatch(cmd, frameIndex)
    Font(Engine2D* engine, uint32_t framesInFlight);
    ~Font();

    Font(const Font&) = delete;
    Font& operator=(const Font&) = delete;

    // Call on swapchain recreate / output spec change.
    // format must be a storage+sampled RGBA format (e.g. VK_FORMAT_R8G8B8A8_UNORM).
    void resize(VkExtent2D extent, VkFormat format);

    // Load a .ttf; if empty, implementation may try to locate nofile.ttf (project-local/system).
    // Returns false on failure; class remains usable (may fall back to a dummy atlas if implemented).
    bool loadFontFile(const std::filesystem::path& ttfPath = {});

    // Set pixel height for glyph rasterization (FreeType pixel size).
    void setPixelHeight(uint32_t px);

    // SDF tuning passed through to Text shader (if your atlas is SDF; for bitmap atlas you can still set it, but it won't matter much).
    void setSdfPxRange(float pxRange);

    // Text content. Newlines split into lines. If you want explicit line list, use setLines().
    void setText(std::string textUtf8);

    // Provide explicit lines (already split), e.g. for subtitles.
    void setLines(std::vector<std::string> lines);

    // Style controls
    void setStyle(const Style& s);
    const Style& style() const { return style_; }

    // Optional: clear caches/atlas (e.g. when changing font size drastically).
    void clearCache();

    // Per-frame: ensure atlas has required glyphs, build glyph instances + tile bins, upload to Text.
    // Safe to call multiple times; does CPU work and host buffer writes, no command recording.
    void update(uint32_t frameIndex);

    // Per-frame: record compute dispatch that draws into this pass output (via internal Text).
    void dispatch(VkCommandBuffer cmd, uint32_t frameIndex);

    // Output access mirrors ColorGrading/Text.
    Output output(uint32_t frameIndex) const;
    VkImageLayout outputLayout(uint32_t frameIndex) const;

    // Same “bounds exposure” style as your passes (kept public).
    VkExtent2D outputExtent_{0, 0};
    VkFormat outputFormat_ = VK_FORMAT_UNDEFINED;

private:
    // ---- internal helpers (implemented in font.cpp) ----
    bool ensureFreeType_();
    bool ensureFace_();
    void rebuildAtlasIfNeeded_();
    void ensureAtlasUploaded_();              // upload any dirty regions to GPU
    void buildInstancesAndTiles_(uint32_t frameIndex);
    bool rasterizeAndCacheGlyph(uint32_t codepoint, GlyphEntry& out);

private:
    Engine2D* engine_ = nullptr;
    uint32_t framesInFlight_ = 0;

    // Wrapped renderer (owns output images/views/layout tracking)
    Text* text_ = nullptr;

    // ---- font configuration ----
    std::filesystem::path fontPath_;
    uint32_t pixelHeight_ = 32;
    float sdfPxRange_ = 8.0f;

    Style style_{};
    std::string textUtf8_;
    std::vector<std::string> lines_;

    // ---- FreeType objects (opaque here; defined/used in font.cpp) ----
    void* ftLibrary_ = nullptr;  // FT_Library
    void* ftFace_ = nullptr;     // FT_Face
    bool ftInitialized_ = false;

    // ---- atlas resources (opaque handles; owned by Font) ----
    // For implementation: typically a single VK_FORMAT_R8_UNORM or VK_FORMAT_R8G8B8A8_UNORM sampled image + view + sampler.
    VkImage atlasImage_ = VK_NULL_HANDLE;
    VkDeviceMemory atlasMem_ = VK_NULL_HANDLE;
    VkImageView atlasView_ = VK_NULL_HANDLE;
    VkSampler atlasSampler_ = VK_NULL_HANDLE;
    VkFormat atlasFormat_ = VK_FORMAT_R8_UNORM;

    uint32_t atlasWidth_ = 1024;
    uint32_t atlasHeight_ = 1024;
    bool atlasDirty_ = false;

    // We avoid including <unordered_map> in the header unless you want it; implementation can hold the real map.
    void* glyphMap_ = nullptr; // pointer to an internal map<GlyphKey,GlyphEntry> stored/managed in font.cpp

    // ---- per-frame draw data fed into Text ----
    // Stored here so update(frameIndex) can rebuild and upload via Text::upload.
    struct FrameData
    {
        // These types come from Text; forward-declared here to keep header light.
        // In font.cpp you will include "text.h" and use Text::GlyphInstance and Text::TileSpan directly.
        std::vector<uint8_t> glyphInstancesBytes; // packed array of Text::GlyphInstance
        std::vector<uint8_t> tileSpansBytes;      // packed array of Text::TileSpan
        std::vector<uint32_t> tileGlyphIndices;   // uint32 glyph indices

        uint32_t glyphCount = 0;
        bool prepared = false;
    };

    std::vector<FrameData> frame_;
};
