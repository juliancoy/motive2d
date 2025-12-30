#include "fonts.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>
#include <vector>

#include <ft2build.h>
#include FT_FREETYPE_H

namespace fonts
{
namespace
{
FT_Library gLibrary = nullptr;
FT_Face gFace = nullptr;
bool gInitialized = false;
bool gTriedInit = false;

constexpr uint32_t kMinimumFontSize = 12;

std::filesystem::path locateFont()
{
    // Expanded list of candidate fonts and paths
    const std::vector<std::filesystem::path> candidates = {
        // Common system font paths
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        // Relative paths from executable
        "fonts/DejaVuSans.ttf",
        "../fonts/DejaVuSans.ttf",
        "../../fonts/DejaVuSans.ttf",
        // Placeholder
        "nofile.ttf",
        "../nofile.ttf",
        "../../nofile.ttf",
    };

    for (const auto &candidate : candidates)
    {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec))
        {
            std::cout << "[Fonts] Found font at: " << std::filesystem::absolute(candidate, ec) << std::endl;
            return std::filesystem::absolute(candidate, ec);
        }
    }

    std::cerr << "[Fonts] No suitable font file found in candidate paths." << std::endl;
    return {};
}

bool initializeFontFace()
{
    if (gInitialized)
    {
        return true;
    }

    if (gTriedInit)
    {
        return false;
    }
    gTriedInit = true;

    if (FT_Init_FreeType(&gLibrary) != 0)
    {
        std::cerr << "[Fonts] Failed to initialize FreeType." << std::endl;
        return false;
    }

    const auto fontPath = locateFont();
    if (fontPath.empty())
    {
        std::cerr << "[Fonts] Could not find nofile.ttf. Expected it in the project root." << std::endl;
        return false;
    }

    if (FT_New_Face(gLibrary, fontPath.string().c_str(), 0, &gFace) != 0)
    {
        std::cerr << "[Fonts] Failed to open font: " << fontPath << std::endl;
        return false;
    }

    gInitialized = true;
    return true;
}

FontBitmap buildFallbackBitmap(const std::string &text)
{
    FontBitmap bitmap;
    const uint32_t width = static_cast<uint32_t>(std::max<size_t>(1, text.size() * 8));
    bitmap.width = width;
    bitmap.height = 12;
    bitmap.pixels.assign(static_cast<size_t>(bitmap.width) * bitmap.height * 4, 0);

    for (uint32_t y = 0; y < bitmap.height; ++y)
    {
        for (uint32_t x = 0; x < bitmap.width; ++x)
        {
            const size_t idx = (static_cast<size_t>(y) * bitmap.width + x) * 4;
            bitmap.pixels[idx + 0] = 255;
            bitmap.pixels[idx + 1] = 0;
            bitmap.pixels[idx + 2] = 255;
            bitmap.pixels[idx + 3] = 200;
        }
    }
    return bitmap;
}
} // namespace

FontBitmap renderText(const std::string &text, uint32_t pixelHeight)
{
    FontBitmap bitmap;
    if (text.empty())
    {
        return bitmap;
    }

    if (!initializeFontFace())
    {
        return buildFallbackBitmap(text);
    }

    const uint32_t requestedSize = std::max(pixelHeight, kMinimumFontSize);
    FT_Set_Pixel_Sizes(gFace, 0, requestedSize);

    int penX = 0;
    int maxAboveBaseline = 0;
    int maxBelowBaseline = 0;

    const auto advanceChar = [&](unsigned long codepoint) {
        if (FT_Load_Char(gFace, codepoint, FT_LOAD_RENDER) != 0)
        {
            return;
        }
        FT_GlyphSlot slot = gFace->glyph;
        maxAboveBaseline = std::max(maxAboveBaseline, slot->bitmap_top);
        const int below = slot->bitmap.rows - slot->bitmap_top;
        maxBelowBaseline = std::max(maxBelowBaseline, below);
        penX += (slot->advance.x >> 6);
        penX += 1; // additional spacing between glyphs
    };

    for (unsigned char c : text)
    {
        if (c == '\n')
        {
            continue;
        }
        advanceChar(static_cast<unsigned long>(c));
    }

    if (penX <= 0)
    {
        return bitmap;
    }

    bitmap.width = static_cast<uint32_t>(std::max(1, penX));
    bitmap.height = static_cast<uint32_t>(std::max(1, maxAboveBaseline + maxBelowBaseline));
    bitmap.pixels.assign(static_cast<size_t>(bitmap.width) * bitmap.height * 4, 0);

    int penPosition = 0;
    for (unsigned char c : text)
    {
        if (c == '\n')
        {
            continue;
        }
        if (FT_Load_Char(gFace, static_cast<unsigned long>(c), FT_LOAD_RENDER) != 0)
        {
            continue;
        }
        FT_GlyphSlot slot = gFace->glyph;
        const FT_Bitmap &glyphBitmap = slot->bitmap;
        const int xOrigin = penPosition + slot->bitmap_left;
        const int yOrigin = maxAboveBaseline - slot->bitmap_top;

        for (int row = 0; row < glyphBitmap.rows; ++row)
        {
            for (int col = 0; col < glyphBitmap.width; ++col)
            {
                const int dstX = xOrigin + col;
                const int dstY = yOrigin + row;
                if (dstX < 0 || dstX >= static_cast<int>(bitmap.width) ||
                    dstY < 0 || dstY >= static_cast<int>(bitmap.height))
                {
                    continue;
                }

                const size_t srcIndex = static_cast<size_t>(row) * glyphBitmap.pitch + col;
                const uint8_t alpha = glyphBitmap.buffer[srcIndex];
                if (alpha == 0)
                {
                    continue;
                }

                const size_t dstIndex = (static_cast<size_t>(dstY) * bitmap.width + dstX) * 4;
                bitmap.pixels[dstIndex + 0] = 255;
                bitmap.pixels[dstIndex + 1] = 255;
                bitmap.pixels[dstIndex + 2] = 255;
                bitmap.pixels[dstIndex + 3] = alpha;
            }
        }

        penPosition += (slot->advance.x >> 6);
        penPosition += 1;
    }

    return bitmap;
}

} // namespace fonts
