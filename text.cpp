#include "text.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>
#include <vector>
#include <cstdlib>
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

const bool kFontDebugLogging = (::getenv("MOTIVE2D_DEBUG_FONTS") != nullptr);

std::filesystem::path locateFont()
{
    // Expanded list of candidate fonts and paths
    const std::vector<std::filesystem::path> candidates = {
        "nofile.ttf",
        // Preferred font
        "../nofile.ttf",
        "../../nofile.ttf",
        // Common system font paths
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        // Relative paths from executable
        "fonts/DejaVuSans.ttf",
        "../fonts/DejaVuSans.ttf",
        "../../fonts/DejaVuSans.ttf",
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

    // Create a visible fallback: blue background with white text outline
    for (uint32_t y = 0; y < bitmap.height; ++y)
    {
        for (uint32_t x = 0; x < bitmap.width; ++x)
        {
            const size_t idx = (static_cast<size_t>(y) * bitmap.width + x) * 4;
            // Blue background for debugging
            bitmap.pixels[idx + 0] = 0;     // R
            bitmap.pixels[idx + 1] = 0;     // G
            bitmap.pixels[idx + 2] = 255;   // B - BLUE for debugging
            bitmap.pixels[idx + 3] = 200;   // A
        }
    }
    
    // Add a simple white X pattern to indicate fallback
    for (uint32_t i = 0; i < std::min(width, bitmap.height); ++i)
    {
        // Diagonal from top-left to bottom-right
        if (i < width && i < bitmap.height)
        {
            size_t idx1 = (static_cast<size_t>(i) * bitmap.width + i) * 4;
            bitmap.pixels[idx1 + 0] = 255;  // R
            bitmap.pixels[idx1 + 1] = 255;  // G
            bitmap.pixels[idx1 + 2] = 255;  // B
            bitmap.pixels[idx1 + 3] = 255;  // A
            
            // Diagonal from top-right to bottom-left
            size_t x2 = width - 1 - i;
            if (x2 < width)
            {
                size_t idx2 = (static_cast<size_t>(i) * bitmap.width + x2) * 4;
                bitmap.pixels[idx2 + 0] = 255;  // R
                bitmap.pixels[idx2 + 1] = 255;  // G
                bitmap.pixels[idx2 + 2] = 255;  // B
                bitmap.pixels[idx2 + 3] = 255;  // A
            }
        }
    }
    
    std::cout << "[Fonts] Using fallback bitmap for text: \"" << text << "\" (size: " 
              << width << "x" << bitmap.height << ")" << std::endl;
    return bitmap;
}
} // namespace

FontBitmap renderText(const std::string &text, uint32_t pixelHeight)
{
    FontBitmap bitmap;
    if (text.empty())
    {
    std::cout << "[Fonts] renderText called with empty text" << std::endl;
        return bitmap;
    }

    if (kFontDebugLogging)
    {
        std::cout << "[Fonts] Rendering text: \"" << text << "\" at size: " << pixelHeight << std::endl;
    }

    if (!initializeFontFace())
    {
        std::cout << "[Fonts] Font initialization failed, using fallback" << std::endl;
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
            std::cout << "[Fonts] Failed to load character: " << static_cast<char>(codepoint) << std::endl;
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
        std::cout << "[Fonts] penX <= 0, using fallback" << std::endl;
        return buildFallbackBitmap(text);
    }

    bitmap.width = static_cast<uint32_t>(std::max(1, penX));
    bitmap.height = static_cast<uint32_t>(std::max(1, maxAboveBaseline + maxBelowBaseline));
    bitmap.pixels.assign(static_cast<size_t>(bitmap.width) * bitmap.height * 4, 0);

    if (kFontDebugLogging)
    {
        std::cout << "[Fonts] Bitmap dimensions: " << bitmap.width << "x" << bitmap.height << std::endl;
    }

    int penPosition = 0;
    for (unsigned char c : text)
    {
        if (c == '\n')
        {
            continue;
        }
        if (FT_Load_Char(gFace, static_cast<unsigned long>(c), FT_LOAD_RENDER) != 0)
        {
            std::cout << "[Fonts] Failed to render character: " << c << std::endl;
            continue;
        }
        FT_GlyphSlot slot = gFace->glyph;
        const FT_Bitmap &glyphBitmap = slot->bitmap;
        const int xOrigin = penPosition + slot->bitmap_left;
        const int yOrigin = maxAboveBaseline - slot->bitmap_top;

        if (kFontDebugLogging)
        {
            std::cout << "[Fonts] Character '" << c << "' at (" << xOrigin << "," << yOrigin 
                      << ") size: " << glyphBitmap.width << "x" << glyphBitmap.rows << std::endl;
        }

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

    // Count non-zero alpha pixels for debugging
    size_t nonZeroAlpha = 0;
    for (size_t i = 3; i < bitmap.pixels.size(); i += 4)
    {
        if (bitmap.pixels[i] > 0)
        {
            nonZeroAlpha++;
        }
    }
    if (kFontDebugLogging)
    {
        std::cout << "[Fonts] Rendered text with " << nonZeroAlpha << " non-zero alpha pixels" << std::endl;
    }

    return bitmap;
}

} // namespace fonts
