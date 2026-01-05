#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fonts
{

struct FontBitmap
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels; // RGBA
};

// Render the given text using the nofile.ttf font at the requested pixel height.
// Returns an RGBA bitmap (white text with alpha) positioned on the baseline.
FontBitmap renderText(const std::string &text, uint32_t pixelHeight);

} // namespace fonts
