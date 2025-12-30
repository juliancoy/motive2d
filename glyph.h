#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace glyph
{
struct OverlayBitmap
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t offsetX = 0;
    uint32_t offsetY = 0;
    std::vector<uint8_t> pixels;
};

OverlayBitmap buildLabeledOverlay(uint32_t referenceWidth,
                                  uint32_t referenceHeight,
                                  std::string_view label,
                                  float value);

OverlayBitmap buildFrameRateOverlay(uint32_t referenceWidth,
                                    uint32_t referenceHeight,
                                    float fps);
}
