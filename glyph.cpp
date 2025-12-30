#include "glyph.h"

#include "fonts.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace glyph
{
namespace
{
constexpr int kBaseMargin = 8;
constexpr int kBasePadding = 4;
constexpr int kBaseFontSize = 16;

uint32_t computePixelScale(uint32_t referenceWidth, uint32_t referenceHeight)
{
    int scaleFromWidth = static_cast<int>(referenceWidth / 640);
    int scaleFromHeight = static_cast<int>(referenceHeight / 360);
    scaleFromWidth = std::max(1, scaleFromWidth);
    scaleFromHeight = std::max(1, scaleFromHeight);
    return static_cast<uint32_t>(std::max(1, std::min(scaleFromWidth, scaleFromHeight)));
}

void fillBackground(OverlayBitmap &bitmap)
{
    bitmap.pixels.assign(static_cast<size_t>(bitmap.width) * bitmap.height * 4, 0);
    for (uint32_t y = 0; y < bitmap.height; ++y)
    {
        for (uint32_t x = 0; x < bitmap.width; ++x)
        {
            const size_t index = (static_cast<size_t>(y) * bitmap.width + x) * 4;
            bitmap.pixels[index + 0] = 0;
            bitmap.pixels[index + 1] = 0;
            bitmap.pixels[index + 2] = 0;
            bitmap.pixels[index + 3] = 200; // translucent background
        }
    }
}

void blitText(const fonts::FontBitmap &textBitmap,
              OverlayBitmap &overlay,
              uint32_t padding)
{
    if (textBitmap.width == 0 || textBitmap.height == 0 || overlay.pixels.empty())
    {
        return;
    }

    const uint32_t startX = std::min(padding, overlay.width);
    const uint32_t startY = std::min(padding, overlay.height);

    const uint32_t maxCopyWidth = std::min(textBitmap.width, overlay.width > startX ? overlay.width - startX : 0);
    const uint32_t maxCopyHeight = std::min(textBitmap.height, overlay.height > startY ? overlay.height - startY : 0);

    for (uint32_t row = 0; row < maxCopyHeight; ++row)
    {
        for (uint32_t col = 0; col < maxCopyWidth; ++col)
        {
            const size_t srcIndex = (static_cast<size_t>(row) * textBitmap.width + col) * 4;
            const uint8_t alpha = textBitmap.pixels[srcIndex + 3];
            if (alpha == 0)
            {
                continue;
            }

            const size_t dstIndex = (static_cast<size_t>(row + startY) * overlay.width + (col + startX)) * 4;
            const uint8_t srcR = textBitmap.pixels[srcIndex + 0];
            const uint8_t srcG = textBitmap.pixels[srcIndex + 1];
            const uint8_t srcB = textBitmap.pixels[srcIndex + 2];

            const uint8_t dstR = overlay.pixels[dstIndex + 0];
            const uint8_t dstG = overlay.pixels[dstIndex + 1];
            const uint8_t dstB = overlay.pixels[dstIndex + 2];
            const uint8_t dstA = overlay.pixels[dstIndex + 3];

            const float srcAF = static_cast<float>(alpha) / 255.0f;
            const float dstAF = static_cast<float>(dstA) / 255.0f;
            const float outA = srcAF + dstAF * (1.0f - srcAF);
            if (outA <= 0.0f)
            {
                continue;
            }

            const auto blendChannel = [&](uint8_t srcC, uint8_t dstC) -> uint8_t {
                const float srcCF = static_cast<float>(srcC) / 255.0f;
                const float dstCF = static_cast<float>(dstC) / 255.0f;
                const float outCF = (srcCF * srcAF + dstCF * dstAF * (1.0f - srcAF)) / outA;
                return static_cast<uint8_t>(std::clamp(outCF * 255.0f, 0.0f, 255.0f));
            };

            overlay.pixels[dstIndex + 0] = blendChannel(srcR, dstR);
            overlay.pixels[dstIndex + 1] = blendChannel(srcG, dstG);
            overlay.pixels[dstIndex + 2] = blendChannel(srcB, dstB);
            overlay.pixels[dstIndex + 3] = static_cast<uint8_t>(std::clamp(outA * 255.0f, 0.0f, 255.0f));
        }
    }
}
} // namespace

OverlayBitmap buildLabeledOverlay(uint32_t referenceWidth,
                                  uint32_t referenceHeight,
                                  std::string_view label,
                                  float value)
{
    OverlayBitmap bitmap{};
    if (referenceWidth == 0 || referenceHeight == 0)
    {
        return bitmap;
    }

    char text[64];
    const std::string labelText = label.empty() ? "" : std::string(label) + " ";
    if (value > 0.0f)
    {
        std::snprintf(text, sizeof(text), "%s%.1f", labelText.c_str(), value);
    }
    else
    {
        std::snprintf(text, sizeof(text), "%s----", labelText.c_str());
    }

    const uint32_t pixelScale = computePixelScale(referenceWidth, referenceHeight);
    const uint32_t fontSize = std::max<uint32_t>(kBaseFontSize, kBaseFontSize * pixelScale);
    const fonts::FontBitmap textBitmap = fonts::renderText(text, fontSize);
    if (textBitmap.width == 0 || textBitmap.height == 0)
    {
        return bitmap;
    }

    const uint32_t padding = static_cast<uint32_t>(kBasePadding * static_cast<int>(pixelScale));
    const uint32_t desiredWidth = textBitmap.width + padding * 2;
    const uint32_t desiredHeight = textBitmap.height + padding * 2;
    bitmap.width = std::min(desiredWidth, referenceWidth);
    bitmap.height = std::min(desiredHeight, referenceHeight);

    fillBackground(bitmap);

    const uint32_t scaledMargin = static_cast<uint32_t>(kBaseMargin * static_cast<int>(pixelScale));
    const uint32_t maxOffsetX = referenceWidth > bitmap.width ? referenceWidth - bitmap.width : 0;
    const uint32_t maxOffsetY = referenceHeight > bitmap.height ? referenceHeight - bitmap.height : 0;
    bitmap.offsetX = std::min(scaledMargin, maxOffsetX);
    bitmap.offsetY = std::min(scaledMargin, maxOffsetY);

    blitText(textBitmap, bitmap, padding);
    return bitmap;
}

OverlayBitmap buildFrameRateOverlay(uint32_t referenceWidth,
                                    uint32_t referenceHeight,
                                    float fps)
{
    return buildLabeledOverlay(referenceWidth, referenceHeight, "FPS", fps);
}

} // namespace glyph
