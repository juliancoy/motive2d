#include "grading.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>

#include "engine2d.h"
#include "glyph.h"
#include "overlay.hpp"

namespace
{
float clamp01(float v)
{
    return std::clamp(v, 0.0f, 1.0f);
}

float normalizeExposure(float exposure)
{
    return clamp01((exposure + 2.0f) / 4.0f); // -2..2 -> 0..1
}

float denormalizeExposure(float norm)
{
    return std::clamp(norm * 4.0f - 2.0f, -4.0f, 4.0f);
}

float normalizeContrast(float contrast)
{
    // 0.5..2.0
    return clamp01((contrast - 0.5f) / 1.5f);
}

float denormalizeContrast(float norm)
{
    return std::clamp(0.5f + norm * 1.5f, 0.1f, 3.0f);
}

float normalizeSaturation(float sat)
{
    // 0..2
    return clamp01(sat * 0.5f);
}

float denormalizeSaturation(float norm)
{
    return std::clamp(norm * 2.0f, 0.0f, 3.0f);
}

float clampCurveX(float x, float prevX, float nextX)
{
    const float eps = 0.01f;
    return std::clamp(x, prevX + eps, nextX - eps);
}

void drawLine(std::vector<uint8_t>& buf,
              uint32_t width,
              uint32_t height,
              int x0,
              int y0,
              int x1,
              int y1,
              uint8_t r,
              uint8_t g,
              uint8_t b,
              uint8_t a)
{
    const int dx = std::abs(x1 - x0);
    const int dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int x = x0;
    int y = y0;
    while (true)
    {
        if (x >= 0 && x < static_cast<int>(width) && y >= 0 && y < static_cast<int>(height))
        {
            size_t idx = (static_cast<size_t>(y) * width + static_cast<size_t>(x)) * 4;
            buf[idx + 0] = r;
            buf[idx + 1] = g;
            buf[idx + 2] = b;
            buf[idx + 3] = a;
        }
        if (x == x1 && y == y1)
        {
            break;
        }
        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y += sy;
        }
    }
}

void drawRect(std::vector<uint8_t>& buf,
              uint32_t width,
              uint32_t height,
              uint32_t x0,
              uint32_t y0,
              uint32_t x1,
              uint32_t y1,
              uint8_t r,
              uint8_t g,
              uint8_t b,
              uint8_t a)
{
    x0 = std::min(x0, width);
    x1 = std::min(x1, width);
    y0 = std::min(y0, height);
    y1 = std::min(y1, height);
    for (uint32_t y = y0; y < y1; ++y)
    {
        for (uint32_t x = x0; x < x1; ++x)
        {
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            buf[idx + 0] = r;
            buf[idx + 1] = g;
            buf[idx + 2] = b;
            buf[idx + 3] = a;
        }
    }
}

void drawHandle(std::vector<uint8_t>& buf,
                uint32_t width,
                uint32_t height,
                float cx,
                float cy,
                float radius,
                uint8_t r,
                uint8_t g,
                uint8_t b,
                uint8_t a)
{
    int minX = std::max(0, static_cast<int>(std::floor(cx - radius - 1)));
    int maxX = std::min(static_cast<int>(width), static_cast<int>(std::ceil(cx + radius + 1)));
    int minY = std::max(0, static_cast<int>(std::floor(cy - radius - 1)));
    int maxY = std::min(static_cast<int>(height), static_cast<int>(std::ceil(cy + radius + 1)));
    float rad2 = radius * radius;
    for (int y = minY; y < maxY; ++y)
    {
        for (int x = minX; x < maxX; ++x)
        {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float dist2 = dx * dx + dy * dy;
            if (dist2 <= rad2)
            {
                size_t idx = (static_cast<size_t>(y) * width + static_cast<uint32_t>(x)) * 4;
                buf[idx + 0] = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
                buf[idx + 3] = a;
            }
        }
    }
}
} // namespace

namespace grading
{

bool buildGradingOverlay(Engine2D* engine,
                         const GradingSettings& settings,
                         overlay::ImageResource& image,
                         OverlayImageInfo& info,
                         uint32_t fbWidth,
                         uint32_t fbHeight,
                         SliderLayout& layout,
                         bool previewEnabled,
                         bool detectionEnabled)
{
    constexpr std::array<const char*, 12> kDefaultLabels = {
        "Exposure",
        "Contrast",
        "Saturation",
        "Shadows R",
        "Shadows G",
        "Shadows B",
        "Midtones R",
        "Midtones G",
        "Midtones B",
        "Highlights R",
        "Highlights G",
        "Highlights B",
    };

    if (fbWidth == 0 || fbHeight == 0)
    {
        return false;
    }
    layout.width = 420;
    layout.height = 880;
    layout.margin = 20;
    layout.barHeight = 20;
    layout.curvesPadding = 16;
    layout.curvesHeight = 200;
    layout.curvesX0 = layout.curvesPadding;
    layout.curvesX1 = layout.width > layout.curvesPadding * 2
                          ? layout.width - layout.curvesPadding
                          : layout.width;
    layout.curvesY0 = layout.curvesPadding;
    layout.curvesY1 = layout.curvesY0 + layout.curvesHeight;
    layout.barYStart = layout.curvesY1 + 32;
    layout.rowSpacing = 12;
    layout.handleRadius = 10;

    std::vector<uint8_t> pixels(static_cast<size_t>(layout.width) * layout.height * 4, 0);
    drawRect(pixels, layout.width, layout.height, 0, 0, layout.width, layout.height, 0, 0, 0, 180);

    const uint32_t padding = 12;
    const uint32_t barWidth = layout.width - padding * 2;
    const uint32_t barYStart = layout.barYStart;

    struct SliderDesc
    {
        float norm;
        uint32_t color[3];
    };
    SliderDesc sliders[12] = {
        {normalizeExposure(settings.exposure), {255, 180, 80}},
        {normalizeContrast(settings.contrast), {80, 200, 255}},
        {normalizeSaturation(settings.saturation), {180, 255, 160}},
        {clamp01(settings.shadows.r * 0.5f), {255, 120, 120}},
        {clamp01(settings.shadows.g * 0.5f), {120, 255, 120}},
        {clamp01(settings.shadows.b * 0.5f), {120, 120, 255}},
        {clamp01(settings.midtones.r * 0.5f), {255, 120, 120}},
        {clamp01(settings.midtones.g * 0.5f), {120, 255, 120}},
        {clamp01(settings.midtones.b * 0.5f), {120, 120, 255}},
        {clamp01(settings.highlights.r * 0.5f), {255, 120, 120}},
        {clamp01(settings.highlights.g * 0.5f), {120, 255, 120}},
        {clamp01(settings.highlights.b * 0.5f), {120, 120, 255}},
    };

    // Curves editor background
    {
        drawRect(pixels,
                 layout.width,
                 layout.height,
                 layout.curvesX0,
                 layout.curvesY0,
                 layout.curvesX1,
                 layout.curvesY1,
                 35,
                 35,
                 35,
                 255);
        const uint32_t curvesW = layout.curvesX1 - layout.curvesX0;
        const uint32_t curvesH = layout.curvesY1 - layout.curvesY0;
        // Grid lines (4x4)
        for (int i = 1; i < 4; ++i)
        {
            uint32_t x = layout.curvesX0 + static_cast<uint32_t>(std::round(static_cast<float>(curvesW) * (i / 4.0f)));
            uint32_t y = layout.curvesY0 + static_cast<uint32_t>(std::round(static_cast<float>(curvesH) * (i / 4.0f)));
            drawRect(pixels, layout.width, layout.height, x, layout.curvesY0, x + 1, layout.curvesY1, 55, 55, 55, 255);
            drawRect(pixels, layout.width, layout.height, layout.curvesX0, y, layout.curvesX1, y + 1, 55, 55, 55, 255);
        }
        // Axes
        drawRect(pixels, layout.width, layout.height, layout.curvesX0, layout.curvesY1 - 1, layout.curvesX1, layout.curvesY1, 80, 80, 80, 255);
        drawRect(pixels, layout.width, layout.height, layout.curvesX0, layout.curvesY0, layout.curvesX0 + 1, layout.curvesY1, 80, 80, 80, 255);

        // Curve polyline (linear between control points)
        auto toPx = [&](const glm::vec2& p) -> glm::ivec2 {
            float xNorm = clamp01(p.x);
            float yNorm = clamp01(p.y);
            int x = static_cast<int>(std::round(static_cast<float>(layout.curvesX0) + xNorm * static_cast<float>(curvesW)));
            int y = static_cast<int>(std::round(static_cast<float>(layout.curvesY1) - yNorm * static_cast<float>(curvesH)));
            return {x, y};
        };
        if (settings.curves.size() >= 2)
        {
            glm::ivec2 last = toPx(settings.curves.front());
            for (size_t i = 1; i < settings.curves.size(); ++i)
            {
                glm::ivec2 curr = toPx(settings.curves[i]);
                drawLine(pixels, layout.width, layout.height, last.x, last.y, curr.x, curr.y, 200, 200, 220, 255);
                last = curr;
            }
            // Handles
            for (size_t i = 0; i < settings.curves.size(); ++i)
            {
                glm::ivec2 p = toPx(settings.curves[i]);
                uint8_t handleA = (i == 0 || i == settings.curves.size() - 1) ? 160 : 255;
                drawHandle(pixels,
                           layout.width,
                           layout.height,
                           static_cast<float>(p.x),
                           static_cast<float>(p.y),
                           static_cast<float>(layout.handleRadius) * 0.6f,
                           255,
                           255,
                           255,
                           handleA);
            }
        }
    }

    for (int i = 0; i < 12; ++i)
    {
        uint32_t y0 = barYStart + i * (layout.barHeight + layout.rowSpacing);
        uint32_t y1 = y0 + layout.barHeight;
        uint32_t x0 = padding;
        uint32_t x1 = padding + barWidth;
        drawRect(pixels, layout.width, layout.height, x0, y0, x1, y1, 40, 40, 40, 255);

        uint32_t fillW = static_cast<uint32_t>(std::round(sliders[i].norm * static_cast<float>(barWidth)));
        drawRect(pixels, layout.width, layout.height, x0, y0, x0 + fillW, y1,
                 sliders[i].color[0], sliders[i].color[1], sliders[i].color[2], 255);

        float handleX = static_cast<float>(x0 + fillW);
        float handleY = static_cast<float>(y0 + layout.barHeight * 0.5f);
        drawHandle(pixels, layout.width, layout.height, handleX, handleY, static_cast<float>(layout.handleRadius),
                   255, 255, 255, 255);

        // Label text drawn above the bar
        const char* labelText = (i < layout.labels.size() && !layout.labels[i].empty()) ? layout.labels[i].c_str() : kDefaultLabels[i];
        glyph::OverlayBitmap label = glyph::buildLabeledOverlay(layout.width, layout.height, labelText, 0.0f);
        if (!label.pixels.empty() && label.width > 0 && label.height > 0)
        {
            uint32_t textX = x0;
            uint32_t textY = y0 >= label.height + 2 ? y0 - label.height - 2 : 0;
            for (uint32_t ty = 0; ty < label.height && (textY + ty) < layout.height; ++ty)
            {
                for (uint32_t tx = 0; tx < label.width && (textX + tx) < layout.width; ++tx)
                {
                    size_t srcIdx = (static_cast<size_t>(ty) * label.width + tx) * 4;
                    size_t dstIdx = (static_cast<size_t>(textY + ty) * layout.width + (textX + tx)) * 4;
                    uint8_t a = label.pixels[srcIdx + 3];
                    if (a == 0)
                    {
                        continue;
                    }
                    pixels[dstIdx + 0] = label.pixels[srcIdx + 0];
                    pixels[dstIdx + 1] = label.pixels[srcIdx + 1];
                    pixels[dstIdx + 2] = label.pixels[srcIdx + 2];
                    pixels[dstIdx + 3] = std::max<uint8_t>(pixels[dstIdx + 3], a);
                }
            }
        }
    }

    // Reset button near the bottom
    const uint32_t buttonPadding = 12;
    const uint32_t totalButtonsHeight = layout.resetHeight + layout.loadHeight + layout.saveHeight +
                                        layout.previewHeight + layout.detectionHeight + buttonPadding * 4;
    const uint32_t barBlockHeight = 12 * (layout.barHeight + layout.rowSpacing);
    const uint32_t minButtonY = layout.barYStart + barBlockHeight + 20;
    uint32_t buttonStartY =
        layout.height > (totalButtonsHeight + buttonPadding) ? layout.height - totalButtonsHeight - buttonPadding : 0;
    if (buttonStartY < minButtonY)
    {
        buttonStartY = minButtonY;
    }
    auto centerButtonX = [&](uint32_t w) {
        return (layout.width > w) ? (layout.width - w) / 2 : 0u;
    };

    layout.resetX0 = centerButtonX(layout.resetWidth);
    layout.resetX1 = layout.resetX0 + layout.resetWidth;
    layout.resetY0 = buttonStartY;
    layout.resetY1 = layout.resetY0 + layout.resetHeight;

    layout.loadX0 = centerButtonX(layout.loadWidth);
    layout.loadX1 = layout.loadX0 + layout.loadWidth;
    layout.loadY0 = layout.resetY1 + buttonPadding;
    layout.loadY1 = layout.loadY0 + layout.loadHeight;

    layout.saveX0 = centerButtonX(layout.saveWidth);
    layout.saveX1 = layout.saveX0 + layout.saveWidth;
    layout.saveY0 = layout.loadY1 + buttonPadding;
    layout.saveY1 = layout.saveY0 + layout.saveHeight;

    layout.previewX0 = centerButtonX(layout.previewWidth);
    layout.previewX1 = layout.previewX0 + layout.previewWidth;
    layout.previewY0 = layout.saveY1 + buttonPadding;
    layout.previewY1 = layout.previewY0 + layout.previewHeight;

    layout.detectionX0 = centerButtonX(layout.detectionWidth);
    layout.detectionX1 = layout.detectionX0 + layout.detectionWidth;
    layout.detectionY0 = layout.previewY1 + buttonPadding;
    layout.detectionY1 = layout.detectionY0 + layout.detectionHeight;

    auto drawButton = [&](uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, const char* text) {
        drawRect(pixels, layout.width, layout.height, x0, y0, x0 + w, y0 + h, 70, 70, 70, 255);
        drawRect(pixels, layout.width, layout.height, x0 + 2, y0 + 2, x0 + w - 2, y0 + h - 2, 30, 30, 30, 255);
        glyph::OverlayBitmap textBmp = glyph::buildLabeledOverlay(layout.width, layout.height, text, 0.0f);
        if (!textBmp.pixels.empty())
        {
            uint32_t textX = x0 + (w > textBmp.width ? (w - textBmp.width) / 2 : 0);
            uint32_t textY = y0 + (h > textBmp.height ? (h - textBmp.height) / 2 : 0);
            for (uint32_t ty = 0; ty < textBmp.height && (textY + ty) < layout.height; ++ty)
            {
                for (uint32_t tx = 0; tx < textBmp.width && (textX + tx) < layout.width; ++tx)
                {
                    size_t srcIdx = (static_cast<size_t>(ty) * textBmp.width + tx) * 4;
                    size_t dstIdx = (static_cast<size_t>(textY + ty) * layout.width + (textX + tx)) * 4;
                    uint8_t a = textBmp.pixels[srcIdx + 3];
                    if (a == 0)
                    {
                        continue;
                    }
                    pixels[dstIdx + 0] = textBmp.pixels[srcIdx + 0];
                    pixels[dstIdx + 1] = textBmp.pixels[srcIdx + 1];
                    pixels[dstIdx + 2] = textBmp.pixels[srcIdx + 2];
                    pixels[dstIdx + 3] = std::max<uint8_t>(pixels[dstIdx + 3], a);
                }
            }
        }
    };

    drawButton(layout.resetX0, layout.resetY0, layout.resetWidth, layout.resetHeight, "Reset");
    drawButton(layout.loadX0, layout.loadY0, layout.loadWidth, layout.loadHeight, "Load");
    drawButton(layout.saveX0, layout.saveY0, layout.saveWidth, layout.saveHeight, "Save");
    drawButton(layout.previewX0,
               layout.previewY0,
               layout.previewWidth,
               layout.previewHeight,
               previewEnabled ? "Preview On" : "Preview Off");
    
    drawButton(layout.detectionX0,
               layout.detectionY0,
               layout.detectionWidth,
               layout.detectionHeight,
               detectionEnabled ? "Detection On" : "Detection Off");

    const uint32_t overlayX = (fbWidth > layout.width) ? (fbWidth - layout.width) / 2 : 0;
    const uint32_t overlayY = (fbHeight > layout.height + layout.margin) ? fbHeight - layout.height - layout.margin : 0;
    layout.offset = {static_cast<int32_t>(overlayX), static_cast<int32_t>(overlayY)};

    if (!overlay::uploadImageData(engine,
                                  image,
                                  pixels.data(),
                                  pixels.size(),
                                  layout.width,
                                  layout.height,
                                  VK_FORMAT_R8G8B8A8_UNORM))
    {
        return false;
    }

    info.overlay.view = image.view;
    info.overlay.sampler = VK_NULL_HANDLE; // sampler set by caller
    info.extent = {layout.width, layout.height};
    info.offset = layout.offset;
    info.enabled = true;
    return true;
}

bool handleOverlayClick(const SliderLayout& layout,
                        double cursorX,
                        double cursorY,
                        GradingSettings& settings,
                        bool doubleClick,
                        bool rightClick,
                        bool* loadRequested,
                        bool* saveRequested,
                        bool* previewToggleRequested,
                        bool* detectionToggleRequested)
{
    const double relX = cursorX - static_cast<double>(layout.offset.x);
    const double relY = cursorY - static_cast<double>(layout.offset.y);
    if (relX < 0.0 || relY < 0.0 || relX >= layout.width || relY >= layout.height)
    {
        return false;
    }

    // Curves editor hit test
    if (relX >= layout.curvesX0 && relX <= layout.curvesX1 && relY >= layout.curvesY0 && relY <= layout.curvesY1)
    {
        const float width = static_cast<float>(layout.curvesX1 - layout.curvesX0);
        const float height = static_cast<float>(layout.curvesY1 - layout.curvesY0);
        if (doubleClick)
        {
            settings.curves = {glm::vec2(0.0f, 0.0f),
                               glm::vec2(0.33f, 0.33f),
                               glm::vec2(0.66f, 0.66f),
                               glm::vec2(1.0f, 1.0f)};
            return true;
        }
        float normX = clamp01(static_cast<float>(relX - layout.curvesX0) / width);
        float normY = clamp01(1.0f - static_cast<float>(relY - layout.curvesY0) / height);
        const size_t pointCount = settings.curves.size();
        if (pointCount < 2)
        {
            return true;
        }

        // Find closest handle
        size_t closest = 0;
        float bestDist2 = std::numeric_limits<float>::max();
        for (size_t i = 0; i < pointCount; ++i)
        {
            float dx = normX - settings.curves[i].x;
            float dy = normY - settings.curves[i].y;
            float d2 = dx * dx + dy * dy;
            if (d2 < bestDist2)
            {
                bestDist2 = d2;
                closest = i;
            }
        }

        constexpr float kRemoveThreshold2 = 0.015f * 0.015f; // normalized distance squared
        if (rightClick)
        {
            if (bestDist2 <= kRemoveThreshold2 && pointCount > 3 && closest > 0 && closest + 1 < pointCount)
            {
                settings.curves.erase(settings.curves.begin() + static_cast<std::ptrdiff_t>(closest));
            }
            else
            {
                constexpr size_t kMaxCurvePoints = 8;
                if (pointCount < kMaxCurvePoints)
                {
                    // Insert keeping x-order
                    size_t insertIdx = pointCount - 1;
                    for (size_t i = 1; i < pointCount; ++i)
                    {
                        if (normX <= settings.curves[i].x)
                        {
                            insertIdx = i;
                            break;
                        }
                    }
                    float prevX = settings.curves[insertIdx - 1].x;
                    float nextX = settings.curves[insertIdx].x;
                    glm::vec2 p{clampCurveX(normX, prevX, nextX), normY};
                    settings.curves.insert(settings.curves.begin() + static_cast<std::ptrdiff_t>(insertIdx), p);
                }
            }
            return true;
        }

        if (pointCount <= 2)
        {
            return true;
        }
        if (closest == 0)
        {
            settings.curves[closest].x = 0.0f;
            settings.curves[closest].y = normY;
        }
        else if (closest + 1 >= pointCount)
        {
            settings.curves[closest].x = 1.0f;
            settings.curves[closest].y = normY;
        }
        else
        {
            float prevX = settings.curves[closest - 1].x;
            float nextX = settings.curves[closest + 1].x;
            settings.curves[closest].x = clampCurveX(normX, prevX, nextX);
            settings.curves[closest].y = normY;
        }
        return true;
    }

    // Reset button hit test
    if (relX >= layout.resetX0 && relX <= layout.resetX1 && relY >= layout.resetY0 && relY <= layout.resetY1)
    {
        setGradingDefaults(settings);
        return true;
    }
    // Load button hit test
    if (relX >= layout.loadX0 && relX <= layout.loadX1 && relY >= layout.loadY0 && relY <= layout.loadY1)
    {
        if (loadRequested)
        {
            *loadRequested = true;
        }
        return true;
    }
    // Save button hit test
    if (relX >= layout.saveX0 && relX <= layout.saveX1 && relY >= layout.saveY0 && relY <= layout.saveY1)
    {
        if (saveRequested)
        {
            *saveRequested = true;
        }
        return true;
    }
    // Preview button hit test
    if (relX >= layout.previewX0 && relX <= layout.previewX1 && relY >= layout.previewY0 && relY <= layout.previewY1)
    {
        if (previewToggleRequested)
        {
            *previewToggleRequested = true;
        }
        return true;
    }
    // Detection button hit test
    if (relX >= layout.detectionX0 && relX <= layout.detectionX1 && relY >= layout.detectionY0 && relY <= layout.detectionY1)
    {
        if (detectionToggleRequested)
        {
            *detectionToggleRequested = true;
        }
        return true;
    }

    const uint32_t padding = 12;
    const uint32_t barWidth = layout.width - padding * 2;
    const uint32_t barYStart = layout.barYStart;
    const uint32_t rowHeight = layout.barHeight + layout.rowSpacing;
    int sliderIndex = static_cast<int>((relY - barYStart) / rowHeight);
    if (sliderIndex < 0 || sliderIndex >= 12)
    {
        return false;
    }

    float t = clamp01(static_cast<float>((relX - padding) / static_cast<double>(barWidth)));
    if (rightClick)
    {
        switch (sliderIndex)
        {
        case 0: settings.exposure = 0.0f; break;
        case 1: settings.contrast = 1.0f; break;
        case 2: settings.saturation = 1.0f; break;
        case 3: settings.shadows.r = 1.0f; break;
        case 4: settings.shadows.g = 1.0f; break;
        case 5: settings.shadows.b = 1.0f; break;
        case 6: settings.midtones.r = 1.0f; break;
        case 7: settings.midtones.g = 1.0f; break;
        case 8: settings.midtones.b = 1.0f; break;
        case 9: settings.highlights.r = 1.0f; break;
        case 10: settings.highlights.g = 1.0f; break;
        case 11: settings.highlights.b = 1.0f; break;
        default: break;
        }
        return true;
    }
    switch (sliderIndex)
    {
    case 0:
        settings.exposure = doubleClick ? 0.0f : denormalizeExposure(t);
        break;
    case 1:
        settings.contrast = doubleClick ? 1.0f : denormalizeContrast(t);
        break;
    case 2:
        settings.saturation = doubleClick ? 1.0f : denormalizeSaturation(t);
        break;
    case 3:
        settings.shadows.r = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 4:
        settings.shadows.g = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 5:
        settings.shadows.b = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 6:
        settings.midtones.r = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 7:
        settings.midtones.g = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 8:
        settings.midtones.b = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 9:
        settings.highlights.r = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 10:
        settings.highlights.g = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    case 11:
        settings.highlights.b = doubleClick ? 1.0f : std::clamp(t * 2.0f, 0.0f, 2.0f);
        break;
    default:
        break;
    }
    return true;
}

void setGradingDefaults(GradingSettings& settings)
{
    settings.exposure = 0.0f;
    settings.contrast = 1.0f;
    settings.saturation = 1.0f;
    settings.shadows = glm::vec3(1.0f);
    settings.midtones = glm::vec3(1.0f);
    settings.highlights = glm::vec3(1.0f);
    settings.curves = {glm::vec2(0.0f, 0.0f),
                       glm::vec2(0.33f, 0.33f),
                       glm::vec2(0.66f, 0.66f),
                       glm::vec2(1.0f, 1.0f)};
}

namespace
{
bool parseFloat(const std::string& src, const std::string& key, float& outValue)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = src.find(needle);
    if (pos == std::string::npos)
    {
        return false;
    }
    pos = src.find(':', pos);
    if (pos == std::string::npos)
    {
        return false;
    }
    const char* start = src.c_str() + pos + 1;
    char* endPtr = nullptr;
    float v = std::strtof(start, &endPtr);
    if (endPtr == start)
    {
        return false;
    }
    outValue = v;
    return true;
}

glm::vec3 parseVec3(const std::string& src, const std::string& key, const glm::vec3& fallback)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = src.find(needle);
    if (pos == std::string::npos)
    {
        return fallback;
    }
    pos = src.find('[', pos);
    if (pos == std::string::npos)
    {
        return fallback;
    }
    glm::vec3 result = fallback;
    const char* cursor = src.c_str() + pos + 1;
    for (int i = 0; i < 3; ++i)
    {
        char* endPtr = nullptr;
        float v = std::strtof(cursor, &endPtr);
        if (endPtr == cursor)
        {
            return fallback;
        }
        result[i] = v;
        cursor = endPtr;
        while (*cursor != '\0' && *cursor != ',' && *cursor != ']')
        {
            ++cursor;
        }
        if (i < 2)
        {
            if (*cursor != ',')
            {
                return fallback;
            }
            ++cursor;
        }
    }
    return result;
}

std::vector<glm::vec2> parseCurves(const std::string& src,
                                   const std::string& key,
                                   const std::vector<glm::vec2>& fallback)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = src.find(needle);
    if (pos == std::string::npos)
    {
        return fallback;
    }
    pos = src.find('[', pos);
    if (pos == std::string::npos)
    {
        return fallback;
    }
    std::vector<glm::vec2> result;
    const char* cursor = src.c_str() + pos + 1;
    while (true)
    {
        // find opening bracket for pair
        while (*cursor != '[' && *cursor != '\0')
        {
            ++cursor;
        }
        if (*cursor == '\0')
        {
            return fallback;
        }
        ++cursor; // skip '['
        char* endPtr = nullptr;
        float x = std::strtof(cursor, &endPtr);
        if (endPtr == cursor)
        {
            return fallback;
        }
        cursor = endPtr;
        while (*cursor != ',' && *cursor != '\0')
        {
            ++cursor;
        }
        if (*cursor == '\0')
        {
            return fallback;
        }
        ++cursor; // skip ','
        float y = std::strtof(cursor, &endPtr);
        if (endPtr == cursor)
        {
            return fallback;
        }
        result.emplace_back(clamp01(x), clamp01(y));
        cursor = endPtr;
        while (*cursor != ']' && *cursor != '\0')
        {
            ++cursor;
        }
        if (*cursor == '\0')
        {
            return fallback;
        }
        ++cursor; // past ']'
        while (*cursor != '[' && *cursor != '\0')
        {
            if (*cursor == ']')
            {
                // end of curves array
                if (result.size() >= 2)
                {
                    // ensure sorted by x and clamp endpoints
                    std::sort(result.begin(), result.end(), [](const glm::vec2& a, const glm::vec2& b) {
                        return a.x < b.x;
                    });
                    result.front() = glm::vec2(0.0f, result.front().y);
                    result.back() = glm::vec2(1.0f, result.back().y);
                    return result;
                }
                return fallback;
            }
            ++cursor;
        }
    }
}
} // namespace

bool loadGradingSettings(const std::filesystem::path& path, GradingSettings& settings)
{
    setGradingDefaults(settings);
    if (!std::filesystem::exists(path))
    {
        return false;
    }

    std::ifstream in(path);
    if (!in.is_open())
    {
        return false;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string contents = buffer.str();

    parseFloat(contents, "exposure", settings.exposure);
    parseFloat(contents, "contrast", settings.contrast);
    parseFloat(contents, "saturation", settings.saturation);
    settings.shadows = parseVec3(contents, "shadows", settings.shadows);
    settings.midtones = parseVec3(contents, "midtones", settings.midtones);
    settings.highlights = parseVec3(contents, "highlights", settings.highlights);
    settings.curves = parseCurves(contents, "curves", settings.curves);
    return true;
}

bool saveGradingSettings(const std::filesystem::path& path, const GradingSettings& settings)
{
    std::ofstream out(path);
    if (!out.is_open())
    {
        return false;
    }
    out << "{\n";
    out << "  \"exposure\": " << settings.exposure << ",\n";
    out << "  \"contrast\": " << settings.contrast << ",\n";
    out << "  \"saturation\": " << settings.saturation << ",\n";
    out << "  \"shadows\": [" << settings.shadows.r << ", " << settings.shadows.g << ", " << settings.shadows.b << "],\n";
    out << "  \"midtones\": [" << settings.midtones.r << ", " << settings.midtones.g << ", " << settings.midtones.b << "],\n";
    out << "  \"highlights\": [" << settings.highlights.r << ", " << settings.highlights.g << ", " << settings.highlights.b << "],\n";
    out << "  \"curves\": [";
    for (size_t i = 0; i < settings.curves.size(); ++i)
    {
        out << "[" << settings.curves[i].x << ", " << settings.curves[i].y << "]";
        if (i + 1 < settings.curves.size())
        {
            out << ", ";
        }
    }
    out << "]\n";
    out << "}\n";
    return true;
}

void buildCurveLut(const GradingSettings& settings, std::array<float, kCurveLutSize>& outLut)
{
    auto clamp01f = [](float v) { return std::clamp(v, 0.0f, 1.0f); };
    if (settings.curves.size() < 2)
    {
        for (size_t i = 0; i < kCurveLutSize; ++i)
        {
            outLut[i] = static_cast<float>(i) / static_cast<float>(kCurveLutSize - 1);
        }
        return;
    }

    std::vector<glm::vec2> pts = settings.curves;
    std::sort(pts.begin(), pts.end(), [](const glm::vec2& a, const glm::vec2& b) { return a.x < b.x; });
    pts.front().x = 0.0f;
    pts.back().x = 1.0f;
    for (auto& p : pts)
    {
        p.x = clamp01f(p.x);
        p.y = clamp01f(p.y);
    }

    size_t seg = 0;
    for (size_t i = 0; i < kCurveLutSize; ++i)
    {
        float x = static_cast<float>(i) / static_cast<float>(kCurveLutSize - 1);
        while (seg + 1 < pts.size() && x > pts[seg + 1].x)
        {
            ++seg;
        }
        size_t next = std::min(seg + 1, pts.size() - 1);
        float denom = std::max(pts[next].x - pts[seg].x, 1e-4f);
        float t = std::clamp((x - pts[seg].x) / denom, 0.0f, 1.0f);
        float y = pts[seg].y + (pts[next].y - pts[seg].y) * t;
        outLut[i] = clamp01f(y);
    }
}
} // namespace grading
