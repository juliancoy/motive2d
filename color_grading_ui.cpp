#include "color_grading_ui.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <limits>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "engine2d.h"
#include "utils.h"
#include "subtitle.h"
#include "fps.h"
#include "widgets.hpp"

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

glm::vec4 rgbColor(uint32_t r, uint32_t g, uint32_t b, float alpha = 1.0f)
{
    return glm::vec4(static_cast<float>(r) / 255.0f,
                     static_cast<float>(g) / 255.0f,
                     static_cast<float>(b) / 255.0f,
                     alpha);
}

widgets::WidgetRenderer &gradingWidgetRenderer()
{
    static widgets::WidgetRenderer renderer{};
    return renderer;
}

bool ensureGradingWidgetRenderer(Engine2D *engine)
{
    static bool initialized = false;
    static bool failed = false;
    if (initialized)
    {
        return true;
    }
    if (failed)
    {
        return false;
    }
    if (!widgets::initializeWidgetRenderer(engine, gradingWidgetRenderer()))
    {
        failed = true;
        return false;
    }
    initialized = true;
    return true;
}

ColorGradingUi::ColorGradingUi(Engine2D *engine,
                               const GradingSettings &settings,
                               ImageResource &image,
                               OverlayImageInfo &info,
                               uint32_t fbWidth,
                               uint32_t fbHeight,
                               SliderLayout &layout,
                               bool previewEnabled,
                               bool detectionEnabled)
{

    constexpr float kBaseWidth = 420.0f;
    constexpr float kBaseHeight = 880.0f;
    const float scaleW = static_cast<float>(fbWidth) / kBaseWidth;
    const float scaleH = static_cast<float>(fbHeight) / kBaseHeight;
    const float scale = std::min(1.0f, std::min(scaleW, scaleH));
    const auto scaledValue = [&](float value) -> uint32_t
    {
        return std::max<uint32_t>(1, static_cast<uint32_t>(std::round(value * scale)));
    };
    layout.width = std::max<uint32_t>(1, static_cast<uint32_t>(std::round(kBaseWidth * scale)));
    layout.height = std::max<uint32_t>(1, static_cast<uint32_t>(std::round(kBaseHeight * scale)));
    layout.margin = scaledValue(20.0f);
    layout.barHeight = scaledValue(20.0f);
    layout.curvesPadding = scaledValue(16.0f);
    layout.curvesHeight = scaledValue(200.0f);
    layout.curvesX0 = layout.curvesPadding;
    layout.curvesX1 = layout.width > layout.curvesPadding * 2
                          ? layout.width - layout.curvesPadding
                          : layout.width;
    layout.curvesY0 = layout.curvesPadding;
    layout.curvesY1 = layout.curvesY0 + layout.curvesHeight;
    layout.barYStart = layout.curvesY1 + scaledValue(32.0f);
    layout.rowSpacing = scaledValue(12.0f);
    layout.handleRadius = scaledValue(10.0f);
    layout.resetWidth = scaledValue(140.0f);
    layout.resetHeight = scaledValue(36.0f);
    layout.loadWidth = layout.resetWidth;
    layout.loadHeight = layout.resetHeight;
    layout.saveWidth = layout.resetWidth;
    layout.saveHeight = layout.resetHeight;
    layout.previewWidth = scaledValue(160.0f);
    layout.previewHeight = scaledValue(36.0f);
    layout.detectionWidth = scaledValue(180.0f);
    layout.detectionHeight = scaledValue(36.0f);

    const uint32_t padding = scaledValue(12.0f);
    const uint32_t barWidth = layout.width > padding * 2 ? layout.width - padding * 2 : 0;
    const uint32_t barYStart = layout.barYStart;

    struct SliderSpec
    {
        float value;
        glm::vec4 color;
    };
    const SliderSpec sliderSpecs[12] = {
        {normalizeExposure(settings.exposure), rgbColor(255, 180, 80)},
        {normalizeContrast(settings.contrast), rgbColor(80, 200, 255)},
        {normalizeSaturation(settings.saturation), rgbColor(180, 255, 160)},
        {clamp01(settings.shadows.r * 0.5f), rgbColor(255, 120, 120)},
        {clamp01(settings.shadows.g * 0.5f), rgbColor(120, 255, 120)},
        {clamp01(settings.shadows.b * 0.5f), rgbColor(120, 120, 255)},
        {clamp01(settings.midtones.r * 0.5f), rgbColor(255, 120, 120)},
        {clamp01(settings.midtones.g * 0.5f), rgbColor(120, 255, 120)},
        {clamp01(settings.midtones.b * 0.5f), rgbColor(120, 120, 255)},
        {clamp01(settings.highlights.r * 0.5f), rgbColor(255, 120, 120)},
        {clamp01(settings.highlights.g * 0.5f), rgbColor(120, 255, 120)},
        {clamp01(settings.highlights.b * 0.5f), rgbColor(120, 120, 255)},
    };

    const float overlayWidthF = static_cast<float>(layout.width);
    const float overlayHeightF = static_cast<float>(layout.height);
    std::vector<widgets::DrawCommand> commands;
    commands.reserve(256);

    auto pushRect = [&](float x0, float y0, float x1, float y1, const glm::vec4 &color)
    {
        float widthF = x1 - x0;
        float heightF = y1 - y0;
        widgets::DrawCommand cmd{};
        cmd.type = widgets::CMD_RECT;
        cmd.color = color;
        cmd.params = glm::vec4((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, widthF, heightF);
        commands.push_back(cmd);
    };

    auto pushLine = [&](const glm::vec2 &start, const glm::vec2 &end, float thickness, const glm::vec4 &color)
    {
        widgets::DrawCommand cmd{};
        cmd.type = widgets::CMD_LINE;
        cmd.color = color;
        cmd.params = glm::vec4(start, end);
        cmd.params2.x = thickness;
        commands.push_back(cmd);
    };

    auto pushCircle = [&](const glm::vec2 &center, float radius, const glm::vec4 &color)
    {
        widgets::DrawCommand cmd{};
        cmd.type = widgets::CMD_CIRCLE;
        cmd.color = color;
        cmd.params = glm::vec4(center, radius, 0.0f);
        commands.push_back(cmd);
    };

    pushRect(0.0f, 0.0f, overlayWidthF, overlayHeightF, glm::vec4(0.0f, 0.0f, 0.0f, 0.72f));

    const glm::vec4 curvesBg = glm::vec4(0.14f, 0.14f, 0.14f, 1.0f);
    pushRect(static_cast<float>(layout.curvesX0),
             static_cast<float>(layout.curvesY0),
             static_cast<float>(layout.curvesX1),
             static_cast<float>(layout.curvesY1),
             curvesBg);

    const float curvesWidth = static_cast<float>(layout.curvesX1 - layout.curvesX0);
    const float curvesHeight = static_cast<float>(layout.curvesY1 - layout.curvesY0);
    const glm::vec4 gridColor(0.22f, 0.22f, 0.22f, 1.0f);
    for (int i = 1; i < 4; ++i)
    {
        float x = static_cast<float>(layout.curvesX0) + (static_cast<float>(i) / 4.0f) * curvesWidth;
        pushLine(glm::vec2(x, static_cast<float>(layout.curvesY0)),
                 glm::vec2(x, static_cast<float>(layout.curvesY1)),
                 1.5f,
                 gridColor);
    }
    for (int i = 1; i < 4; ++i)
    {
        float y = static_cast<float>(layout.curvesY0) + (static_cast<float>(i) / 4.0f) * curvesHeight;
        pushLine(glm::vec2(static_cast<float>(layout.curvesX0), y),
                 glm::vec2(static_cast<float>(layout.curvesX1), y),
                 1.5f,
                 gridColor);
    }

    const glm::vec4 axisColor(0.35f, 0.35f, 0.35f, 1.0f);
    pushLine(glm::vec2(static_cast<float>(layout.curvesX0), static_cast<float>(layout.curvesY1)),
             glm::vec2(static_cast<float>(layout.curvesX1), static_cast<float>(layout.curvesY1)),
             2.0f,
             axisColor);
    pushLine(glm::vec2(static_cast<float>(layout.curvesX0), static_cast<float>(layout.curvesY0)),
             glm::vec2(static_cast<float>(layout.curvesX0), static_cast<float>(layout.curvesY1)),
             2.0f,
             axisColor);

    auto curveToPixel = [&](const glm::vec2 &point) -> glm::vec2
    {
        glm::vec2 clamped = {clamp01(point.x), clamp01(point.y)};
        float x = static_cast<float>(layout.curvesX0) + clamped.x * curvesWidth;
        float y = static_cast<float>(layout.curvesY1) - clamped.y * curvesHeight;
        return glm::vec2(x, y);
    };
    if (settings.curves.size() >= 2)
    {
        glm::vec2 previous = curveToPixel(settings.curves.front());
        for (size_t i = 1; i < settings.curves.size(); ++i)
        {
            glm::vec2 current = curveToPixel(settings.curves[i]);
            pushLine(previous, current, 3.0f, glm::vec4(0.9f, 0.9f, 0.9f, 1.0f));
            previous = current;
        }
        for (size_t i = 0; i < settings.curves.size(); ++i)
        {
            glm::vec2 center = curveToPixel(settings.curves[i]);
            float handleAlpha = (i == 0 || i + 1 == settings.curves.size()) ? 0.6f : 1.0f;
            pushCircle(center, static_cast<float>(layout.handleRadius) * 0.6f, glm::vec4(1.0f, 1.0f, 1.0f, handleAlpha));
            pushCircle(center, static_cast<float>(layout.handleRadius) * 0.6f + 2.0f, glm::vec4(0.0f, 0.0f, 0.0f, 0.65f));
        }
    }

    for (int i = 0; i < 12; ++i)
    {
        uint32_t y0 = barYStart + i * (layout.barHeight + layout.rowSpacing);
        float centerY = static_cast<float>(y0) + static_cast<float>(layout.barHeight) * 0.5f;

        widgets::SliderDescriptor sliderDesc{};
        sliderDesc.start = glm::vec2(static_cast<float>(padding), centerY);
        sliderDesc.end = glm::vec2(static_cast<float>(padding + barWidth), centerY);
        sliderDesc.trackThickness = static_cast<float>(layout.barHeight);
        sliderDesc.trackColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);
        sliderDesc.progressColor = sliderSpecs[i].color;
        sliderDesc.value = sliderSpecs[i].value;
        sliderDesc.handleRadius = static_cast<float>(layout.handleRadius);
        sliderDesc.handleColor = glm::vec4(0.95f, 0.95f, 0.95f, 1.0f);
        sliderDesc.handleBorderColor = glm::vec4(0.0f, 0.0f, 0.0f, 0.85f);
        widgets::appendSliderCommands(commands, sliderDesc);
    }

    // Reset button near the bottom
    const uint32_t buttonPadding = scaledValue(12.0f);
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
    auto centerButtonX = [&](uint32_t w)
    {
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

    const glm::vec4 buttonBaseColor = glm::vec4(0.23f, 0.23f, 0.23f, 1.0f);
    const glm::vec4 previewActiveColor = glm::vec4(0.18f, 0.6f, 0.95f, 1.0f);
    const glm::vec4 detectionActiveColor = glm::vec4(0.26f, 0.7f, 0.35f, 1.0f);
    auto emitButton = [&](uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, const glm::vec4 &color)
    {
        widgets::ButtonDescriptor desc{};
        desc.center = glm::vec2(static_cast<float>(x0) + static_cast<float>(w) * 0.5f,
                                static_cast<float>(y0) + static_cast<float>(h) * 0.5f);
        desc.size = glm::vec2(static_cast<float>(w), static_cast<float>(h));
        desc.borderThickness = 3.0f;
        desc.backgroundColor = color;
        desc.borderColor = glm::vec4(0.06f, 0.06f, 0.06f, 1.0f);
        widgets::appendButtonCommands(commands, desc);
    };

    emitButton(layout.resetX0, layout.resetY0, layout.resetWidth, layout.resetHeight, buttonBaseColor);
    emitButton(layout.loadX0, layout.loadY0, layout.loadWidth, layout.loadHeight, buttonBaseColor);
    emitButton(layout.saveX0, layout.saveY0, layout.saveWidth, layout.saveHeight, buttonBaseColor);
    emitButton(layout.previewX0,
               layout.previewY0,
               layout.previewWidth,
               layout.previewHeight,
               previewEnabled ? previewActiveColor : buttonBaseColor);
    emitButton(layout.detectionX0,
               layout.detectionY0,
               layout.detectionWidth,
               layout.detectionHeight,
               detectionEnabled ? detectionActiveColor : buttonBaseColor);

    const uint32_t overlayX = (fbWidth > layout.width) ? (fbWidth - layout.width) / 2 : 0;
    const uint32_t overlayY = (fbHeight > layout.height + layout.margin) ? fbHeight - layout.height - layout.margin : 0;
    layout.offset = {static_cast<int32_t>(overlayX), static_cast<int32_t>(overlayY)};

    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    bool recreated = false;
    ImageResource(engine,
                  image,
                  layout.width,
                  layout.height,
                  VK_FORMAT_R8G8B8A8_UNORM,
                  recreated,
                  usage);

    if (!widgets::runWidgetRenderer(engine,
                                    gradingWidgetRenderer(),
                                    image,
                                    layout.width,
                                    layout.height,
                                    commands,
                                    true))
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

bool ColorGradingUi::handleOverlayClick(const SliderLayout &layout,
                                        double cursorX,
                                        double cursorY,
                                        GradingSettings &settings,
                                        bool doubleClick,
                                        bool rightClick,
                                        bool *loadRequested,
                                        bool *saveRequested,
                                        bool *previewToggleRequested,
                                        bool *detectionToggleRequested)
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
        case 0:
            settings.exposure = 0.0f;
            break;
        case 1:
            settings.contrast = 1.0f;
            break;
        case 2:
            settings.saturation = 1.0f;
            break;
        case 3:
            settings.shadows.r = 1.0f;
            break;
        case 4:
            settings.shadows.g = 1.0f;
            break;
        case 5:
            settings.shadows.b = 1.0f;
            break;
        case 6:
            settings.midtones.r = 1.0f;
            break;
        case 7:
            settings.midtones.g = 1.0f;
            break;
        case 8:
            settings.midtones.b = 1.0f;
            break;
        case 9:
            settings.highlights.r = 1.0f;
            break;
        case 10:
            settings.highlights.g = 1.0f;
            break;
        case 11:
            settings.highlights.b = 1.0f;
            break;
        default:
            break;
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

void ColorGradingUi::setGradingDefaults(GradingSettings &settings)
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

void buildCurveLut(const GradingSettings &settings, std::array<float, kCurveLutSize> &outLut)
{
    auto clamp01f = [](float v)
    { return std::clamp(v, 0.0f, 1.0f); };
    if (settings.curves.size() < 2)
    {
        for (size_t i = 0; i < kCurveLutSize; ++i)
        {
            outLut[i] = static_cast<float>(i) / static_cast<float>(kCurveLutSize - 1);
        }
        return;
    }

    std::vector<glm::vec2> pts = settings.curves;
    std::sort(pts.begin(), pts.end(), [](const glm::vec2 &a, const glm::vec2 &b)
              { return a.x < b.x; });
    pts.front().x = 0.0f;
    pts.back().x = 1.0f;
    for (auto &p : pts)
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
