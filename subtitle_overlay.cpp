#include "subtitle_overlay.hpp"

#include "engine2d.h"
#include "fonts.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace
{
void skipWhitespace(const char*& ptr, const char* end)
{
    while (ptr < end && std::isspace(static_cast<unsigned char>(*ptr)))
    {
        ++ptr;
    }
}

bool expectChar(const char*& ptr, const char* end, char expected)
{
    skipWhitespace(ptr, end);
    if (ptr >= end || *ptr != expected)
    {
        return false;
    }
    ++ptr;
    return true;
}

bool parseJsonString(const char*& ptr, const char* end, std::string& out)
{
    if (!expectChar(ptr, end, '"'))
    {
        return false;
    }
    const char* start = ptr;
    std::string result;
    while (ptr < end)
    {
        char ch = *ptr;
        if (ch == '\\' && ptr + 1 < end)
        {
            ++ptr;
            char escaped = *ptr;
            switch (escaped)
            {
                case '\\':
                case '/':
                case '"':
                    result += escaped;
                    break;
                case 'b':
                    result += '\b';
                    break;
                case 'f':
                    result += '\f';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 'r':
                    result += '\r';
                    break;
                case 't':
                    result += '\t';
                    break;
                default:
                    result += escaped;
                    break;
            }
            ++ptr;
            continue;
        }
        if (ch == '"')
        {
            break;
        }
        result += ch;
        ++ptr;
    }
    if (ptr >= end || *ptr != '"')
    {
        return false;
    }
    ++ptr;
    out = std::move(result);
    return true;
}

bool parseNumber(const char*& ptr, const char* end, double& out)
{
    skipWhitespace(ptr, end);
    if (ptr >= end)
    {
        return false;
    }
    char* parseEnd = nullptr;
    errno = 0;
    double value = std::strtod(ptr, &parseEnd);
    if (parseEnd == ptr)
    {
        return false;
    }
    ptr = parseEnd;
    out = value;
    return true;
}

bool skipValue(const char*& ptr, const char* end);

bool skipObject(const char*& ptr, const char* end)
{
    if (!expectChar(ptr, end, '{'))
    {
        return false;
    }
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == '}')
        {
            ++ptr;
            return true;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        std::string key;
        if (!parseJsonString(ptr, end, key))
        {
            return false;
        }
        if (!expectChar(ptr, end, ':'))
        {
            return false;
        }
        if (!skipValue(ptr, end))
        {
            return false;
        }
    }
    return true;
}

bool skipArray(const char*& ptr, const char* end)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == ']')
        {
            ++ptr;
            return true;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        if (!skipValue(ptr, end))
        {
            return false;
        }
    }
    return true;
}

bool skipValue(const char*& ptr, const char* end)
{
    skipWhitespace(ptr, end);
    if (ptr >= end)
    {
        return false;
    }
    char ch = *ptr;
    if (ch == '{')
    {
        return skipObject(ptr, end);
    }
    if (ch == '[')
    {
        return skipArray(ptr, end);
    }
    if (ch == '"')
    {
        std::string dummy;
        return parseJsonString(ptr, end, dummy);
    }
    if (ch == '-' || ch == '+' || (ch >= '0' && ch <= '9'))
    {
        double dummy;
        return parseNumber(ptr, end, dummy);
    }
    if (std::strncmp(ptr, "true", 4) == 0)
    {
        ptr += 4;
        return true;
    }
    if (std::strncmp(ptr, "false", 5) == 0)
    {
        ptr += 5;
        return true;
    }
    if (std::strncmp(ptr, "null", 4) == 0)
    {
        ptr += 4;
        return true;
    }
    return false;
}

struct SubtitleWord
{
    double start = 0.0;
    double end = 0.0;
    std::string text;
};

struct SubtitleSegment
{
    double start = 0.0;
    double end = 0.0;
    std::string text;
    std::vector<SubtitleWord> words;
};

bool parseWordObject(const char*& ptr, const char* end, SubtitleWord& word)
{
    if (!expectChar(ptr, end, '{'))
    {
        return false;
    }
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == '}')
        {
            ++ptr;
            break;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        std::string key;
        if (!parseJsonString(ptr, end, key))
        {
            return false;
        }
        if (!expectChar(ptr, end, ':'))
        {
            return false;
        }
        if (key == "word")
        {
            parseJsonString(ptr, end, word.text);
        }
        else if (key == "start")
        {
            parseNumber(ptr, end, word.start);
        }
        else if (key == "end")
        {
            parseNumber(ptr, end, word.end);
        }
        else
        {
            if (!skipValue(ptr, end))
            {
                return false;
            }
        }
    }
    return !word.text.empty();
}

bool parseWordsArray(const char*& ptr, const char* end, std::vector<SubtitleWord>& words)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == ']')
        {
            ++ptr;
            break;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        SubtitleWord word;
        if (!parseWordObject(ptr, end, word))
        {
            skipValue(ptr, end);
            continue;
        }
        words.push_back(std::move(word));
    }
    return true;
}

bool parseSegmentObject(const char*& ptr, const char* end, SubtitleSegment& segment)
{
    if (!expectChar(ptr, end, '{'))
    {
        return false;
    }
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == '}')
        {
            ++ptr;
            break;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        std::string key;
        if (!parseJsonString(ptr, end, key))
        {
            return false;
        }
        if (!expectChar(ptr, end, ':'))
        {
            return false;
        }
        if (key == "start")
        {
            parseNumber(ptr, end, segment.start);
        }
        else if (key == "end")
        {
            parseNumber(ptr, end, segment.end);
        }
        else if (key == "text")
        {
            parseJsonString(ptr, end, segment.text);
        }
        else if (key == "words")
        {
            segment.words.clear();
            parseWordsArray(ptr, end, segment.words);
        }
        else
        {
            if (!skipValue(ptr, end))
            {
                return false;
            }
        }
    }
    return true;
}

bool parseSegmentsArray(const char*& ptr, const char* end, std::vector<SubtitleSegment>& segments)
{
    if (!expectChar(ptr, end, '['))
    {
        return false;
    }
    while (true)
    {
        skipWhitespace(ptr, end);
        if (ptr >= end)
        {
            return false;
        }
        if (*ptr == ']')
        {
            ++ptr;
            break;
        }
        if (*ptr == ',')
        {
            ++ptr;
            continue;
        }
        SubtitleSegment segment;
        if (!parseSegmentObject(ptr, end, segment))
        {
            skipValue(ptr, end);
            continue;
        }
        segments.push_back(std::move(segment));
    }
    return true;
}

bool findSegmentsArray(const std::string& text, const char*& ptr, const char* end)
{
    const char* base = text.data();
    const char* key = std::strstr(base, "\"segments\"");
    if (!key)
    {
        return false;
    }
    ptr = key + std::strlen("\"segments\"");
    skipWhitespace(ptr, end);
    if (!expectChar(ptr, end, ':'))
    {
        return false;
    }
    skipWhitespace(ptr, end);
    if (ptr >= end || *ptr != '[')
    {
        return false;
    }
    return true;
}

std::vector<SubtitleOverlay::Line> buildLines(const std::vector<SubtitleSegment>& segments)
{
    constexpr size_t kMaxWordsPerLine = 7;
    constexpr size_t kMaxCharsPerLine = 64;

    std::vector<SubtitleOverlay::Line> lines;
    lines.reserve(segments.size() * 2);

    for (const auto& segment : segments)
    {
        if (!segment.words.empty())
        {
            size_t index = 0;
            while (index < segment.words.size())
            {
                size_t endIndex = index;
                size_t chars = segment.words[endIndex].text.size();
                double startTime = segment.words[endIndex].start;
                double endTime = segment.words[endIndex].end;
                std::string text = segment.words[endIndex].text;
                ++endIndex;
                size_t words = 1;
                while (endIndex < segment.words.size())
                {
                    const auto& nextWord = segment.words[endIndex];
                    size_t nextLen = nextWord.text.size();
                    if (words >= kMaxWordsPerLine || chars + 1 + nextLen > kMaxCharsPerLine)
                    {
                        break;
                    }
                    text += " ";
                    text += nextWord.text;
                    chars += 1 + nextLen;
                    endTime = nextWord.end;
                    ++endIndex;
                    ++words;
                }
                lines.push_back({startTime, endTime, std::move(text)});
                index = endIndex;
            }
        }
        else if (!segment.text.empty())
        {
            lines.push_back({segment.start, segment.end, segment.text});
        }
    }

    std::sort(lines.begin(), lines.end(), [](const SubtitleOverlay::Line& a, const SubtitleOverlay::Line& b) {
        return a.start < b.start;
    });
    return lines;
}

fonts::FontBitmap prepareLineBitmap(const std::string& text, uint32_t fontSize)
{
    return fonts::renderText(text, fontSize);
}

void blendPixels(uint8_t* dest,
                 uint32_t destWidth,
                 uint32_t destHeight,
                 uint8_t* src,
                 uint32_t srcWidth,
                 uint32_t srcHeight,
                 uint32_t destX,
                 uint32_t destY)
{
    if (!dest || !src)
    {
        return;
    }
    if (destX >= destWidth || destY >= destHeight)
    {
        return;
    }
    uint32_t maxWidth = std::min(destWidth - destX, srcWidth);
    uint32_t maxHeight = std::min(destHeight - destY, srcHeight);

    for (uint32_t row = 0; row < maxHeight; ++row)
    {
        for (uint32_t col = 0; col < maxWidth; ++col)
        {
            size_t srcIdx = (static_cast<size_t>(row) * srcWidth + col) * 4;
            uint8_t srcA = src[srcIdx + 3];
            if (srcA == 0)
            {
                continue;
            }
            size_t dstIdx = (static_cast<size_t>(destY + row) * destWidth + (destX + col)) * 4;
            uint8_t dstA = dest[dstIdx + 3];
            float srcAF = static_cast<float>(srcA) / 255.0f;
            float dstAF = static_cast<float>(dstA) / 255.0f;
            float outA = srcAF + dstAF * (1.0f - srcAF);
            if (outA <= 0.0f)
            {
                continue;
            }
            auto blendChannel = [&](uint8_t srcC, uint8_t dstC) -> uint8_t {
                float srcCF = static_cast<float>(srcC) / 255.0f;
                float dstCF = static_cast<float>(dstC) / 255.0f;
                float outCF = (srcCF * srcAF + dstCF * dstAF * (1.0f - srcAF)) / outA;
                return static_cast<uint8_t>(std::clamp(outCF * 255.0f, 0.0f, 255.0f));
            };
            dest[dstIdx + 0] = blendChannel(src[srcIdx + 0], dest[dstIdx + 0]);
            dest[dstIdx + 1] = blendChannel(src[srcIdx + 1], dest[dstIdx + 1]);
            dest[dstIdx + 2] = blendChannel(src[srcIdx + 2], dest[dstIdx + 2]);
            dest[dstIdx + 3] = static_cast<uint8_t>(std::clamp(outA * 255.0f, 0.0f, 255.0f));
        }
    }
}
} // namespace

// SubtitleOverlay

bool SubtitleOverlay::load(const std::filesystem::path& path)
{
    if (path.empty() || !std::filesystem::exists(path))
    {
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }
    std::string contents;
    contents.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (contents.empty())
    {
        return false;
    }

    const char* ptr = contents.data();
    const char* end = ptr + contents.size();
    skipWhitespace(ptr, end);
    if (!findSegmentsArray(contents, ptr, end))
    {
        return false;
    }

    std::vector<SubtitleSegment> segments;
    if (!parseSegmentsArray(ptr, end, segments))
    {
        return false;
    }

    lines_ = buildLines(segments);
    lastIndex_ = 0;
    return !lines_.empty();
}

std::vector<const SubtitleOverlay::Line*> SubtitleOverlay::activeLines(double currentTime, size_t maxLines) const
{
    std::vector<const Line*> result;
    if (lines_.empty() || maxLines == 0)
    {
        lastIndex_ = 0;
        return result;
    }

    const double lookbehind = 0.3;
    const double lookahead = 0.5;
    size_t index = lastIndex_;
    if (index >= lines_.size())
    {
        index = lines_.size() ? lines_.size() - 1 : 0;
    }
    while (index > 0 && lines_[index].start > currentTime)
    {
        --index;
    }
    while (index < lines_.size() && lines_[index].end + lookbehind < currentTime)
    {
        ++index;
    }

    lastIndex_ = index;
    for (size_t i = index; i < lines_.size(); ++i)
    {
        const Line& line = lines_[i];
        if (line.start - lookahead > currentTime)
        {
            if (result.empty())
            {
                result.push_back(&line);
            }
            break;
        }
        if (line.end + lookbehind >= currentTime)
        {
            result.push_back(&line);
            if (result.size() >= maxLines)
            {
                break;
            }
        }
    }

    if (result.empty() && index < lines_.size())
    {
        result.push_back(&lines_[index]);
    }
    if (result.size() > maxLines)
    {
        result.resize(maxLines);
    }
    return result;
}

// Subtitle overlay resources

bool updateSubtitleOverlay(Engine2D* engine,
                           SubtitleOverlayResources& resources,
                           const SubtitleOverlay& overlay,
                           double currentTime,
                           uint32_t fbWidth,
                           uint32_t fbHeight,
                           glm::vec2 overlayCenter,
                           glm::vec2 overlaySize,
                           VkSampler overlaySampler,
                           VkSampler fallbackSampler,
                           size_t maxLines)
{
    resources.info.enabled = false;
    if (!engine || fbWidth == 0 || fbHeight == 0 || overlaySize.x <= 0.0f || overlaySize.y <= 0.0f || !overlay.hasData())
    {
        return false;
    }

    const std::vector<const SubtitleOverlay::Line*> candidateLines = overlay.activeLines(currentTime, maxLines);
    if (candidateLines.empty())
    {
        return false;
    }

    std::vector<const SubtitleOverlay::Line*> lines;
    lines.reserve(candidateLines.size());
    std::string lastText;
    for (const SubtitleOverlay::Line* line : candidateLines)
    {
        if (!line)
        {
            continue;
        }
        std::string text = line->text;
        if (text.empty() || text == lastText)
        {
            continue;
        }
        lastText = text;
        lines.push_back(line);
        if (lines.size() >= maxLines)
        {
            break;
        }
    }

    if (lines.empty())
    {
        return false;
    }

    const uint32_t overlayWidth = static_cast<uint32_t>(
        std::clamp(std::lround(overlaySize.x), 1l, static_cast<long>(fbWidth)));
    const uint32_t overlayHeight = static_cast<uint32_t>(
        std::clamp(std::lround(overlaySize.y), 1l, static_cast<long>(fbHeight)));
    if (overlayWidth == 0 || overlayHeight == 0)
    {
        return false;
    }

    const uint32_t padding = std::max<uint32_t>(8, overlayHeight / 20);
    const uint32_t innerWidth = overlayWidth > padding * 2 ? overlayWidth - padding * 2 : 0;
    const uint32_t innerHeight = overlayHeight > padding * 2 ? overlayHeight - padding * 2 : 0;
    if (innerWidth == 0 || innerHeight == 0)
    {
        return false;
    }

    uint32_t fontSize = std::max<uint32_t>(12, innerHeight / (static_cast<uint32_t>(lines.size()) + 1));
    fontSize = std::min(fontSize, innerHeight);
    std::vector<fonts::FontBitmap> bitmaps;
    bitmaps.reserve(lines.size());
    uint32_t finalLineSpacing = std::max<uint32_t>(2, fontSize / 2);
    uint32_t maxLineWidth = 0;
    uint32_t combinedHeight = 0;
    while (true)
    {
        bitmaps.clear();
        maxLineWidth = 0;
        combinedHeight = 0;
        for (const SubtitleOverlay::Line* line : lines)
        {
            fonts::FontBitmap bmp = prepareLineBitmap(line->text, fontSize);
            if (bmp.width == 0 || bmp.height == 0 || bmp.pixels.empty())
            {
                continue;
            }
            maxLineWidth = std::max(maxLineWidth, bmp.width);
            combinedHeight += bmp.height;
            bitmaps.push_back(std::move(bmp));
        }

        if (bitmaps.empty())
        {
            return false;
        }

        combinedHeight += (bitmaps.size() > 1 ? (static_cast<uint32_t>(bitmaps.size()) - 1) * finalLineSpacing : 0);
        if (maxLineWidth <= innerWidth && combinedHeight <= innerHeight)
        {
            break;
        }

        if (fontSize <= 12)
        {
            break;
        }

        --fontSize;
        finalLineSpacing = std::max<uint32_t>(2, fontSize / 2);
    }

    const uint32_t textWidth = std::min(maxLineWidth, innerWidth);
    const uint32_t textHeight = std::min(combinedHeight, innerHeight);
    if (textWidth == 0 || textHeight == 0)
    {
        return false;
    }
    const uint32_t texWidth = textWidth + padding * 2;
    const uint32_t texHeight = textHeight + padding * 2;
    const uint32_t pixelCount = texWidth * texHeight;
    std::vector<uint8_t> pixels(static_cast<size_t>(pixelCount) * 4);
    std::fill(pixels.begin(), pixels.end(), 0);
    uint32_t yOffset = padding;
    for (size_t idx = 0; idx < bitmaps.size(); ++idx)
    {
        const auto& bmp = bitmaps[idx];
        if (bmp.pixels.empty())
        {
            continue;
        }

        uint32_t destX = padding;
        if (bmp.width < innerWidth)
        {
            destX += (innerWidth - bmp.width) / 2;
        }
        blendPixels(pixels.data(),
                    overlayWidth,
                    overlayHeight,
                    const_cast<uint8_t*>(bmp.pixels.data()),
                    bmp.width,
                    bmp.height,
                    destX,
                    yOffset);
        yOffset += bmp.height;
        if (idx + 1 < bitmaps.size())
        {
            yOffset += finalLineSpacing;
        }
    }

    if (!overlay::uploadImageData(engine,
                                  resources.image,
                                  pixels.data(),
                                  pixels.size(),
                                  texWidth,
                                  texHeight,
                                  VK_FORMAT_R8G8B8A8_UNORM))
    {
        return false;
    }

    const int32_t maxOffsetX =
        static_cast<int32_t>(std::max<uint32_t>(0u, fbWidth > texWidth ? fbWidth - texWidth : 0u));
    const int32_t maxOffsetY =
        static_cast<int32_t>(std::max<uint32_t>(0u, fbHeight > texHeight ? fbHeight - texHeight : 0u));
    const float halfWidth = overlaySize.x * 0.5f;
    const float halfHeight = overlaySize.y * 0.5f;
    int32_t offsetX =
        static_cast<int32_t>(std::lround(overlayCenter.x - halfWidth));
    int32_t offsetY =
        static_cast<int32_t>(std::lround(overlayCenter.y - halfHeight));
    offsetX = std::clamp(offsetX, 0, maxOffsetX);
    offsetY = std::clamp(offsetY, 0, maxOffsetY);

    VkSampler sampler = overlaySampler != VK_NULL_HANDLE ? overlaySampler : fallbackSampler;
    resources.info.overlay.view = resources.image.view;
    resources.info.overlay.sampler = sampler;
    resources.info.extent = {texWidth, texHeight};
    resources.info.offset = {offsetX, offsetY};
    resources.info.enabled = true;
    return true;
}

void destroySubtitleOverlayResources(Engine2D* engine, SubtitleOverlayResources& resources)
{
    if (engine)
    {
        overlay::destroyImageResource(engine, resources.image);
    }
    resources.info = {};
}
