#include "video_frame_utils.h"

#include <algorithm>
#include <cstring>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

namespace video {
namespace {

struct PlanarFormatConfig {
    AVPixelFormat pixelFormat;
    uint32_t chromaDivX;
    uint32_t chromaDivY;
    uint32_t bitDepth;
    PrimitiveYuvFormat outputFormat;
    bool chromaInterleaved;
};

constexpr PlanarFormatConfig kPlanarFormats[] = {
    {AV_PIX_FMT_YUV420P, 2, 2, 8, PrimitiveYuvFormat::Planar420, false},
    {AV_PIX_FMT_YUV420P10LE, 2, 2, 10, PrimitiveYuvFormat::Planar420, false},
    {AV_PIX_FMT_YUV420P12LE, 2, 2, 12, PrimitiveYuvFormat::Planar420, false},
    {AV_PIX_FMT_YUV422P, 2, 1, 8, PrimitiveYuvFormat::Planar422, false},
    {AV_PIX_FMT_YUV422P10LE, 2, 1, 10, PrimitiveYuvFormat::Planar422, false},
    {AV_PIX_FMT_YUV422P12LE, 2, 1, 12, PrimitiveYuvFormat::Planar422, false},
    {AV_PIX_FMT_YUV444P, 1, 1, 8, PrimitiveYuvFormat::Planar444, false},
    {AV_PIX_FMT_YUV444P10LE, 1, 1, 10, PrimitiveYuvFormat::Planar444, false},
    {AV_PIX_FMT_YUV444P12LE, 1, 1, 12, PrimitiveYuvFormat::Planar444, false},
#if defined(AV_PIX_FMT_P010)
    {AV_PIX_FMT_P010, 2, 2, 10, PrimitiveYuvFormat::Planar420, true},
#endif
#if defined(AV_PIX_FMT_P012)
    {AV_PIX_FMT_P012, 2, 2, 12, PrimitiveYuvFormat::Planar420, true},
#endif
#if defined(AV_PIX_FMT_P016)
    {AV_PIX_FMT_P016, 2, 2, 16, PrimitiveYuvFormat::Planar420, true},
#endif
};

bool configureNv12Format(VideoDecoder& decoder, AVPixelFormat format)
{
    if (format != AV_PIX_FMT_NV12
#if defined(AV_PIX_FMT_NV21)
        && format != AV_PIX_FMT_NV21
#endif
    ) {
        return false;
    }
    decoder.planarYuv = false;
    decoder.chromaInterleaved = false;
    decoder.outputFormat = PrimitiveYuvFormat::NV12;
    decoder.bytesPerComponent = 1;
    decoder.bitDepth = 8;
    decoder.swapChromaUV =
#if defined(AV_PIX_FMT_NV21)
        (format == AV_PIX_FMT_NV21);
#else
        false;
#endif
    decoder.chromaDivX = 2;
    decoder.chromaDivY = 2;
    decoder.chromaWidth = std::max<uint32_t>(1u, (decoder.width + decoder.chromaDivX - 1) / decoder.chromaDivX);
    decoder.chromaHeight = std::max<uint32_t>(1u, (decoder.height + decoder.chromaDivY - 1) / decoder.chromaDivY);
    decoder.yPlaneSize = decoder.width * decoder.height;
    decoder.uvPlaneSize = decoder.yPlaneSize / 2;
    decoder.yPlaneBytes = static_cast<size_t>(decoder.yPlaneSize);
    decoder.uvPlaneBytes = static_cast<size_t>(decoder.uvPlaneSize);
    decoder.bufferSize = av_image_get_buffer_size(AV_PIX_FMT_NV12, decoder.width, decoder.height, 1);
    return true;
}

bool configurePlanarFormat(VideoDecoder& decoder, AVPixelFormat pixFormat)
{
    for (const auto& config : kPlanarFormats) {
        if (pixFormat != config.pixelFormat) {
            continue;
        }
        if (config.chromaDivX == 0 || config.chromaDivY == 0) {
            continue;
        }

        decoder.planarYuv = true;
        decoder.swapChromaUV = false;
        decoder.outputFormat = config.outputFormat;
        decoder.chromaDivX = config.chromaDivX;
        decoder.chromaDivY = config.chromaDivY;
        decoder.chromaInterleaved = config.chromaInterleaved;
        decoder.bytesPerComponent = config.bitDepth > 8 ? 2 : 1;
        decoder.bitDepth = config.bitDepth;
        decoder.chromaWidth = std::max<uint32_t>(1u, (decoder.width + decoder.chromaDivX - 1) / decoder.chromaDivX);
        decoder.chromaHeight = std::max<uint32_t>(1u, (decoder.height + decoder.chromaDivY - 1) / decoder.chromaDivY);
        decoder.yPlaneSize = decoder.width * decoder.height;
        decoder.uvPlaneSize = decoder.chromaWidth * decoder.chromaHeight;
        decoder.yPlaneBytes = static_cast<size_t>(decoder.width) * decoder.height * decoder.bytesPerComponent;
        decoder.uvPlaneBytes = static_cast<size_t>(decoder.chromaWidth) *
                               decoder.chromaHeight *
                               decoder.bytesPerComponent * 2;
        decoder.bufferSize = static_cast<int>(decoder.yPlaneBytes + decoder.uvPlaneBytes);
        return true;
    }
    return false;
}

void copyNv12Plane(uint8_t* dst,
                   const uint8_t* src,
                   int width,
                   int height,
                   int dstStride,
                   int srcStride)
{
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dstStride,
                    src + static_cast<size_t>(y) * srcStride,
                    width);
    }
}

void copyNv12FrameToBuffer(const AVFrame* frame,
                           std::vector<uint8_t>& buffer,
                           int width,
                           int height,
                           bool swapChromaUV)
{
    const size_t yPlaneSize = static_cast<size_t>(width) * height;
    const size_t uvPlaneSize = yPlaneSize / 2;
    const size_t requiredSize = yPlaneSize + uvPlaneSize;
    if (buffer.size() != requiredSize) {
        buffer.resize(requiredSize);
    }

    uint8_t* dstY = buffer.data();
    uint8_t* dstUV = buffer.data() + yPlaneSize;
    copyNv12Plane(dstY, frame->data[0], width, height, width, frame->linesize[0]);
    copyNv12Plane(dstUV, frame->data[1], width, height / 2, width, frame->linesize[1]);
    if (swapChromaUV) {
        for (size_t i = 0; i + 1 < uvPlaneSize; i += 2) {
            std::swap(dstUV[i], dstUV[i + 1]);
        }
    }
}

void copyPlanarFrameToBuffer(const VideoDecoder& decoder,
                             AVFrame* frame,
                             std::vector<uint8_t>& buffer)
{
    const uint32_t width = decoder.width;
    const uint32_t height = decoder.height;
    const uint32_t chromaWidth = decoder.chromaWidth;
    const uint32_t chromaHeight = decoder.chromaHeight;
    const size_t bytesPerComponent = decoder.bytesPerComponent;

    if (width == 0 || height == 0 || chromaWidth == 0 || chromaHeight == 0) {
        return;
    }

    if (buffer.size() != static_cast<size_t>(decoder.bufferSize)) {
        buffer.resize(static_cast<size_t>(decoder.bufferSize));
    }

    // DEBUG: Print frame info
    static int frameCount = 0;
    frameCount++;

    uint8_t* dstY = buffer.data();
    uint64_t ySum = 0;
    uint64_t uSum = 0;
    uint64_t vSum = 0;
    uint32_t sampleCount = 0;
    const size_t lumaRowBytes = static_cast<size_t>(width) * bytesPerComponent;
    const bool interleaved = decoder.chromaInterleaved;
    const uint32_t bitDepth = decoder.bitDepth > 0 ? decoder.bitDepth : 8;
    const uint32_t shift = (interleaved || bitDepth >= 16) ? 0u : 16u - bitDepth;
    if (bytesPerComponent == 1) {
        for (uint32_t row = 0; row < height; ++row) {
            const uint8_t* srcRow = frame->data[0] + static_cast<size_t>(row) * frame->linesize[0];
            std::memcpy(dstY + static_cast<size_t>(row) * lumaRowBytes, srcRow, lumaRowBytes);
            // Calculate sum for debugging
            for (uint32_t col = 0; col < width; ++col) {
                ySum += srcRow[col];
                sampleCount++;
            }
        }
    } else {
        uint16_t* dstY16 = reinterpret_cast<uint16_t*>(dstY);
        for (uint32_t row = 0; row < height; ++row) {
            const uint16_t* srcRow = reinterpret_cast<const uint16_t*>(frame->data[0] + static_cast<size_t>(row) * frame->linesize[0]);
            uint16_t* dstRow = dstY16 + static_cast<size_t>(row) * width;
            for (uint32_t col = 0; col < width; ++col) {
                uint16_t value = srcRow[col];
                dstRow[col] = shift ? static_cast<uint16_t>(value << shift) : value;
                ySum += value;
                sampleCount++;
            }
        }
    }

    uint8_t* dstUv = buffer.data() + decoder.yPlaneBytes;
    if (bytesPerComponent == 1) {
        if (!interleaved) {
            for (uint32_t row = 0; row < chromaHeight; ++row) {
                const uint8_t* uRow = frame->data[1] + static_cast<size_t>(row) * frame->linesize[1];
                const uint8_t* vRow = frame->data[2] + static_cast<size_t>(row) * frame->linesize[2];
                uint8_t* dstRow = dstUv + static_cast<size_t>(row) * chromaWidth * 2;
                for (uint32_t col = 0; col < chromaWidth; ++col) {
                    dstRow[col * 2 + 0] = uRow[col];
                    dstRow[col * 2 + 1] = vRow[col];
                    uSum += uRow[col];
                    vSum += vRow[col];
                }
            }
        } else {
            const size_t chromaRowBytes = static_cast<size_t>(chromaWidth) * 2;
            for (uint32_t row = 0; row < chromaHeight; ++row) {
                const uint8_t* srcRow = frame->data[1] + static_cast<size_t>(row) * frame->linesize[1];
                uint8_t* dstRow = dstUv + static_cast<size_t>(row) * chromaRowBytes;
                if (decoder.swapChromaUV) {
                    for (uint32_t col = 0; col < chromaWidth; ++col) {
                        dstRow[col * 2 + 0] = srcRow[col * 2 + 1];
                        dstRow[col * 2 + 1] = srcRow[col * 2 + 0];
                    }
                } else {
                    std::memcpy(dstRow, srcRow, chromaRowBytes);
                }
                // Calculate chroma sums for interleaved 8-bit data
                for (uint32_t col = 0; col < chromaWidth; ++col) {
                    if (decoder.swapChromaUV) {
                        uSum += srcRow[col * 2 + 1];
                        vSum += srcRow[col * 2 + 0];
                    } else {
                        uSum += srcRow[col * 2 + 0];
                        vSum += srcRow[col * 2 + 1];
                    }
                }
            }
        }
    } else {
        uint16_t* dstUv16 = reinterpret_cast<uint16_t*>(dstUv);
        for (uint32_t row = 0; row < chromaHeight; ++row) {
            uint16_t* dstRow = dstUv16 + static_cast<size_t>(row) * chromaWidth * 2;
            if (!interleaved) {
                const uint16_t* uRow = reinterpret_cast<const uint16_t*>(frame->data[1] + static_cast<size_t>(row) * frame->linesize[1]);
                const uint16_t* vRow = reinterpret_cast<const uint16_t*>(frame->data[2] + static_cast<size_t>(row) * frame->linesize[2]);
                for (uint32_t col = 0; col < chromaWidth; ++col) {
                    uint16_t uVal = shift ? static_cast<uint16_t>(uRow[col] << shift) : uRow[col];
                    uint16_t vVal = shift ? static_cast<uint16_t>(vRow[col] << shift) : vRow[col];
                    dstRow[col * 2 + 0] = uVal;
                    dstRow[col * 2 + 1] = vVal;
                    uSum += uVal;
                    vSum += vVal;
                }
            } else {
                const uint16_t* srcRow = reinterpret_cast<const uint16_t*>(frame->data[1] + static_cast<size_t>(row) * frame->linesize[1]);
                for (uint32_t col = 0; col < chromaWidth; ++col) {
                    uint16_t uVal = srcRow[col * 2 + 0];
                    uint16_t vVal = srcRow[col * 2 + 1];
                    if (shift) {
                        uVal = static_cast<uint16_t>(uVal << shift);
                        vVal = static_cast<uint16_t>(vVal << shift);
                    }
                    if (decoder.swapChromaUV) {
                        dstRow[col * 2 + 0] = vVal;
                        dstRow[col * 2 + 1] = uVal;
                        uSum += vVal;
                        vSum += uVal;
                    } else {
                        dstRow[col * 2 + 0] = uVal;
                        dstRow[col * 2 + 1] = vVal;
                        uSum += uVal;
                        vSum += vVal;
                    }
                }
            }
        }
    }
    
    // Print debug info
    if (frameCount <= 10) {  // Only print first 10 frames to avoid spam
        double yAvg = sampleCount > 0 ? static_cast<double>(ySum) / sampleCount : 0.0;
        double uAvg = (chromaWidth * chromaHeight) > 0 ? static_cast<double>(uSum) / (chromaWidth * chromaHeight) : 0.0;
        double vAvg = (chromaWidth * chromaHeight) > 0 ? static_cast<double>(vSum) / (chromaWidth * chromaHeight) : 0.0;
        printf("[VideoFrameUtils] Frame %d: %dx%d, bitDepth=%u, shift=%u, Y avg=%.1f, U avg=%.1f, V avg=%.1f\n",
               frameCount, width, height, bitDepth, shift, yAvg, uAvg, vAvg);
    }
}

} // namespace

bool configureFormatForPixelFormat(VideoDecoder& decoder, AVPixelFormat pixFmt)
{
    if (pixFmt == AV_PIX_FMT_NONE) {
        return false;
    }

    decoder.sourcePixelFormat = pixFmt;
    if (configureNv12Format(decoder, pixFmt)) {
        return true;
    }
    if (configurePlanarFormat(decoder, pixFmt)) {
        return true;
    }
    return false;
}

void copyDecodedFrameToBuffer(const VideoDecoder& decoder,
                              AVFrame* frame,
                              std::vector<uint8_t>& buffer)
{
    if (decoder.planarYuv) {
        copyPlanarFrameToBuffer(decoder, frame, buffer);
    } else {
        copyNv12FrameToBuffer(frame, buffer, decoder.width, decoder.height, decoder.swapChromaUV);
    }
}

} // namespace video
