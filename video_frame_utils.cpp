#include <algorithm>
#include <cstring>
#include "video_frame_utils.h"
#include "decoder.h"
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
#ifdef AV_PIX_FMT_P010LE
    {AV_PIX_FMT_P010LE, 2, 2, 10, PrimitiveYuvFormat::Planar420, true},
#endif
#ifdef AV_PIX_FMT_P012LE
    {AV_PIX_FMT_P012LE, 2, 2, 12, PrimitiveYuvFormat::Planar420, true},
#endif
#ifdef AV_PIX_FMT_P016LE
    {AV_PIX_FMT_P016LE, 2, 2, 16, PrimitiveYuvFormat::Planar420, true},
#endif
};

bool configureNv12Format(Decoder& decoder, AVPixelFormat format)
{
    if (format != AV_PIX_FMT_NV12
#ifdef AV_PIX_FMT_NV21
        && format != AV_PIX_FMT_NV21
#endif
    ) {
        return false;
    }
    decoder.planarYuv = false;
    decoder.chromaInterleaved = true;  // NV12 has interleaved chroma
    decoder.outputFormat = PrimitiveYuvFormat::NV12;
    decoder.bytesPerComponent = 1;
    decoder.bitDepth = 8;
    decoder.swapChromaUV =
#ifdef AV_PIX_FMT_NV21
        (format == AV_PIX_FMT_NV21);
#else
        false;
#endif
    decoder.chromaDivX = 2;
    decoder.chromaDivY = 2;
    decoder.chromaWidth = std::max<uint32_t>(1u, (decoder.width + decoder.chromaDivX - 1) / decoder.chromaDivX);
    decoder.chromaHeight = std::max<uint32_t>(1u, (decoder.height + decoder.chromaDivY - 1) / decoder.chromaDivY);
    decoder.yPlaneBytes = decoder.width * decoder.height * decoder.bytesPerComponent;
    decoder.uvPlaneBytes = decoder.chromaWidth * decoder.chromaHeight * 2 * decoder.bytesPerComponent;
    decoder.bufferSize = decoder.yPlaneBytes + decoder.uvPlaneBytes;
    return true;
}

bool configurePlanarFormat(Decoder& decoder, AVPixelFormat pixFormat)
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
        decoder.yPlaneBytes = decoder.width * decoder.height * decoder.bytesPerComponent;
        decoder.uvPlaneBytes = decoder.chromaWidth * decoder.chromaHeight * 2 * decoder.bytesPerComponent;
        decoder.bufferSize = decoder.yPlaneBytes + decoder.uvPlaneBytes;
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
    if (dstStride == srcStride && srcStride == width) {
        // Fast path: contiguous memory
        std::memcpy(dst, src, static_cast<size_t>(width) * height);
    } else {
        for (int y = 0; y < height; ++y) {
            std::memcpy(dst + static_cast<size_t>(y) * dstStride,
                        src + static_cast<size_t>(y) * srcStride,
                        width);
        }
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
    
    if (buffer.capacity() < requiredSize) {
        buffer.reserve(requiredSize);
    }
    buffer.resize(requiredSize);

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

void copyPlanarFrameToBuffer(const Decoder& decoder,
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

    const size_t requiredSize = decoder.bufferSize;
    if (buffer.capacity() < requiredSize) {
        buffer.reserve(requiredSize);
    }
    buffer.resize(requiredSize);

    static int frameCount = 0;
    frameCount++;

    uint8_t* dstY = buffer.data();
    uint64_t ySum = 0;
    uint64_t uSum = 0;
    uint64_t vSum = 0;
    const bool interleaved = decoder.chromaInterleaved;
    const uint32_t bitDepth = decoder.bitDepth > 0 ? decoder.bitDepth : 8;
    
    // Calculate shift for high bit depth formats (only for 10/12-bit to 16-bit conversion)
    uint32_t shift = 0;
    if (bytesPerComponent == 2 && bitDepth < 16) {
        // Only shift if we're converting from 10/12-bit to 16-bit
        shift = 16 - bitDepth;
    }

    // Copy Y plane
    if (bytesPerComponent == 1) {
        // 8-bit Y plane
        const size_t yRowBytes = width;
        for (uint32_t row = 0; row < height; ++row) {
            const uint8_t* srcRow = frame->data[0] + static_cast<size_t>(row) * frame->linesize[0];
            uint8_t* dstRow = dstY + static_cast<size_t>(row) * yRowBytes;
            std::memcpy(dstRow, srcRow, yRowBytes);
            
            // Calculate sum for debugging
            for (uint32_t col = 0; col < width; ++col) {
                ySum += srcRow[col];
            }
        }
    } else {
        // 16-bit Y plane
        const size_t yRowPixels = width;
        uint16_t* dstY16 = reinterpret_cast<uint16_t*>(dstY);
        for (uint32_t row = 0; row < height; ++row) {
            const uint16_t* srcRow = reinterpret_cast<const uint16_t*>(
                frame->data[0] + static_cast<size_t>(row) * frame->linesize[0]);
            uint16_t* dstRow = dstY16 + static_cast<size_t>(row) * yRowPixels;
            
            if (shift > 0) {
                // Apply shift for 10/12-bit to 16-bit conversion
                for (uint32_t col = 0; col < width; ++col) {
                    uint16_t value = srcRow[col];
                    dstRow[col] = static_cast<uint16_t>(value << shift);
                    ySum += srcRow[col]; // Use original value for average calculation
                }
            } else {
                // No shift needed (16-bit or raw copy)
                std::memcpy(dstRow, srcRow, yRowPixels * sizeof(uint16_t));
                for (uint32_t col = 0; col < width; ++col) {
                    ySum += srcRow[col];
                }
            }
        }
    }

    // Copy UV planes
    uint8_t* dstUv = buffer.data() + decoder.yPlaneBytes;
    
    if (!interleaved) {
        // Separate U and V planes
        if (bytesPerComponent == 1) {
            // 8-bit separate planes
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
            // 16-bit separate planes
            uint16_t* dstUv16 = reinterpret_cast<uint16_t*>(dstUv);
            for (uint32_t row = 0; row < chromaHeight; ++row) {
                const uint16_t* uRow = reinterpret_cast<const uint16_t*>(
                    frame->data[1] + static_cast<size_t>(row) * frame->linesize[1]);
                const uint16_t* vRow = reinterpret_cast<const uint16_t*>(
                    frame->data[2] + static_cast<size_t>(row) * frame->linesize[2]);
                uint16_t* dstRow = dstUv16 + static_cast<size_t>(row) * chromaWidth * 2;
                
                if (shift > 0) {
                    for (uint32_t col = 0; col < chromaWidth; ++col) {
                        uint16_t uVal = static_cast<uint16_t>(uRow[col] << shift);
                        uint16_t vVal = static_cast<uint16_t>(vRow[col] << shift);
                        dstRow[col * 2 + 0] = uVal;
                        dstRow[col * 2 + 1] = vVal;
                        uSum += uRow[col]; // Original value
                        vSum += vRow[col]; // Original value
                    }
                } else {
                    for (uint32_t col = 0; col < chromaWidth; ++col) {
                        dstRow[col * 2 + 0] = uRow[col];
                        dstRow[col * 2 + 1] = vRow[col];
                        uSum += uRow[col];
                        vSum += vRow[col];
                    }
                }
            }
        }
    } else {
        // Interleaved UV plane
        if (bytesPerComponent == 1) {
            // 8-bit interleaved
            const size_t chromaRowBytes = chromaWidth * 2;
            for (uint32_t row = 0; row < chromaHeight; ++row) {
                const uint8_t* srcRow = frame->data[1] + static_cast<size_t>(row) * frame->linesize[1];
                uint8_t* dstRow = dstUv + static_cast<size_t>(row) * chromaRowBytes;
                
                if (decoder.swapChromaUV) {
                    for (uint32_t col = 0; col < chromaWidth; ++col) {
                        dstRow[col * 2 + 0] = srcRow[col * 2 + 1];
                        dstRow[col * 2 + 1] = srcRow[col * 2 + 0];
                        uSum += srcRow[col * 2 + 1];
                        vSum += srcRow[col * 2 + 0];
                    }
                } else {
                    std::memcpy(dstRow, srcRow, chromaRowBytes);
                    for (uint32_t col = 0; col < chromaWidth; ++col) {
                        uSum += srcRow[col * 2 + 0];
                        vSum += srcRow[col * 2 + 1];
                    }
                }
            }
        } else {
            // 16-bit interleaved
            uint16_t* dstUv16 = reinterpret_cast<uint16_t*>(dstUv);
            const size_t chromaRowPixels = chromaWidth * 2;
            for (uint32_t row = 0; row < chromaHeight; ++row) {
                const uint16_t* srcRow = reinterpret_cast<const uint16_t*>(
                    frame->data[1] + static_cast<size_t>(row) * frame->linesize[1]);
                uint16_t* dstRow = dstUv16 + static_cast<size_t>(row) * chromaRowPixels;
                
                if (decoder.swapChromaUV) {
                    if (shift > 0) {
                        for (uint32_t col = 0; col < chromaWidth; ++col) {
                            uint16_t uVal = static_cast<uint16_t>(srcRow[col * 2 + 1] << shift);
                            uint16_t vVal = static_cast<uint16_t>(srcRow[col * 2 + 0] << shift);
                            dstRow[col * 2 + 0] = uVal;
                            dstRow[col * 2 + 1] = vVal;
                            uSum += srcRow[col * 2 + 1]; // Original value
                            vSum += srcRow[col * 2 + 0]; // Original value
                        }
                    } else {
                        for (uint32_t col = 0; col < chromaWidth; ++col) {
                            dstRow[col * 2 + 0] = srcRow[col * 2 + 1];
                            dstRow[col * 2 + 1] = srcRow[col * 2 + 0];
                            uSum += srcRow[col * 2 + 1];
                            vSum += srcRow[col * 2 + 0];
                        }
                    }
                } else {
                    if (shift > 0) {
                        for (uint32_t col = 0; col < chromaWidth; ++col) {
                            uint16_t uVal = static_cast<uint16_t>(srcRow[col * 2 + 0] << shift);
                            uint16_t vVal = static_cast<uint16_t>(srcRow[col * 2 + 1] << shift);
                            dstRow[col * 2 + 0] = uVal;
                            dstRow[col * 2 + 1] = vVal;
                            uSum += srcRow[col * 2 + 0]; // Original value
                            vSum += srcRow[col * 2 + 1]; // Original value
                        }
                    } else {
                        std::memcpy(dstRow, srcRow, chromaRowPixels * sizeof(uint16_t));
                        for (uint32_t col = 0; col < chromaWidth; ++col) {
                            uSum += srcRow[col * 2 + 0];
                            vSum += srcRow[col * 2 + 1];
                        }
                    }
                }
            }
        }
    }
    
    // Print debug info for first few frames
    if (frameCount <= 10) {
        double yAvg = (width * height) > 0 ? static_cast<double>(ySum) / (width * height) : 0.0;
        double uAvg = (chromaWidth * chromaHeight) > 0 ? static_cast<double>(uSum) / (chromaWidth * chromaHeight) : 0.0;
        double vAvg = (chromaWidth * chromaHeight) > 0 ? static_cast<double>(vSum) / (chromaWidth * chromaHeight) : 0.0;
        
        // Scale 16-bit values down for display
        if (bytesPerComponent == 2) {
            double scale = (bitDepth < 16) ? (65535.0 / ((1 << bitDepth) - 1)) : 1.0;
            yAvg *= scale;
            uAvg *= scale;
            vAvg *= scale;
        }
        
        printf("[VideoFrameUtils] Frame %d: %dx%d, bitDepth=%u, shift=%u, Y avg=%.1f, U avg=%.1f, V avg=%.1f\n",
               frameCount, width, height, bitDepth, shift, yAvg, uAvg, vAvg);
    }
}

} // namespace

bool configureFormatForPixelFormat(Decoder& decoder, AVPixelFormat pixFmt)
{
    if (pixFmt == AV_PIX_FMT_NONE) {
        return false;
    }

    decoder.sourcePixelFormat = pixFmt;
    
    // Try NV12 first (including NV21)
    if (configureNv12Format(decoder, pixFmt)) {
        return true;
    }
    
    // Try planar formats
    if (configurePlanarFormat(decoder, pixFmt)) {
        return true;
    }
    
    return false;
}

void copyDecodedFrameToBuffer(const Decoder& decoder,
                              AVFrame* frame,
                              std::vector<uint8_t>& buffer)
{
    if (!frame || !frame->data[0]) {
        buffer.clear();
        return;
    }
    
    if (decoder.outputFormat == PrimitiveYuvFormat::NV12) {
        copyNv12FrameToBuffer(frame, buffer, decoder.width, decoder.height, decoder.swapChromaUV);
    } else {
        copyPlanarFrameToBuffer(decoder, frame, buffer);
    }
}

} // namespace video