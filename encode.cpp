#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "engine2d.h"
#include "grading.hpp"
#include "utils.h"
#include "annexb_demuxer.h"

namespace
{
// For raw Annex-B elementary streams (e.g., .h264 / .h265)
const std::filesystem::path kDefaultVideoPath("input.h264");

struct OffscreenBlit
{
    Engine2D* engine = nullptr;
    VkExtent2D extent{0, 0};
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    bool ready() const { return image != VK_NULL_HANDLE; }

    bool initialize(Engine2D* inEngine, uint32_t width, uint32_t height, const std::array<float, kCurveLutSize>&)
    {
        engine = inEngine;
        extent = {width, height};
        if (!engine || width == 0 || height == 0)
        {
            return false;
        }

        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent = {extent.width, extent.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(engine->logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            return false;
        }
        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(engine->logicalDevice, image, &memReq);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = memReq.size;
        alloc.memoryTypeIndex = engine->renderDevice.findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(engine->logicalDevice, &alloc, nullptr, &memory) != VK_SUCCESS)
        {
            return false;
        }
        vkBindImageMemory(engine->logicalDevice, image, memory, 0);
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(engine->logicalDevice, &viewInfo, nullptr, &view) != VK_SUCCESS)
        {
            return false;
        }
        return true;
    }

    bool copyFrom(VkImage srcImage, VkExtent2D srcExtent, VkImageLayout srcLayout)
    {
        if (!engine || image == VK_NULL_HANDLE || srcImage == VK_NULL_HANDLE)
        {
            return false;
        }
        VkCommandBuffer cmd = engine->beginSingleTimeCommands();
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.levelCount = 1;
        range.layerCount = 1;

        VkImageMemoryBarrier2 srcBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        srcBarrier.image = srcImage;
        srcBarrier.oldLayout = srcLayout;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.subresourceRange = range;
        srcBarrier.srcAccessMask = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
        srcBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        srcBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
        srcBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

        VkImageMemoryBarrier2 dstBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        dstBarrier.image = image;
        dstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dstBarrier.subresourceRange = range;
        dstBarrier.srcAccessMask = 0;
        dstBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        dstBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        dstBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

        std::array<VkImageMemoryBarrier2, 2> barriers{srcBarrier, dstBarrier};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dep.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(cmd, &dep);

        VkImageCopy copy{};
        copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.srcSubresource.layerCount = 1;
        copy.dstSubresource = copy.srcSubresource;
        copy.extent = {std::min(extent.width, srcExtent.width), std::min(extent.height, srcExtent.height), 1};
        vkCmdCopyImage(cmd,
                       srcImage,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &copy);

        VkImageMemoryBarrier2 toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        toGeneral.image = image;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.subresourceRange = range;
        toGeneral.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        VkDependencyInfo dep2{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep2.imageMemoryBarrierCount = 1;
        dep2.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cmd, &dep2);

        engine->endSingleTimeCommands(cmd);
        return true;
    }

    void shutdown()
    {
        if (!engine) return;
        if (view) vkDestroyImageView(engine->logicalDevice, view, nullptr);
        if (image) vkDestroyImage(engine->logicalDevice, image, nullptr);
        if (memory) vkFreeMemory(engine->logicalDevice, memory, nullptr);
        view = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

ColorAdjustments gradingToAdjustments(const GradingSettings& settings)
{
    ColorAdjustments adj{};
    adj.exposure = settings.exposure;
    adj.contrast = settings.contrast;
    adj.saturation = settings.saturation;
    adj.shadows = settings.shadows;
    adj.midtones = settings.midtones;
    adj.highlights = settings.highlights;
    grading::buildCurveLut(settings, adj.curveLut);
    adj.curveEnabled = true;
    return adj;
}
} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path videoPath = kDefaultVideoPath;
    bool showHelp = false;
    size_t frameLimit = 0; // 0 means no limit
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--video" || arg == "-v")
        {
            if (i + 1 < argc)
            {
                videoPath = std::filesystem::path(argv[++i]);
            }
            else
            {
                std::cerr << "[Encode] Error: --video requires a file path\n";
                return 1;
            }
        }
        else if (arg == "--framecount" || arg == "-f")
        {
            if (i + 1 < argc)
            {
                try
                {
                    frameLimit = std::stoul(argv[++i]);
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[Encode] Error: --framecount requires a positive integer\n";
                    return 1;
                }
            }
            else
            {
                std::cerr << "[Encode] Error: --framecount requires a value\n";
                return 1;
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            showHelp = true;
        }
        else if (arg[0] != '-')
        {
            videoPath = std::filesystem::path(arg);
        }
        else
        {
            std::cerr << "[Encode] Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            return 1;
        }
    }

    if (showHelp)
    {
        std::cout << "Usage: encode [OPTIONS] [VIDEO_FILE]\n";
        std::cout << "Vulkan-only skeleton (no FFmpeg) — decode/encode not yet implemented\n\n";
        std::cout << "Options:\n";
        std::cout << "  -v, --video FILE      Input video file (default: " << kDefaultVideoPath << ")\n";
        std::cout << "  -f, --framecount N    Process only first N frames (0 = no limit)\n";
        std::cout << "  -h, --help            Show this help message\n";
        return 0;
    }

    std::cout << "[Encode] Engine2D constructed\n";
    Engine2D engine;

    // Minimal decoder path setup
    // Try H.265 first (since our test file is H.265), fall back to H.264
    std::cout << "[Encode] Selecting decode profile...\n";
    auto profile = selectDecodeProfile(engine, MiniCodec::H265);
    if (!profile)
    {
        std::cerr << "[Encode] H.265 decode not available, trying H.264...\n";
        profile = selectDecodeProfile(engine, MiniCodec::H264);
    }
    if (!profile)
    {
        std::cerr << "[Encode] No suitable decode profile found.\n";
        return 1;
    }
    std::cout << "[Encode] Profile selected, creating decode session...\n";
    auto session = createDecodeSession(engine, *profile, VkExtent2D{1920, 1080}, /*maxDpbSlots*/4);
    if (!session)
    {
        std::cerr << "[Encode] Failed to create decode session.\n";
        return 1;
    }
    std::cout << "[Encode] Decode session created, initializing resources...\n";
    MiniDecodeResources decRes{};
    if (!initDecodeResources(engine, *session, decRes, 4 * 1024 * 1024))
    {
        std::cerr << "[Encode] Failed to init decode resources.\n";
        return 1;
    }
    std::cout << "[Encode] Decode resources initialized.\n";

    // Encode pipeline setup
    std::cout << "[Encode] Selecting encode profile...\n";
    MiniEncodeCodec encodeCodec = (profile->codec == MiniCodec::H264) ? MiniEncodeCodec::H264 : MiniEncodeCodec::H265;
    auto encodeProfile = selectEncodeProfile(engine, encodeCodec);
    if (!encodeProfile)
    {
        std::cerr << "[Encode] No suitable encode profile found.\n";
        return 1;
    }
    std::cout << "[Encode] Encode profile selected, creating encode session...\n";
    auto encodeSession = createEncodeSession(engine, *encodeProfile, VkExtent2D{1920, 1080}, /*maxDpbSlots*/4);
    if (!encodeSession)
    {
        std::cerr << "[Encode] Failed to create encode session.\n";
        return 1;
    }
    std::cout << "[Encode] Encode session created, initializing resources...\n";
    MiniEncodeResources encRes{};
    if (!initEncodeResources(engine, *encodeSession, encRes))
    {
        std::cerr << "[Encode] Failed to init encode resources.\n";
        return 1;
    }
    std::cout << "[Encode] Encode resources initialized.\n";

    // Open output file for encoded bitstream
    std::filesystem::path outputPath = videoPath;
    outputPath.replace_extension("encoded.h265");
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile)
    {
        std::cerr << "[Encode] Failed to open output file " << outputPath << "\n";
        return 1;
    }
    std::cout << "[Encode] Writing encoded bitstream to " << outputPath << "\n";

    std::cout << "[Encode] Opening Annex-B demuxer for " << videoPath << "...\n";
    AnnexBDemuxer demux(videoPath);
    std::cout << "[Encode] Demuxer constructed, valid=" << demux.valid() << "\n";
    if (!demux.valid())
    {
        std::cerr << "[Encode] Failed to open Annex-B input.\n";
        return 1;
    }
    std::cout << "[Encode] Demuxer ready, starting decode loop\n";

    size_t framesProcessed = 0;
    OffscreenBlit blit;
    const uint8_t* nal = nullptr;
    size_t nalSize = 0;
    bool isIdr = false;
    while (demux.nextNalu(nal, nalSize, isIdr))
    {
        if (nalSize == 0)
        {
            std::cout << "[Encode] NAL size zero, breaking.\n";
            break;
        }
        std::cout << "[Encode] Decoding NAL of size " << nalSize << " bytes (IDR=" << isIdr << ")\n";
        // Check frame limit
        if (frameLimit > 0 && framesProcessed >= frameLimit)
        {
            std::cout << "[Encode] Reached frame limit of " << frameLimit << " frames\n";
            break;
        }
        
        uint32_t slot = 0;
        std::cout << "[Encode] Calling recordDecode...\n";
        if (!recordDecode(engine, *session, decRes, nal, nalSize, VkExtent2D{1920, 1080}, slot))
        {
            std::cerr << "[Encode] Decode submit failed at frame " << framesProcessed << "\n";
            break;
        }
        std::cout << "[Encode] Decode submitted, slot=" << slot << "\n";
        // Temporary sync: ensure decode queue idle before handoff to encode.
        vkQueueWaitIdle(engine.getVideoDecodeQueue());
        // Encode the decoded DPB image directly (same format as encoder expects)
        std::cout << "[Encode] Encoding frame...\n";
        uint32_t encodeSlot = 0;
        // Use the decoded DPB image as source, it's already in the right YUV format
        if (!recordEncode(engine,
                          *encodeSession,
                          encRes,
                          session->dpbImages[slot],
                          session->codedExtent,
                          VK_IMAGE_LAYOUT_GENERAL,
                          engine.getVideoDecodeQueueFamilyIndex(),
                          encodeSlot))
        {
            std::cerr << "[Encode] Encode submit failed at frame " << framesProcessed << "\n";
            break;
        }
        std::cout << "[Encode] Encode submitted, retrieving bitstream...\n";
        std::vector<uint8_t> bitstream;
        if (!retrieveEncodedBitstream(engine, *encodeSession, encRes, bitstream))
        {
            std::cerr << "[Encode] Failed to retrieve encoded bitstream\n";
            break;
        }
        // Write bitstream to file (TODO: need actual encoded size)
        outFile.write(reinterpret_cast<const char*>(bitstream.data()), bitstream.size());
        if (!outFile)
        {
            std::cerr << "[Encode] Failed to write to output file\n";
            break;
        }
        std::cout << "[Encode] Wrote " << bitstream.size() << " bytes to output.\n";
        
        framesProcessed++;
        std::cout << "[Encode] Frame " << framesProcessed << " processed.\n";
    }
    if (blit.ready())
    {
        blit.shutdown();
    }

    // Cleanup encode resources
    destroyEncodeResources(engine, encRes);
    if (encodeSession)
    {
        destroyEncodeSession(engine, *encodeSession);
    }
    // decode resources cleanup already handled by destructors? Not yet, but we can add later.

    std::cout << "[Encode] Processed " << framesProcessed << " Annex-B frames (decode functional, encode attempted).\n";
    return 0;
}
