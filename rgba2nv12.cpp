#include "rgba2nv12.h"

#include "engine2d.h"
#include "utils.h"

#include <stdexcept>

VkPipeline createRgbaToNv12Pipeline(Engine2D* engine, VkPipelineLayout pipelineLayout)
{
    if (!engine || pipelineLayout == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Invalid parameters while creating RGBA-to-NV12 pipeline");
    }

    auto shaderCode = readSPIRVFile("shaders/rgba2nv12.comp.spv");
    VkShaderModule shaderModule = engine->createShaderModule(shaderCode);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(engine->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
        throw std::runtime_error("Failed to create RGBA-to-NV12 compute pipeline");
    }

    vkDestroyShaderModule(engine->logicalDevice, shaderModule, nullptr);
    return pipeline;
}

void destroyRgbaToNv12Pipeline(Engine2D* engine, VkPipeline pipeline)
{
    if (engine && pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
    }
}

void dispatchRgbaToNv12(VkCommandBuffer commandBuffer,
                        VkPipeline pipeline,
                        VkPipelineLayout pipelineLayout,
                        VkDescriptorSet descriptorSet,
                        const RgbaToNv12PushConstants& pushConstants,
                        uint32_t groupX,
                        uint32_t groupY)
{
    if (commandBuffer == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE)
    {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RgbaToNv12PushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, groupX, groupY, 1);
}


bool convertNv12ToBgr(const uint8_t* nv12,
                      size_t yBytes,
                      size_t uvBytes,
                      int width,
                      int height,
                      std::vector<uint8_t>& bgr)
{
    if (!nv12 || width <= 0 || height <= 0 || yBytes == 0 || uvBytes == 0)
    {
        return false;
    }

    const size_t framePixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (bgr.size() != framePixels * 3)
    {
        bgr.resize(framePixels * 3);
    }

    const uint8_t* yPlane = nv12;
    const uint8_t* uvPlane = nv12 + yBytes;

    for (int row = 0; row < height; ++row)
    {
        const uint8_t* yRow = yPlane + static_cast<size_t>(row) * static_cast<size_t>(width);
        const uint8_t* uvRow = uvPlane + static_cast<size_t>(row / 2) * static_cast<size_t>(width);
        for (int col = 0; col < width; ++col)
        {
            const float Y = static_cast<float>(yRow[col]);
            const float U = static_cast<float>(uvRow[(col / 2) * 2]) - 128.0f;
            const float V = static_cast<float>(uvRow[(col / 2) * 2 + 1]) - 128.0f;

            const float y = std::max(0.0f, (Y - 16.0f)) * 1.164383f;
            const float r = y + 1.596027f * V;
            const float g = y - 0.391762f * U - 0.812968f * V;
            const float b = y + 2.017232f * U;

            const size_t dstIndex = (static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)) * 3;
            bgr[dstIndex + 0] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
            bgr[dstIndex + 1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
            bgr[dstIndex + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
        }
    }

    return true;
}


bool savePpmFromYuv(const std::filesystem::path& path, 
                    const uint8_t* yuvData, 
                    int width, 
                    int height,
                    int bytesPerComponent,
                    int yPlaneBytes,
                    int uvPlaneBytes)
{
    if (!yuvData || width <= 0 || height <= 0 || bytesPerComponent != 2)
    {
        std::cerr << "[motive2d] Unsupported format for PPM conversion\n";
        return false;
    }
    
    // For p010le (10-bit YUV 4:2:0), we need to convert to 8-bit RGB
    // This is a simplified conversion for debugging
    std::vector<uint8_t> rgb(width * height * 3);
    
    const uint16_t* yPlane = reinterpret_cast<const uint16_t*>(yuvData);
    const uint16_t* uvPlane = reinterpret_cast<const uint16_t*>(yuvData + yPlaneBytes);
    
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // Simplified YUV to RGB conversion (p010le is 10-bit in 16-bit words)
            int yIdx = y * width + x;
            int uvIdx = (y / 2) * (width / 2) + (x / 2);
            
            // Scale 10-bit to 8-bit (shift right by 2)
            uint8_t Y = static_cast<uint8_t>(std::min(255, yPlane[yIdx] >> 2));
            uint8_t U = static_cast<uint8_t>(std::min(255, uvPlane[uvIdx * 2] >> 2));
            uint8_t V = static_cast<uint8_t>(std::min(255, uvPlane[uvIdx * 2 + 1] >> 2));
            
            // Simple YUV to RGB conversion (ITU-R BT.601)
            int C = Y - 16;
            int D = U - 128;
            int E = V - 128;
            
            int r = std::clamp((298 * C + 409 * E + 128) >> 8, 0, 255);
            int g = std::clamp((298 * C - 100 * D - 208 * E + 128) >> 8, 0, 255);
            int b = std::clamp((298 * C + 516 * D + 128) >> 8, 0, 255);
            
            int idx = (y * width + x) * 3;
            rgb[idx] = static_cast<uint8_t>(r);
            rgb[idx + 1] = static_cast<uint8_t>(g);
            rgb[idx + 2] = static_cast<uint8_t>(b);
        }
    }
    
    // Write PPM file
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "[motive2d] Failed to open PPM file for writing: " << path << "\n";
        return false;
    }
    
    // PPM header
    file << "P6\n" << width << " " << height << "\n255\n";
    if (!file)
    {
        std::cerr << "[motive2d] Failed to write PPM header\n";
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
    if (!file)
    {
        std::cerr << "[motive2d] Failed to write PPM data\n";
        return false;
    }
    
    std::cout << "[motive2d] Saved PPM image (" << width << "x" << height << ") to: " << path << "\n";
    return true;
}
