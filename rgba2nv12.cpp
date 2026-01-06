#include "rgba2nv12.h"

#include "engine2d.h"
#include "utils.h"
#include "debug_logging.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

rgba2nv12::rgba2nv12(Engine2D* engine,
                       uint32_t groupX,
                       uint32_t groupY)
    : engine(engine), groupX(groupX), groupY(groupY)
{
    // Pipeline creation deferred until pipelineLayout is set
    // TODO: create pipeline layout and pipeline properly
}

rgba2nv12::~rgba2nv12()
{
    if (engine && pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(engine->logicalDevice, pipeline, nullptr);
    }
}

bool rgba2nv12::run()
{
    if (commandBuffer == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE)
    {
        return false;
    }

    if (renderDebugEnabled())
    {
        std::cout << "[rgba2nv12] run: groupX=" << groupX << ", groupY=" << groupY << std::endl;
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
