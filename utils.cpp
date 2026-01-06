#include "utils.h"

#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

// Include stb_image_write from ncnn
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ncnn/src/stb_image_write.h"

std::vector<char> readSPIRVFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open SPIR-V file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize % 4 != 0)
    {
        throw std::runtime_error("SPIR-V file size not multiple of 4: " + filename);
    }

    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    if (fileSize >= 4)
    {
        uint32_t magic = *reinterpret_cast<uint32_t *>(buffer.data());
        if (magic != 0x07230203)
        {
            throw std::runtime_error("Invalid SPIR-V magic number in file: " + filename);
        }
    }

    return buffer;
}

VkSampleCountFlagBits msaaFlagFromInt(int samples)
{
    switch (samples)
    {
    case 1:
        return VK_SAMPLE_COUNT_1_BIT;
    case 2:
        return VK_SAMPLE_COUNT_2_BIT;
    case 4:
        return VK_SAMPLE_COUNT_4_BIT;
    case 8:
        return VK_SAMPLE_COUNT_8_BIT;
    case 16:
        return VK_SAMPLE_COUNT_16_BIT;
    case 32:
        return VK_SAMPLE_COUNT_32_BIT;
    case 64:
        return VK_SAMPLE_COUNT_64_BIT;
    default:
        return static_cast<VkSampleCountFlagBits>(VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM);
    }
}

int msaaIntFromFlag(VkSampleCountFlagBits flag)
{
    switch (flag)
    {
    case VK_SAMPLE_COUNT_1_BIT:
        return 1;
    case VK_SAMPLE_COUNT_2_BIT:
        return 2;
    case VK_SAMPLE_COUNT_4_BIT:
        return 4;
    case VK_SAMPLE_COUNT_8_BIT:
        return 8;
    case VK_SAMPLE_COUNT_16_BIT:
        return 16;
    case VK_SAMPLE_COUNT_32_BIT:
        return 32;
    case VK_SAMPLE_COUNT_64_BIT:
        return 64;
    default:
        return 1;
    }
}

// Image saving implementations
bool saveImageToPNG(const std::filesystem::path& path, 
                    const void* data, 
                    int width, 
                    int height, 
                    int channels)
{
    if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4)
    {
        std::cerr << "[saveImageToPNG] Invalid parameters" << std::endl;
        return false;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::create_directories(path.parent_path());
    
    int stride = width * channels;
    int result = stbi_write_png(path.string().c_str(), width, height, channels, data, stride);
    
    if (result == 0)
    {
        std::cerr << "[saveImageToPNG] Failed to write PNG: " << path << std::endl;
        return false;
    }
    
    std::cout << "[saveImageToPNG] Saved: " << path << std::endl;
    return true;
}

bool saveImageToJPG(const std::filesystem::path& path, 
                    const void* data, 
                    int width, 
                    int height, 
                    int channels,
                    int quality)
{
    if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4)
    {
        std::cerr << "[saveImageToJPG] Invalid parameters" << std::endl;
        return false;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::create_directories(path.parent_path());
    
    int result = stbi_write_jpg(path.string().c_str(), width, height, channels, data, quality);
    
    if (result == 0)
    {
        std::cerr << "[saveImageToJPG] Failed to write JPG: " << path << std::endl;
        return false;
    }
    
    std::cout << "[saveImageToJPG] Saved: " << path << std::endl;
    return true;
}
