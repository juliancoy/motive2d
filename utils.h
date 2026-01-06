#pragma once

#include <vector>
#include <string>
#include <vulkan/vulkan.h>
#include <filesystem>

std::vector<char> readSPIRVFile(const std::string &filename);
VkSampleCountFlagBits msaaFlagFromInt(int samples);
int msaaIntFromFlag(VkSampleCountFlagBits flag);

// Image saving utilities
bool saveImageToPNG(const std::filesystem::path& path, 
                    const void* data, 
                    int width, 
                    int height, 
                    int channels = 4);
bool saveImageToJPG(const std::filesystem::path& path, 
                    const void* data, 
                    int width, 
                    int height, 
                    int channels = 4,
                    int quality = 90);
