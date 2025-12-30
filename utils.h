#pragma once

#include <vector>
#include <string>
#include <vulkan/vulkan.h>

std::vector<char> readSPIRVFile(const std::string &filename);
VkSampleCountFlagBits msaaFlagFromInt(int samples);
int msaaIntFromFlag(VkSampleCountFlagBits flag);
