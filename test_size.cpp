#include "decoder_vulkan.h"
#include <iostream>
int main() {
    std::cout << "sizeof(DecodedFrame) = " << sizeof(DecodedFrame) << std::endl;
    std::cout << "sizeof(VulkanSurface) = " << sizeof(VulkanSurface) << std::endl;
    std::cout << "sizeof(std::array<VkImage, 3>) = " << sizeof(std::array<VkImage, 3>) << std::endl;
    return 0;
}
