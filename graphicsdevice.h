#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

class RenderDevice
{
public:
    RenderDevice();
    ~RenderDevice();

    void initialize();
    void shutdown();

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void nameVulkanObject(uint64_t handle, VkObjectType type, const char *name);
    void createDescriptorSetLayouts();

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
                    VkCommandPool commandPool, VkQueue graphicsQueue);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer &buffer, VkDeviceMemory &bufferMemory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    VkShaderModule createShaderModule(const std::vector<char> &code);

    // Accessors providing references for legacy code paths
    VkInstance &getInstance();
    VkDevice &getLogicalDevice();
    VkPhysicalDevice &getPhysicalDevice();
    VkDescriptorSetLayout &getDescriptorSetLayout();
    VkDescriptorSetLayout &getPrimitiveDescriptorSetLayout();
    VkDescriptorPool &getDescriptorPool();
    VkCommandPool &getCommandPool();
    VkQueue &getGraphicsQueue();
    uint32_t &getGraphicsQueueFamilyIndex();
    VkQueue &getVideoQueue();
    uint32_t &getVideoQueueFamilyIndex();
    VkQueue &getVideoDecodeQueue();
    uint32_t &getVideoDecodeQueueFamilyIndex();
    VkQueue &getVideoEncodeQueue();
    uint32_t &getVideoEncodeQueueFamilyIndex();
    VkPhysicalDeviceMemoryProperties &getMemoryProperties();
    VkPhysicalDeviceProperties &getDeviceProperties();
    VkPhysicalDeviceFeatures &getDeviceFeatures();
    const std::vector<const char *> &getEnabledInstanceExtensionNames() const { return enabledInstanceExtensionNamePtrs; }
    const std::vector<const char *> &getEnabledDeviceExtensionNames() const { return enabledDeviceExtensionNamePtrs; }
    const VkPhysicalDeviceFeatures2 &getEnabledFeatures2() const { return enabledFeatures2; }

private:
    void createInstance();
    void setupDebugMessenger();
    void destroyDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    std::vector<const char *> getRequiredDeviceExtensions();
    bool validationLayersEnabled;
    bool debugUtilsEnabled;

    VkInstance instance;
    VkDevice logicalDevice;
    VkPhysicalDevice physicalDevice;
    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamilyIndex;
    VkQueue videoDecodeQueue;
    uint32_t videoDecodeQueueFamilyIndex;
    VkQueue videoEncodeQueue;
    uint32_t videoEncodeQueueFamilyIndex;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayout primitiveDescriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkCommandPool commandPool;

    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceMemoryProperties memProperties;
    VkPhysicalDeviceFeatures2 enabledFeatures2{};
    VkPhysicalDeviceSynchronization2Features enabledSync2Features{};
    VkPhysicalDeviceTimelineSemaphoreFeatures enabledTimelineFeatures{};

    std::vector<std::string> enabledInstanceExtensionNames;
    std::vector<const char *> enabledInstanceExtensionNamePtrs;
    std::vector<std::string> enabledDeviceExtensionNames;
    std::vector<const char *> enabledDeviceExtensionNamePtrs;

    VkDebugUtilsMessengerEXT debugMessenger;
};
