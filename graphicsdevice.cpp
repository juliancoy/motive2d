#include "graphicsdevice.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
    void * /*userData*/)
{
    std::cerr << "Validation Layer: " << callbackData->pMessage << std::endl;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        std::cerr << "Vulkan error encountered, aborting." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return VK_FALSE;
}

PFN_vkCreateDebugUtilsMessengerEXT loadCreateMessenger(VkInstance instance)
{
    return reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
}

PFN_vkDestroyDebugUtilsMessengerEXT loadDestroyMessenger(VkInstance instance)
{
    return reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
}

std::string videoCodecOperationsToString(VkVideoCodecOperationFlagsKHR flags) {
    std::string s;
    if (flags & VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR) s += "H264D ";
    if (flags & VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR) s += "H265D ";
    if (flags & VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR) s += "H264E ";
    if (flags & VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR) s += "H265E ";
    if (s.empty()) return "None";
    return s;
}

} // namespace

RenderDevice::RenderDevice()
    : validationLayersEnabled(false),
      debugUtilsEnabled(false),
      instance(VK_NULL_HANDLE),
      logicalDevice(VK_NULL_HANDLE),
      physicalDevice(VK_NULL_HANDLE),
      graphicsQueue(VK_NULL_HANDLE),
      graphicsQueueFamilyIndex(0),
      videoDecodeQueue(VK_NULL_HANDLE),
      videoDecodeQueueFamilyIndex(0),
      videoEncodeQueue(VK_NULL_HANDLE),
      videoEncodeQueueFamilyIndex(0),
      descriptorSetLayout(VK_NULL_HANDLE),
      primitiveDescriptorSetLayout(VK_NULL_HANDLE),
      descriptorPool(VK_NULL_HANDLE),
      commandPool(VK_NULL_HANDLE),
      debugMessenger(VK_NULL_HANDLE),
      useGlfwExtensions(true)
{
    props = {};
    features = {};
    memProperties = {};
}

RenderDevice::~RenderDevice()
{
    shutdown();
}

void RenderDevice::initialize(bool requireWindowExtensions)
{
    useGlfwExtensions = requireWindowExtensions;
    createInstance();
    setupDebugMessenger();
    pickPhysicalDevice();
    createLogicalDevice();
    createDescriptorSetLayouts();
}

void RenderDevice::shutdown()
{
    if (logicalDevice != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(logicalDevice);
    }

    if (descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (primitiveDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(logicalDevice, primitiveDescriptorSetLayout, nullptr);
        primitiveDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(logicalDevice, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    if (logicalDevice != VK_NULL_HANDLE)
    {
        vkDestroyDevice(logicalDevice, nullptr);
        logicalDevice = VK_NULL_HANDLE;
    }

    destroyDebugMessenger();

    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

VkCommandBuffer RenderDevice::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void RenderDevice::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
}

void RenderDevice::nameVulkanObject(uint64_t handle, VkObjectType type, const char *name)
{
    auto func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(logicalDevice, "vkSetDebugUtilsObjectNameEXT"));
    if (func)
    {
        VkDebugUtilsObjectNameInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        func(logicalDevice, &info);
    }
}

void RenderDevice::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
                              VkCommandPool inCommandPool, VkQueue inGraphicsQueue)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = inCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(inGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(inGraphicsQueue);

    vkFreeCommandBuffers(logicalDevice, inCommandPool, 1, &commandBuffer);
}

void RenderDevice::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                VkBuffer &buffer, VkDeviceMemory &bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }

    vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);
}

uint32_t RenderDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

VkShaderModule RenderDevice::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

VkInstance &RenderDevice::getInstance() { return instance; }
VkDevice &RenderDevice::getLogicalDevice() { return logicalDevice; }
VkPhysicalDevice &RenderDevice::getPhysicalDevice() { return physicalDevice; }
VkDescriptorSetLayout &RenderDevice::getDescriptorSetLayout() { return descriptorSetLayout; }
VkDescriptorSetLayout &RenderDevice::getPrimitiveDescriptorSetLayout() { return primitiveDescriptorSetLayout; }
VkDescriptorPool &RenderDevice::getDescriptorPool() { return descriptorPool; }
VkCommandPool &RenderDevice::getCommandPool() { return commandPool; }
VkQueue &RenderDevice::getGraphicsQueue() { return graphicsQueue; }
uint32_t &RenderDevice::getGraphicsQueueFamilyIndex() { return graphicsQueueFamilyIndex; }
VkQueue &RenderDevice::RenderDevice::getVideoQueue() { return videoDecodeQueue; }
uint32_t &RenderDevice::RenderDevice::getVideoQueueFamilyIndex() { return videoDecodeQueueFamilyIndex; }
VkQueue &RenderDevice::getVideoDecodeQueue() { return videoDecodeQueue; }
uint32_t &RenderDevice::getVideoDecodeQueueFamilyIndex() { return videoDecodeQueueFamilyIndex; }
VkQueue &RenderDevice::getVideoEncodeQueue() { return videoEncodeQueue; }
uint32_t &RenderDevice::getVideoEncodeQueueFamilyIndex() { return videoEncodeQueueFamilyIndex; }
VkPhysicalDeviceMemoryProperties &RenderDevice::getMemoryProperties() { return memProperties; }
VkPhysicalDeviceProperties &RenderDevice::getDeviceProperties() { return props; }
VkPhysicalDeviceFeatures &RenderDevice::getDeviceFeatures() { return features; }

void RenderDevice::createInstance()
{
    validationLayersEnabled = false;
    debugUtilsEnabled = false;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Renderer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Motive";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    uint32_t loaderApiVersion = VK_API_VERSION_1_0;
    auto enumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerateInstanceVersion && enumerateInstanceVersion(&loaderApiVersion) != VK_SUCCESS)
    {
        loaderApiVersion = VK_API_VERSION_1_0;
    }
    uint32_t targetApi = VK_API_VERSION_1_2;
#ifdef VK_API_VERSION_1_4
    targetApi = VK_API_VERSION_1_4;
#elif defined(VK_API_VERSION_1_3)
    targetApi = VK_API_VERSION_1_3;
#endif
    const uint32_t requestedApiVersion = std::min(loaderApiVersion, targetApi);
    appInfo.apiVersion = requestedApiVersion;
    std::cout << "[RenderDevice] Using Vulkan API "
              << VK_VERSION_MAJOR(requestedApiVersion) << "."
              << VK_VERSION_MINOR(requestedApiVersion) << "."
              << VK_VERSION_PATCH(requestedApiVersion) << std::endl;

    const std::vector<const char *> validationLayers = {
        "VK_LAYER_KHRONOS_validation"};

    struct ExtensionRequirement
    {
        const char *name;
        bool required;
    };

    std::vector<ExtensionRequirement> extensionCandidates;
    if (useGlfwExtensions)
    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0)
        {
            throw std::runtime_error("Failed to get required GLFW extensions");
        }
        extensionCandidates.reserve(glfwExtensionCount);
        for (uint32_t i = 0; i < glfwExtensionCount; ++i)
        {
            extensionCandidates.push_back({glfwExtensions[i], true});
        }
    }
    else
    {
        const char *surfaceExtensions[] = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            "VK_KHR_xcb_surface",
            "VK_KHR_wayland_surface",
            "VK_KHR_display"};
        for (const char *ext : surfaceExtensions)
        {
            extensionCandidates.push_back({ext, false});
        }
    }

    uint32_t availableLayerCount = 0;
    vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(availableLayerCount);
    vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data());

    const auto layerSupported = [&availableLayers](const char *name) {
        return std::any_of(
            availableLayers.begin(), availableLayers.end(),
            [name](const VkLayerProperties &layer) { return strcmp(layer.layerName, name) == 0; });
    };

    if (layerSupported("VK_LAYER_KHRONOS_validation"))
    {
        validationLayersEnabled = true;
    }
    else
    {
        std::cerr << "[RenderDevice] Validation layer VK_LAYER_KHRONOS_validation not available. "
                  << "Continuing without validation layers." << std::endl;
    }

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    const auto extensionSupported = [&extensions](const char *name) {
        return std::any_of(
            extensions.begin(), extensions.end(),
            [name](const VkExtensionProperties &ext) { return strcmp(ext.extensionName, name) == 0; });
    };

    std::vector<const char *> requiredExtensions;
    requiredExtensions.reserve(extensionCandidates.size());
    for (const auto &candidate : extensionCandidates)
    {
        if (extensionSupported(candidate.name))
        {
            requiredExtensions.push_back(candidate.name);
        }
        else if (candidate.required)
        {
            throw std::runtime_error(std::string("Missing required extension: ") + candidate.name);
        }
    }

    if (validationLayersEnabled && extensionSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
    {
        requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        debugUtilsEnabled = true;
    }
    else if (validationLayersEnabled)
    {
        std::cerr << "[RenderDevice] VK_EXT_debug_utils not available. "
                  << "Validation layers will run without debug utils callbacks." << std::endl;
    }

    enabledInstanceExtensionNames.clear();
    enabledInstanceExtensionNamePtrs.clear();
    enabledInstanceExtensionNames.reserve(requiredExtensions.size());
    enabledInstanceExtensionNamePtrs.reserve(requiredExtensions.size());
    for (const char* ext : requiredExtensions)
    {
        enabledInstanceExtensionNames.emplace_back(ext);
    }
    for (const auto& ext : enabledInstanceExtensionNames)
    {
        enabledInstanceExtensionNamePtrs.push_back(ext.c_str());
    }

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    instanceCreateInfo.enabledLayerCount = validationLayersEnabled ? static_cast<uint32_t>(validationLayers.size()) : 0;
    instanceCreateInfo.ppEnabledLayerNames = validationLayersEnabled ? validationLayers.data() : nullptr;
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstanceExtensionNamePtrs.size());
    instanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensionNamePtrs.data();

    std::cout << "[RenderDevice] Enabled instance extensions (" << enabledInstanceExtensionNamePtrs.size() << "): ";
    for (size_t i = 0; i < enabledInstanceExtensionNamePtrs.size(); ++i)
    {
        std::cout << enabledInstanceExtensionNamePtrs[i];
        if (i + 1 < enabledInstanceExtensionNamePtrs.size())
        {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    VkResult instanceResult = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    if (instanceResult != VK_SUCCESS)
    {
        std::cerr << "[RenderDevice] vkCreateInstance failed with code " << instanceResult << std::endl;
        throw std::runtime_error("Failed to create Vulkan instance");
    }
}

void RenderDevice::setupDebugMessenger()
{
    if (!debugUtilsEnabled || instance == VK_NULL_HANDLE)
    {
        return;
    }
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    auto func = loadCreateMessenger(instance);
    if (func && func(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to set up debug messenger!");
    }
}

void RenderDevice::destroyDebugMessenger()
{
    if (!debugUtilsEnabled || debugMessenger == VK_NULL_HANDLE || instance == VK_NULL_HANDLE)
    {
        return;
    }
    auto func = loadDestroyMessenger(instance);
    if (func)
    {
        func(instance, debugMessenger, nullptr);
    }
    debugMessenger = VK_NULL_HANDLE;
}

void RenderDevice::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0)
    {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    std::cout << "\nAvailable Physical Devices:\n";
    std::cout << "-------------------------\n";

    for (const auto &device : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        std::cout << "Device: " << props.deviceName << "\n";
    }

    // Find a suitable device and queue
    VkPhysicalDevice selectedDevice = VK_NULL_HANDLE;
    int selectedGraphicsQueueFamilyIndex = -1;
    int selectedVideoQueueFamilyIndex = -1;
    int selectedVideoEncodeQueueFamilyIndex = -1;
    auto enumerateQueueFamiliesWithVideo = [](VkPhysicalDevice device,
                                              std::vector<VkQueueFamilyProperties2> &queueFamilies,
                                              std::vector<VkQueueFamilyVideoPropertiesKHR> &videoProps) -> uint32_t {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties2(device, &queueFamilyCount, nullptr);
        queueFamilies.resize(queueFamilyCount);
        videoProps.resize(queueFamilyCount);

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            videoProps[i] = {};
            videoProps[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
            videoProps[i].pNext = nullptr;
            queueFamilies[i] = {};
            queueFamilies[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
            queueFamilies[i].pNext = &videoProps[i];
        }

        vkGetPhysicalDeviceQueueFamilyProperties2(device, &queueFamilyCount, queueFamilies.data());
        return queueFamilyCount;
    };

    auto logQueueFamilies = [](const std::vector<VkQueueFamilyProperties2> &queueFamilies,
                               const std::vector<VkQueueFamilyVideoPropertiesKHR> &videoProps) {
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            const auto &queueFamily = queueFamilies[i].queueFamilyProperties;
            const auto &videoProp = videoProps[i];
            std::cout << "  Queue family " << i << ": " << queueFamily.queueCount << " queues, flags: "
                      << ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) ? "G " : "")
                      << ((queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) ? "C " : "")
                      << ((queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) ? "T " : "")
                      << ((queueFamily.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) ? "S " : "")
                      << ((queueFamily.queueFlags & VK_QUEUE_PROTECTED_BIT) ? "P " : "")
                      << ((queueFamily.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) ? "VD " : "")
                      << ((queueFamily.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) ? "VE " : "")
                      << "Video: " << videoCodecOperationsToString(videoProp.videoCodecOperations)
                      << std::endl;
        }
    };
    
    // Prefer discrete GPU
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        std::cout << "Evaluating device: " << props.deviceName << std::endl;
        std::vector<VkQueueFamilyProperties2> queueFamilies2;
        std::vector<VkQueueFamilyVideoPropertiesKHR> videoProperties;
        uint32_t queueFamilyCount = enumerateQueueFamiliesWithVideo(device, queueFamilies2, videoProperties);
        logQueueFamilies(queueFamilies2, videoProperties);

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            // Find a unified queue for graphics and video decode
            for (uint32_t i = 0; i < queueFamilyCount; i++) {
                const auto& queueFamily = queueFamilies2[i].queueFamilyProperties;
                if (queueFamily.queueCount > 0 && 
                    (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && 
                    (queueFamily.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR)) {
                    selectedDevice = device;
                    selectedGraphicsQueueFamilyIndex = i;
                    selectedVideoQueueFamilyIndex = i; // Same queue for video
                    // Also look for encode queue
                    for (uint32_t j = 0; j < queueFamilyCount; j++) {
                        const auto& queueFamily2 = queueFamilies2[j].queueFamilyProperties;
                        if (queueFamily2.queueCount > 0 && (queueFamily2.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR)) {
                            selectedVideoEncodeQueueFamilyIndex = j;
                            break;
                        }
                    }
                    std::cout << "  -> Selected discrete GPU with unified queue family " << i << std::endl;
                    goto device_found;
                }
            }
            // If no unified, try to find separate queues on discrete GPU
            int potentialGraphicsQueue = -1;
            int potentialVideoQueue = -1;
            int potentialVideoEncodeQueue = -1;
            for (uint32_t i = 0; i < queueFamilyCount; i++) {
                const auto& queueFamily = queueFamilies2[i].queueFamilyProperties;
                if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                    potentialGraphicsQueue = i;
                }
                if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR)) {
                    potentialVideoQueue = i;
                }
                if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR)) {
                    potentialVideoEncodeQueue = i;
                }
            }
            if (potentialGraphicsQueue != -1 && potentialVideoQueue != -1) {
                selectedDevice = device;
                selectedGraphicsQueueFamilyIndex = potentialGraphicsQueue;
                selectedVideoQueueFamilyIndex = potentialVideoQueue;
                selectedVideoEncodeQueueFamilyIndex = potentialVideoEncodeQueue;
                std::cout << "  -> Selected discrete GPU with separate graphics (" << potentialGraphicsQueue 
                          << ") and video decode (" << potentialVideoQueue << ") queue families.";
                if (potentialVideoEncodeQueue != -1) {
                    std::cout << " Found video encode queue family " << potentialVideoEncodeQueue;
                }
                std::cout << std::endl;
                goto device_found;
            }
        }
    }

    // Fallback to any device with a unified queue
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        std::vector<VkQueueFamilyProperties2> queueFamilies2;
        std::vector<VkQueueFamilyVideoPropertiesKHR> videoProperties;
        uint32_t queueFamilyCount = enumerateQueueFamiliesWithVideo(device, queueFamilies2, videoProperties);
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            const auto& queueFamily = queueFamilies2[i].queueFamilyProperties;
            if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR)) {
                selectedDevice = device;
                selectedGraphicsQueueFamilyIndex = i;
                selectedVideoQueueFamilyIndex = i;
                std::cout << "  -> Selected device '" << props.deviceName << "' with unified queue family " << i << std::endl;
                goto device_found;
            }
        }
    }

    // Fallback to any device with separate graphics and video queues
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        std::vector<VkQueueFamilyProperties2> queueFamilies2;
        std::vector<VkQueueFamilyVideoPropertiesKHR> videoProperties;
        uint32_t queueFamilyCount = enumerateQueueFamiliesWithVideo(device, queueFamilies2, videoProperties);
        int potentialGraphicsQueue = -1;
        int potentialVideoQueue = -1;
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            const auto& queueFamily = queueFamilies2[i].queueFamilyProperties;
            if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                potentialGraphicsQueue = i;
            }
            if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR)) {
                potentialVideoQueue = i;
            }
        }
        if (potentialGraphicsQueue != -1 && potentialVideoQueue != -1) {
            selectedDevice = device;
            selectedGraphicsQueueFamilyIndex = potentialGraphicsQueue;
            selectedVideoQueueFamilyIndex = potentialVideoQueue;
            std::cout << "  -> Selected device '" << props.deviceName << "' with separate graphics (" << potentialGraphicsQueue 
                      << ") and video (" << potentialVideoQueue << ") queue families." << std::endl;
            goto device_found;
        }
    }

    // Last resort fallback: any device with at least a graphics queue (no video)
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        std::vector<VkQueueFamilyProperties2> queueFamilies2;
        std::vector<VkQueueFamilyVideoPropertiesKHR> videoProperties;
        uint32_t queueFamilyCount = enumerateQueueFamiliesWithVideo(device, queueFamilies2, videoProperties);
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            const auto& queueFamily = queueFamilies2[i].queueFamilyProperties;
            if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                selectedDevice = device;
                selectedGraphicsQueueFamilyIndex = i;
                selectedVideoQueueFamilyIndex = -1; // No video queue found
                std::cout << "  -> Selected device '" << props.deviceName << "' with graphics-only queue family " << i << std::endl;
                goto device_found;
            }
        }
    }


device_found:
    if (selectedDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
    physicalDevice = selectedDevice;
    graphicsQueueFamilyIndex = selectedGraphicsQueueFamilyIndex;
    videoDecodeQueueFamilyIndex = selectedVideoQueueFamilyIndex;
    videoEncodeQueueFamilyIndex = selectedVideoEncodeQueueFamilyIndex;

    std::cout << "\nSelected Device Properties:\n";
    std::cout << "-------------------------\n";
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    std::cout << "Using graphics queue family: " << graphicsQueueFamilyIndex;
    if (videoDecodeQueueFamilyIndex != -1) {
        std::cout << ", video decode queue family: " << videoDecodeQueueFamilyIndex;
    }
    if (videoEncodeQueueFamilyIndex != -1) {
        std::cout << ", video encode queue family: " << videoEncodeQueueFamilyIndex;
    }
    std::cout << "\n";

    vkGetPhysicalDeviceFeatures(physicalDevice, &features);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    std::cout << "\nDevice: " << props.deviceName << "\n";
    std::cout << "  API Version: " << VK_VERSION_MAJOR(props.apiVersion) << "."
              << VK_VERSION_MINOR(props.apiVersion) << "."
              << VK_VERSION_PATCH(props.apiVersion) << "\n";
    std::cout << "  Driver Version: " << props.driverVersion << "\n";
    std::cout << "  Vendor ID: " << props.vendorID << "\n";
    std::cout << "  Device ID: " << props.deviceID << "\n";
    std::cout << "  Device Type: ";

    switch (props.deviceType)
    {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        std::cout << "Integrated GPU";
        break;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        std::cout << "Discrete GPU";
        break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        std::cout << "Virtual GPU";
        break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        std::cout << "CPU";
        break;
    default:
        std::cout << "Other";
        break;
    }
    std::cout << "\n";

    std::cout << "\n  Important Limits:\n";
    std::cout << "    maxImageDimension2D: " << props.limits.maxImageDimension2D << "\n";
    std::cout << "    maxPushConstantsSize: " << props.limits.maxPushConstantsSize << " bytes\n";
    std::cout << "    maxMemoryAllocationCount: " << props.limits.maxMemoryAllocationCount << "\n";
    std::cout << "    maxSamplerAllocationCount: " << props.limits.maxSamplerAllocationCount << "\n";
    std::cout << "    bufferImageGranularity: " << props.limits.bufferImageGranularity << " bytes\n";

    std::cout << "\n  Important Features:\n";
    std::cout << "    geometryShader: " << (features.geometryShader ? "Yes" : "No") << "\n";
    std::cout << "    tessellationShader: " << (features.tessellationShader ? "Yes" : "No") << "\n";
    std::cout << "    samplerAnisotropy: " << (features.samplerAnisotropy ? "Yes" : "No") << "\n";
    std::cout << "    textureCompressionBC: " << (features.textureCompressionBC ? "Yes" : "No") << "\n";

    std::cout << "\n  Memory Heaps (" << memProperties.memoryHeapCount << "):\n";
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i)
    {
        std::cout << "    Heap " << i << ": " << memProperties.memoryHeaps[i].size / (1024 * 1024) << " MB";
        if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        {
            std::cout << " (Device Local)";
        }
        std::cout << "\n";
    }

    if (physicalDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Physical device not initialized");
    }
}

std::vector<const char *> RenderDevice::getRequiredDeviceExtensions() {
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> availableDevExt(devExtCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, availableDevExt.data());

    const auto deviceExtensionSupported = [&availableDevExt](const char *name) {
        return std::any_of(
            availableDevExt.begin(), availableDevExt.end(),
            [name](const VkExtensionProperties &ext) { return strcmp(ext.extensionName, name) == 0; });
    };

    const char *baseExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
        VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    };

    std::vector<const char *> requiredExtensions;
    requiredExtensions.reserve(20);

    for (const char* ext : baseExtensions)
    {
        if (deviceExtensionSupported(ext))
        {
            requiredExtensions.push_back(ext);
        }
        else if (useGlfwExtensions)
        {
            throw std::runtime_error(std::string("Required device extension missing: ") + ext);
        }
        else
        {
            std::cerr << "[RenderDevice] Optional device extension not available: " << ext << "\n";
        }
    }

    if (deviceExtensionSupported(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
    }
    if (deviceExtensionSupported(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
        requiredExtensions.push_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
    }

    return requiredExtensions;
}

void RenderDevice::createLogicalDevice()
{
    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies;

    uniqueQueueFamilies.insert(graphicsQueueFamilyIndex);
    if (videoDecodeQueueFamilyIndex != -1 && videoDecodeQueueFamilyIndex != graphicsQueueFamilyIndex) {
        uniqueQueueFamilies.insert(videoDecodeQueueFamilyIndex);
    }
    if (videoEncodeQueueFamilyIndex != -1 && videoEncodeQueueFamilyIndex != graphicsQueueFamilyIndex && 
        videoEncodeQueueFamilyIndex != videoDecodeQueueFamilyIndex) {
        uniqueQueueFamilies.insert(videoEncodeQueueFamilyIndex);
    }

    for (uint32_t queueFamilyIndex : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    enabledTimelineFeatures = {};
    enabledTimelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    enabledTimelineFeatures.timelineSemaphore = VK_TRUE;

    enabledSync2Features = {};
    enabledSync2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    enabledSync2Features.synchronization2 = VK_TRUE;
    enabledTimelineFeatures.pNext = &enabledSync2Features;

    enabledFeatures2 = {};
    enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabledFeatures2.features.samplerAnisotropy = VK_TRUE;
    enabledFeatures2.features.sampleRateShading = VK_TRUE;
    enabledFeatures2.pNext = &enabledTimelineFeatures;

    std::vector<const char *> enabledDevExt = getRequiredDeviceExtensions();
    enabledDeviceExtensionNames.clear();
    enabledDeviceExtensionNamePtrs.clear();
    enabledDeviceExtensionNames.reserve(enabledDevExt.size());
    enabledDeviceExtensionNamePtrs.reserve(enabledDevExt.size());
    for (const char* ext : enabledDevExt)
    {
        enabledDeviceExtensionNames.emplace_back(ext);
    }
    for (const auto& ext : enabledDeviceExtensionNames)
    {
        enabledDeviceExtensionNamePtrs.push_back(ext.c_str());
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = nullptr;
    createInfo.pNext = &enabledFeatures2;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensionNamePtrs.size());
    createInfo.ppEnabledExtensionNames = enabledDeviceExtensionNamePtrs.data();

    std::cout << "[RenderDevice] Enabled device extensions (" << enabledDeviceExtensionNamePtrs.size() << "): ";
    for (size_t i = 0; i < enabledDeviceExtensionNamePtrs.size(); ++i)
    {
        std::cout << enabledDeviceExtensionNamePtrs[i];
        if (i + 1 < enabledDeviceExtensionNamePtrs.size())
        {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    const std::vector<const char *> validationLayers = {
        "VK_LAYER_KHRONOS_validation"};
    if (validationLayersEnabled) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    VkResult deviceCreateResult = vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice);
    if (deviceCreateResult != VK_SUCCESS)
    {
        std::cerr << "[RenderDevice] vkCreateDevice failed with code " << deviceCreateResult << std::endl;
        throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(logicalDevice, graphicsQueueFamilyIndex, 0, &graphicsQueue);
    if (videoDecodeQueueFamilyIndex != -1) {
        vkGetDeviceQueue(logicalDevice, videoDecodeQueueFamilyIndex, 0, &videoDecodeQueue);
        std::cout << "[RenderDevice] Using dedicated video decode queue family " << videoDecodeQueueFamilyIndex << std::endl;
    } else {
        videoDecodeQueue = graphicsQueue; // Fallback if no specific video decode queue was selected
        std::cout << "[RenderDevice] No dedicated video decode queue found, falling back to graphics queue" << std::endl;
    }
    if (videoEncodeQueueFamilyIndex != -1) {
        vkGetDeviceQueue(logicalDevice, videoEncodeQueueFamilyIndex, 0, &videoEncodeQueue);
        std::cout << "[RenderDevice] Using dedicated video encode queue family " << videoEncodeQueueFamilyIndex << std::endl;
    } else {
        videoEncodeQueue = graphicsQueue; // Fallback if no specific video encode queue was selected
        std::cout << "[RenderDevice] No dedicated video encode queue found, falling back to graphics queue" << std::endl;
    }


    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create command pool!");
    }

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 300;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 200;

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPoolInfo.maxSets = 100;
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(logicalDevice, &descriptorPoolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

void RenderDevice::createDescriptorSetLayouts()
{
    if (descriptorSetLayout == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 2> globalBindings{};
        globalBindings[0].binding = 0;
        globalBindings[0].descriptorCount = 1;
        globalBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        globalBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        globalBindings[1].binding = 1;
        globalBindings[1].descriptorCount = 1;
        globalBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        globalBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(globalBindings.size());
        layoutInfo.pBindings = globalBindings.data();

        if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create global descriptor set layout!");
        }
        nameVulkanObject((uint64_t)descriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "globalDescriptorSetLayout");
    }

    if (primitiveDescriptorSetLayout == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 4> primitiveBindings{};
        primitiveBindings[0].binding = 0;
        primitiveBindings[0].descriptorCount = 1;
        primitiveBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        primitiveBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        primitiveBindings[1].binding = 1;
        primitiveBindings[1].descriptorCount = 1;
        primitiveBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        primitiveBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        primitiveBindings[2].binding = 2;
        primitiveBindings[2].descriptorCount = 1;
        primitiveBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        primitiveBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        primitiveBindings[3].binding = 3;
        primitiveBindings[3].descriptorCount = 1;
        primitiveBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        primitiveBindings[3].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo primitiveLayoutInfo{};
        primitiveLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        primitiveLayoutInfo.bindingCount = static_cast<uint32_t>(primitiveBindings.size());
        primitiveLayoutInfo.pBindings = primitiveBindings.data();

        if (vkCreateDescriptorSetLayout(logicalDevice, &primitiveLayoutInfo, nullptr, &primitiveDescriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create primitive descriptor set layout!");
        }
        nameVulkanObject((uint64_t)primitiveDescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "primitiveDescriptorSetLayout");
    }
}
