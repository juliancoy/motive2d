
#include <vulkan/vulkan.h>
class RectOverlay
{
public:
    Engine2D* engine = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    RectOverlay(Engine2D* engine);
    ~RectOverlay();
    void run(
        const glm::vec2& rectCenter,
        const glm::vec2& rectSize,
        float outerThickness,
        float innerThickness,
        bool detectionEnabled,
        bool overlayActive);
    

};