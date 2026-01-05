#include "crop.h"
Crop::Crop()
{
}

Crop::~Crop()
{
}

void Crop::dispatch()
{
    CropPushConstants cropPushConstants{};
    cropPushConstants.outputSize = glm::vec2(static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height));
    cropPushConstants.videoSize = glm::vec2(static_cast<float>(videoImages.width), static_cast<float>(videoImages.height));
    cropPushConstants.targetOrigin = glm::vec2(originX, originY);
    cropPushConstants.targetSize = glm::vec2(targetWidth, targetHeight);
    if (overrides && overrides->useCrop)
    {
        cropPushConstants.cropOrigin = overrides->cropOrigin;
        cropPushConstants.cropSize = overrides->cropSize;
    }
    else
    {
        cropPushConstants.cropOrigin = glm::vec2(0.0f, 0.0f);
        cropPushConstants.cropSize = glm::vec2(1.0f, 1.0f);
    }
    // Clamp overlay placement to the current output extent to avoid off-screen coordinates
    uint32_t maxOverlayW = std::max(1u, swapchainExtent.width);
    uint32_t maxOverlayH = std::max(1u, swapchainExtent.height);

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CropPushConstants), &cropPushConstants);


}