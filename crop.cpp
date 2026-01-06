#include "crop.h"
#include "debug_logging.h"
#include <iostream>

Crop::Crop()
{
}

Crop::~Crop()
{
}

void Crop::run()
{
    if (renderDebugEnabled())
    {
        std::cout << "[Crop] run() called" << std::endl;
    }
    // TODO: Implement proper crop computation
    // For now, set default values to avoid compilation errors
    cropPushConstants.outputSize = glm::vec2(1920.0f, 1080.0f);
    cropPushConstants.videoSize = glm::vec2(1920.0f, 1080.0f);
    cropPushConstants.targetOrigin = glm::vec2(0.0f, 0.0f);
    cropPushConstants.targetSize = glm::vec2(1920.0f, 1080.0f);
    cropPushConstants.cropOrigin = glm::vec2(0.0f, 0.0f);
    cropPushConstants.cropSize = glm::vec2(1.0f, 1.0f);
}
