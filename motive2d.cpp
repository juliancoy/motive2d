#include "motive2d.h"

#include "crop.h"
#include "decoder_vulkan.h"
#include "debug_logging.h"
#include "engine2d.h"
#include "fps.h"
#include "pose_overlay.h"
// #include "rect_overlay.h" // Already included via motive2d.h
#include "rgba2nv12.h"
#include "scrubber.h"
#include "subtitle.h"
#include "utils.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

std::mutex g_stageMutex;
std::condition_variable g_stageCv;

Motive2D::Motive2D(CliOptions cliOptions)
    : options(cliOptions)
{
    // Enable debug logging if requested
    if (options.debugLogging)
    {
        setDebugLoggingEnabled(true);
        setRenderDebugEnabled(true);
    }

    engine = new Engine2D();
    if (!engine || !engine->initialize())
    {
        throw std::runtime_error("Failed to initialize Vulkan engine");
    }

    const std::filesystem::path subtitlePath =
        cliOptions.videoPath.parent_path() / (cliOptions.videoPath.stem().string() + ".json");
    subtitle = new Subtitle(subtitlePath, engine);
    rectOverlay = new RectOverlay(engine);
    poseOverlay = new PoseOverlay(engine);
    crop = new Crop();
    blackSampler = VK_NULL_HANDLE;
    scrubber = new Scrubber(engine);
    fpsOverlay = new FpsOverlay(engine);
    // colorGrading and colorGradingUi will be created after windows

    bool requireGraphicsQueue = true;
    bool debugLogging = cliOptions.debugLogging;

    if (cliOptions.gpuDecode)
    {
        requireGraphicsQueue = true;

        std::cout << "[Motive2D] GPU decode requested (Vulkan)" << std::endl;
    }
    else{
        throw std::runtime_error("Only GPU decode is supported in this build");
    }

    decoder = new DecoderVulkan(cliOptions.videoPath, engine);

    // Start async decoding to ensure frames are available
    if (decoder)
    {
        decoder->startAsyncDecoding(2); // Buffer 2 frames

        // Create nv12toBGR pipeline after decoder dimensions are known
        int width = decoder->getWidth();
        int height = decoder->getHeight();
        uint32_t groupX = (static_cast<uint32_t>(width) + 15) / 16;
        uint32_t groupY = (static_cast<uint32_t>(height) + 15) / 16;
        nv12toBGRPipeline = new nv12toBGR(engine, groupX, groupY);
        nv12toBGRPipeline->createPipeline();
    }

    // Create windows based on options
    if (options.showInput)
    {
        inputWindow = new Display2D(engine, 800, 600, "Input");
        windows.emplace_back(inputWindow);
        std::cout << "[Motive2D] Created input window" << std::endl;
    }
    if (options.showRegion)
    {
        regionWindow = new Display2D(engine, 800, 600, "Region");
        windows.emplace_back(regionWindow);
        std::cout << "[Motive2D] Created region window" << std::endl;
    }
    if (options.showGrading)
    {
        gradingWindow = new Display2D(engine, 800, 600, "Grading");
        windows.emplace_back(gradingWindow);
        std::cout << "[Motive2D] Created grading window" << std::endl;
        // Create ColorGrading attached to gradingWindow
        colorGrading = new ColorGrading(gradingWindow);
        // TODO: Create ColorGradingUi with appropriate parameters
        // For now, create a dummy GradingSettings and SliderLayout
        // colorGradingUi = new ColorGradingUi(...);
    }

    // Create pipeline test directory if needed
    if (options.pipelineTest)
    {
        std::filesystem::create_directories(options.pipelineTestDir);
        std::cout << "[Motive2D] Pipeline test mode enabled. Output directory: "
                  << options.pipelineTestDir << std::endl;
    }

    // Create synchronization objects for the pipeline
    createSynchronizationObjects();

    // Create compute resources (images, descriptor sets)
    createComputeResources();
}

Motive2D::~Motive2D()
{
    destroyComputeResources();
    destroySynchronizationObjects();

    delete subtitle;
    delete rectOverlay;
    delete poseOverlay;
    delete crop;
    delete nv12toBGRPipeline;
    delete scrubber;
    delete fpsOverlay;
    delete decoder;
    delete engine;
}

void Motive2D::createSynchronizationObjects()
{
    frames.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        FrameResources &frame = frames[i];

        // Create command buffer
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = engine->renderDevice.getCommandPool();
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(engine->logicalDevice, &allocInfo, &frame.commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffer for frame");
        }

        // Create fence (unsignaled)
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start signaled so first wait doesn't block
        if (vkCreateFence(engine->logicalDevice, &fenceInfo, nullptr, &frame.fence) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create fence for frame");
        }

        // Create decode ready semaphore (timeline)
        VkSemaphoreTypeCreateInfo timelineCreateInfo{};
        timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineCreateInfo.initialValue = 0;

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreInfo.pNext = &timelineCreateInfo;

        if (vkCreateSemaphore(engine->logicalDevice, &semaphoreInfo, nullptr, &frame.decodeReadySemaphore) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create decode ready timeline semaphore");
        }

        // Create compute complete semaphore (binary)
        VkSemaphoreCreateInfo binaryInfo{};
        binaryInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(engine->logicalDevice, &binaryInfo, nullptr, &frame.computeCompleteSemaphore) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create compute complete binary semaphore");
        }

        frame.decodeSemaphoreValue = 0;
    }

    std::cout << "[Motive2D] Created synchronization objects for " << MAX_FRAMES_IN_FLIGHT << " frames" << std::endl;
}

void Motive2D::destroySynchronizationObjects()
{
    if (!engine)
        return;

    vkDeviceWaitIdle(engine->logicalDevice);

    for (auto &frame : frames)
    {
        if (frame.commandBuffer != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(engine->logicalDevice, engine->renderDevice.getCommandPool(), 1, &frame.commandBuffer);
        }
        if (frame.fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(engine->logicalDevice, frame.fence, nullptr);
        }
        if (frame.decodeReadySemaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(engine->logicalDevice, frame.decodeReadySemaphore, nullptr);
        }
        if (frame.computeCompleteSemaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(engine->logicalDevice, frame.computeCompleteSemaphore, nullptr);
        }
    }
    frames.clear();
}

void Motive2D::createComputeResources()
{
    if (!decoder || !nv12toBGRPipeline)
    {
        std::cerr << "[Motive2D] Cannot create compute resources: decoder or pipeline not initialized" << std::endl;
        return;
    }

    int width = decoder->getWidth();
    int height = decoder->getHeight();

    // Create intermediate BGR images (one per frame in flight)
    bgrImages.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        BGRImage &bgrImage = bgrImages[i];
        bgrImage.width = width;
        bgrImage.height = height;

        // Create image
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM; // RGBA8 for storage
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(engine->logicalDevice, &imageInfo, nullptr, &bgrImage.image) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create BGR image");
        }

        // Allocate memory
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(engine->logicalDevice, bgrImage.image, &memReqs);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = engine->findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(engine->logicalDevice, &allocInfo, nullptr, &bgrImage.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate BGR image memory");
        }

        vkBindImageMemory(engine->logicalDevice, bgrImage.image, bgrImage.memory, 0);

        // Create image view
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = bgrImage.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(engine->logicalDevice, &viewInfo, nullptr, &bgrImage.imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create BGR image view");
        }
    }

    // Create descriptor pool for compute pipelines
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT * 3; // 3 images per frame (Y, UV, BGR)

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;

    if (vkCreateDescriptorPool(engine->logicalDevice, &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create compute descriptor pool");
    }

    // Allocate descriptor sets for nv12toBGR pipeline
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, nv12toBGRPipeline->getDescriptorSetLayout());
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = computeDescriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();

    std::vector<VkDescriptorSet> descriptorSets(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(engine->logicalDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate nv12toBGR descriptor sets");
    }

    // Assign descriptor sets to frame resources
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        frames[i].nv12toBGRDescriptorSet = descriptorSets[i];
    }

    std::cout << "[Motive2D] Created compute resources: " << MAX_FRAMES_IN_FLIGHT << " BGR images and descriptor sets" << std::endl;
}

void Motive2D::destroyComputeResources()
{
    if (!engine)
        return;

    vkDeviceWaitIdle(engine->logicalDevice);

    // Destroy BGR images
    for (auto &bgrImage : bgrImages)
    {
        if (bgrImage.imageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(engine->logicalDevice, bgrImage.imageView, nullptr);
        }
        if (bgrImage.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(engine->logicalDevice, bgrImage.image, nullptr);
        }
        if (bgrImage.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(engine->logicalDevice, bgrImage.memory, nullptr);
        }
    }
    bgrImages.clear();

    // Destroy descriptor pool (automatically frees descriptor sets)
    if (computeDescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(engine->logicalDevice, computeDescriptorPool, nullptr);
        computeDescriptorPool = VK_NULL_HANDLE;
    }
}

void Motive2D::recordComputeCommands(VkCommandBuffer commandBuffer, int frameIndex)
{
    // Debug print to verify this function is called
    std::cout << "[Motive2D] recordComputeCommands called for frame " << frameIndex << std::endl;

    // TODO: Implement recording of all compute stages
    // For now, just begin and end the command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to begin recording compute command buffer");
    }

    // TODO: Add pipeline barriers and compute dispatches here
    // 1. Barrier for video image
    // 2. Dispatch NV12toBGR
    // 3. Barrier for crop
    // 4. Dispatch Crop
    // 5. Barrier for color grading
    // 6. Dispatch ColorGrading
    // 7. Barrier for swapchain images
    // 8. Dispatch overlays

    // Temporary: Call pipeline run methods to see debug output
    if (renderDebugEnabled())
    {
        std::cout << "[Motive2D] Starting compute pipeline for frame " << frameIndex << std::endl;
    }

    // Call Crop::run() if crop exists
    if (crop)
    {
        crop->run(); // This will print "[Crop] run() called" if debug enabled
    }

    // Call ColorGrading::dispatch() if colorGrading exists
    if (colorGrading && colorGrading->pipeline() != VK_NULL_HANDLE)
    {
        // We need proper parameters - for now just log
        if (renderDebugEnabled())
        {
            std::cout << "[Motive2D] Would dispatch ColorGrading pipeline" << std::endl;
        }
        // colorGrading->dispatch(commandBuffer, colorGrading->pipelineLayout_,
        //                       VK_NULL_HANDLE, 1, 1);
    }

    // Dispatch nv12toBGR pipeline if created
    if (nv12toBGRPipeline && nv12toBGRPipeline->pipeline != VK_NULL_HANDLE)
    {
        // Set command buffer and descriptor set
        nv12toBGRPipeline->commandBuffer = commandBuffer;
        nv12toBGRPipeline->descriptorSet = frames[frameIndex].nv12toBGRDescriptorSet;
        // Set push constants (default values)
        nv12toBGRPipeline->pushConstants.rgbaSize = glm::ivec2(decoder->getWidth(), decoder->getHeight());
        nv12toBGRPipeline->pushConstants.uvSize = glm::ivec2(decoder->getWidth() / 2, decoder->getHeight() / 2);
        nv12toBGRPipeline->pushConstants.colorSpace = 0; // BT.601
        nv12toBGRPipeline->pushConstants.colorRange = 1; // MPEG limited
        // Run the pipeline (will bind pipeline, descriptor sets, push constants, dispatch)
        nv12toBGRPipeline->run();
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to end recording compute command buffer");
    }

    if (renderDebugEnabled())
    {
        std::cout << "[Motive2D] Recorded compute commands for frame " << frameIndex << std::endl;
    }
}

void Motive2D::updateDescriptorSets(int frameIndex)
{
    if (!decoder || !nv12toBGRPipeline || frameIndex >= bgrImages.size())
    {
        if (renderDebugEnabled())
        {
            std::cerr << "[Motive2D] Cannot update descriptor sets: invalid state" << std::endl;
        }
        return;
    }

    // Get luma and chroma image views from decoder
    VkImageView lumaView = decoder->externalLumaView;
    VkImageView chromaView = decoder->externalChromaView;

    // If decoder hasn't uploaded frames yet, skip update
    if (lumaView == VK_NULL_HANDLE || chromaView == VK_NULL_HANDLE)
    {
        if (renderDebugEnabled())
        {
            std::cout << "[Motive2D] Skipping descriptor set update: decoder images not ready" << std::endl;
        }
        return;
    }

    // Get BGR output image for this frame
    const BGRImage &bgrImage = bgrImages[frameIndex];

    // Update nv12toBGR descriptor set
    VkDescriptorSet descriptorSet = frames[frameIndex].nv12toBGRDescriptorSet;

    std::array<VkDescriptorImageInfo, 3> imageInfos{};

    // Binding 0: Y plane (luma)
    imageInfos[0].imageView = lumaView;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[0].sampler = VK_NULL_HANDLE; // Not used for storage images

    // Binding 1: UV plane (chroma)
    imageInfos[1].imageView = chromaView;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[1].sampler = VK_NULL_HANDLE;

    // Binding 2: BGR output
    imageInfos[2].imageView = bgrImage.imageView;
    imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[2].sampler = VK_NULL_HANDLE;

    std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

    // Write Y plane
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &imageInfos[0];

    // Write UV plane
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfos[1];

    // Write BGR output
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pImageInfo = &imageInfos[2];

    vkUpdateDescriptorSets(engine->logicalDevice,
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(),
                           0, nullptr);

    if (renderDebugEnabled())
    {
        std::cout << "[Motive2D] Updated descriptor sets for frame " << frameIndex
                  << " (luma=" << lumaView << ", chroma=" << chromaView
                  << ", bgr=" << bgrImage.imageView << ")" << std::endl;
    }
}

void Motive2D::run()
{
    // If pipeline test is enabled, export a test frame and exit
    if (options.pipelineTest)
    {
        exportPipelineTestFrame();
        return;
    }

    // Main rendering loop with proper synchronization
    while (!windows.empty())
    {
        FrameResources &frame = frames[currentFrame];

        // 1. Wait for previous frame's GPU work to complete
        vkWaitForFences(engine->logicalDevice, 1, &frame.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(engine->logicalDevice, 1, &frame.fence);

        // 2. Try to acquire a decoded frame
        DecodedFrame decodedFrame;
        bool gotFrame = false;
        if (decoder)
        {
            gotFrame = decoder->acquireDecodedFrame(decodedFrame);
        }

        if (gotFrame)
        {
            // 3. Upload decoded frame to Vulkan images
            if (!decoder->uploadDecodedFrame(engine, decodedFrame))
            {
                if (renderDebugEnabled())
                {
                    std::cerr << "[Motive2D] Failed to upload decoded frame" << std::endl;
                }
            }

            // 4. Update descriptor sets with new frame
            updateDescriptorSets(currentFrame);

            // 4. Record compute commands for this frame
            recordComputeCommands(frame.commandBuffer, currentFrame);

            // 5. Submit compute work to graphics queue
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &frame.commandBuffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &frame.computeCompleteSemaphore;

            // If decoder uses separate queue, we'd wait on decodeReadySemaphore
            // For now, assume decoder uses same queue (software decoding)

            if (vkQueueSubmit(engine->graphicsQueue, 1, &submitInfo, frame.fence) != VK_SUCCESS)
            {
                std::cerr << "[Motive2D] Failed to submit compute work" << std::endl;
            }

            // 6. Present each window
            for (auto &window : windows)
            {
                // TODO: Pass computeCompleteSemaphore to window for proper synchronization
                // For now, just call renderFrame (which will handle presentation internally)
                window->renderFrame();
            }
        }
        else
        {
            // No frame available, just render blank frames
            for (auto &window : windows)
            {
                window->renderFrame();
            }
        }

        // 7. Poll events and advance frame index
        glfwPollEvents();

        // Check if any window should close
        bool anyWindowOpen = false;
        for (auto &win : windows)
        {
            if (!win->shouldClose())
            {
                anyWindowOpen = true;
                break;
            }
        }
        if (!anyWindowOpen)
        {
            break;
        }

        // Advance to next frame (triple buffering)
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Simple frame pacing (should be vsync-based in production)
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 fps
    }
}

void Motive2D::exportPipelineTestFrame()
{
    std::cout << "[Motive2D] Exporting pipeline test frames..." << std::endl;

    std::filesystem::path basePath = options.pipelineTestDir;

    // Variables that will be used for all stages
    int width = 640;
    int height = 480;
    int channels = 4;
    std::vector<uint8_t> testImage;

    // Try to decode and save an actual video frame as input stage
    bool savedRealFrame = false;

    // Seek to beginning of video
    if (decoder->seek(0.0f))
    {
        std::cout << "[Motive2D] Successfully sought to beginning of video" << std::endl;

        // Start async decoding
        if (decoder->startAsyncDecoding(2)) // Buffer 2 frames
        {
            // Wait for decoding to start and buffer a frame
            DecodedFrame frame;
            bool gotFrame = false;
            for (int i = 0; i < 10; ++i)
            {
                if (decoder->acquireDecodedFrame(frame))
                {
                    gotFrame = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            width = decoder->getWidth();
            height = decoder->getHeight();

            if (gotFrame && !frame.buffer.empty() && width > 0 && height > 0)
            {
                std::cout << "[Motive2D] Decoded frame: " << width << "x" << height
                          << ", buffer size: " << frame.buffer.size() << " bytes" << std::endl;

                // Convert YUV to RGB for PNG saving
                // First check if it's NV12 format (common for hardware decoding)
                size_t yPlaneBytes = width * height;                  // Assuming 8-bit YUV
                size_t uvPlaneBytes = (width / 2) * (height / 2) * 2; // NV12 has interleaved UV plane

                // Try to convert NV12 to BGR
                std::vector<uint8_t> bgrImage;
                if (convertNv12ToBgr(frame.buffer.data(), yPlaneBytes, uvPlaneBytes,
                                     width, height, bgrImage))
                {
                    // Convert BGR to RGBA for PNG saving
                    testImage.resize(width * height * 4);
                    for (size_t i = 0; i < bgrImage.size() / 3; ++i)
                    {
                        testImage[i * 4 + 0] = bgrImage[i * 3 + 2]; // R
                        testImage[i * 4 + 1] = bgrImage[i * 3 + 1]; // G
                        testImage[i * 4 + 2] = bgrImage[i * 3 + 0]; // B
                        testImage[i * 4 + 3] = 255;                 // A
                    }

                    std::filesystem::path inputPath = basePath / "input_stage.png";
                    if (saveImageToPNG(inputPath, testImage.data(), width, height, 4))
                    {
                        std::cout << "[Motive2D] Saved actual video frame as input stage: "
                                  << inputPath << " (" << width << "x" << height << ")" << std::endl;
                        savedRealFrame = true;
                    }
                }
                else
                {
                    std::cout << "[Motive2D] Failed to convert NV12 to BGR" << std::endl;
                }

                // Also save raw YUV data for reference
                std::filesystem::path yuvPath = basePath / "frame.yuv";
                if (saveRawFrameData(yuvPath, frame.buffer.data(), frame.buffer.size()))
                {
                    std::cout << "[Motive2D] Saved raw YUV data: " << yuvPath
                              << " (" << frame.buffer.size() << " bytes)" << std::endl;
                }
            }
            else
            {
                std::cout << "[Motive2D] Decoded frame has empty buffer or invalid dimensions" << std::endl;
            }
        }
        else
        {
            std::cout << "[Motive2D] Failed to start async decoding" << std::endl;
        }
    }
    else
    {
        std::cout << "[Motive2D] Failed to seek to beginning of video" << std::endl;
    }

    // If we couldn't save a real frame, fall back to test pattern
    if (!savedRealFrame)
    {
        std::cout << "[Motive2D] Couldn't Decode!" << std::endl;
        return;
    }

    // Region stage (with some modification)
    std::vector<uint8_t> regionImage = testImage;
    // Add a red rectangle in the middle
    int rectX = width / 4;
    int rectY = height / 4;
    int rectW = width / 2;
    int rectH = height / 2;

    for (int y = rectY; y < rectY + rectH; ++y)
    {
        for (int x = rectX; x < rectX + rectW; ++x)
        {
            if (x == rectX || x == rectX + rectW - 1 || y == rectY || y == rectY + rectH - 1)
            {
                int idx = (y * width + x) * channels;
                regionImage[idx + 0] = 255; // R
                regionImage[idx + 1] = 0;   // G
                regionImage[idx + 2] = 0;   // B
            }
        }
    }

    std::filesystem::path regionPath = basePath / "region_stage.png";
    if (saveImageToPNG(regionPath, regionImage.data(), width, height, channels))
    {
        std::cout << "[Motive2D] Saved region stage: " << regionPath << std::endl;
    }

    // Grading stage (with color adjustment)
    std::vector<uint8_t> gradingImage = testImage;
    // Apply a simple color grading (boost reds)
    for (int i = 0; i < width * height * channels; i += channels)
    {
        gradingImage[i + 0] = std::min(255, static_cast<int>(gradingImage[i + 0] * 1.2f)); // Boost red
        gradingImage[i + 1] = std::max(0, static_cast<int>(gradingImage[i + 1] * 0.9f));   // Reduce green
    }

    std::filesystem::path gradingPath = basePath / "grading_stage.png";
    if (saveImageToPNG(gradingPath, gradingImage.data(), width, height, channels))
    {
        std::cout << "[Motive2D] Saved grading stage: " << gradingPath << std::endl;
    }

    // Also save as JPG
    std::filesystem::path jpgPath = basePath / "output.jpg";
    if (saveImageToJPG(jpgPath, testImage.data(), width, height, channels, 90))
    {
        std::cout << "[Motive2D] Saved JPG output: " << jpgPath << std::endl;
    }

    std::cout << "[Motive2D] Pipeline test export complete." << std::endl;
}
