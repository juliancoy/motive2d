# Motive2D Pipeline Synchronization Plan

## Current Issues Identified
1. **No synchronization between pipeline stages** - All `run()` methods are called sequentially without any Vulkan synchronization
2. **No connection between decoder output and rendering pipelines** - Decoded frames aren't passed to compute shaders
3. **Missing command buffer management** - No command buffers are allocated/recorded for compute dispatches
4. **No descriptor set updates** - Video images aren't bound to descriptor sets for compute shaders
5. **No proper frame pacing** - Simple sleep instead of vsync-aware timing
6. **Missing window-specific pipeline execution** - Each window needs its own compute dispatch

## Proposed Pipeline Architecture (Revised)

### 1. Frame Synchronization Strategy - Simplified
```
Frame N-1: [Decode] → [All Compute Stages] → [Present]
Frame N:   [Decode] → [All Compute Stages] → [Present]
```
- **Key Insight**: All compute stages run sequentially in a single command buffer
- **No inter-stage semaphores needed** - Use pipeline barriers instead
- **Minimal semaphore count**: Only decode→compute and compute→present semaphores
- Use fences for CPU-GPU synchronization
- Implement triple-buffering for concurrent frame processing

### 2. Pipeline Stage Dependencies
```
Decoder → (NV12 frames) → VideoImageSet
VideoImageSet → [Sequential Compute Pipeline]:
    1. NV12toBGR (input window)
    2. Crop (region window) 
    3. ColorGrading (grading window)
    4. Composite overlays (all windows)
Compute Results → Present (each window)
```

### 3. Vulkan Synchronization Objects (Simplified)
- **Per-frame objects**: 
  - Command buffer (contains all compute stages)
  - Fence (CPU-GPU sync)
  - `decodeReadySemaphore` (timeline, only if decode uses separate queue)
  - `computeCompleteSemaphore` (binary, for present)
- **Pipeline barriers** (replaces inter-stage semaphores):
  - Memory availability between compute stages
  - Image layout transitions
  - Execution dependencies

### 4. Command Buffer Organization (Optimized)
```
Single Command Buffer Per Frame:
1. Barrier: Wait for decoder image READY (if separate queue)
2. Barrier: VideoImageSet (UNDEFINED → SHADER_READ_ONLY)
3. Dispatch: NV12toBGR (input window)
4. Barrier: Ensure BGR output available for Crop
5. Dispatch: Crop compute
6. Barrier: Ensure crop output available for ColorGrading
7. Dispatch: ColorGrading compute
8. Barrier: Ensure grading output available
9. Barrier: Swapchain images (UNDEFINED → STORAGE)
10. Dispatch: Composite overlays (write to swapchain)
11. Barrier: Swapchain images (STORAGE → PRESENT_SRC)
```
**Benefits**: 
- No semaphores between compute stages
- GPU can optimize entire sequence
- Clear memory dependency chain
- Easier debugging (single command buffer)

## Implementation Steps (Simplified)

### Step 1: Create minimal synchronization objects
- Create fences for each frame in flight (CPU-GPU sync)
- Create `decodeReadySemaphore` (timeline, only if decoder uses separate queue)
- Create `computeCompleteSemaphore` (binary, for present synchronization)
- **Skip inter-compute semaphores** - Use pipeline barriers instead

### Step 2: Modify Decoder for optional timeline semaphore
- Extend `Decoder` to optionally signal a timeline semaphore when frame is ready
- Store decoded frames in `VideoImageSet` with proper image layouts
- If decoder uses same queue as compute, skip semaphore and use pipeline barrier

### Step 3: Create single command buffer per frame
- Allocate command buffers for each frame (triple buffering)
- Record **all compute stages sequentially** in `recordComputeCommands()`:
  - NV12toBGR dispatch
  - Crop dispatch  
  - ColorGrading dispatch
  - Overlay composite dispatch
- Add **pipeline barriers** between stages for memory dependencies
- Add image layout transitions where needed

### Step 4: Connect pipelines to windows via descriptor sets
- Each window gets descriptor set with bound output image (swapchain storage image)
- Compute shaders write directly to window-specific swapchain images
- Update descriptor sets once per frame with new video frame

### Step 5: Implement simplified frame loop
```cpp
void Motive2D::run() {
    while (!windows.empty()) {
        FrameResources& frame = frames[currentFrame];
        
        // 1. Wait for previous frame's GPU work
        vkWaitForFences(engine->logicalDevice, 1, &frame.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(engine->logicalDevice, 1, &frame.fence);
        
        // 2. Acquire decoded frame (may wait on decode semaphore)
        DecodedFrame decoded;
        if (decoder->acquireDecodedFrame(decoded, frame.decodeReadySemaphore, 
                                         frame.decodeSemaphoreValue)) {
            // 3. Update descriptor sets with new frame
            updateDescriptorSets(currentFrame, decoded);
            
            // 4. Record command buffer with all compute stages
            recordComputeCommands(frame.commandBuffer, currentFrame);
            
            // 5. Submit to compute queue
            VkSubmitInfo submitInfo = {};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &frame.commandBuffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &frame.computeCompleteSemaphore;
            
            vkQueueSubmit(engine->graphicsQueue, 1, &submitInfo, frame.fence);
            
            // 6. Present each window
            for (auto& window : windows) {
                window->present(frame.computeCompleteSemaphore);
            }
        }
        
        // 7. Poll events and manage frame pacing
        glfwPollEvents();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}
```

### Step 6: Handle multiple windows efficiently
- Each window has its own swapchain but shares the same compute results
- Compute shaders write to each window's swapchain image in separate dispatches
- Or: Write to intermediate storage image, then copy/blit to each swapchain
- Use `VK_SHARING_MODE_CONCURRENT` if images are used across multiple queues

## Key Code Changes Needed (Simplified)

### 1. Add to `Motive2D` class (simplified):
```cpp
struct FrameResources {
    VkCommandBuffer commandBuffer;      // Single buffer for all compute
    VkFence fence;                      // CPU-GPU sync
    VkSemaphore decodeReadySemaphore;   // Timeline (optional)
    VkSemaphore computeCompleteSemaphore; // Binary (for present)
    uint64_t decodeSemaphoreValue = 0;
};
std::vector<FrameResources> frames;
int currentFrame = 0;
```

### 2. Extend `Decoder` class (optional semaphore):
```cpp
bool acquireDecodedFrame(DecodedFrame& frame, 
                         VkSemaphore waitSemaphore = VK_NULL_HANDLE,
                         uint64_t waitValue = 0);
// If decoder uses separate queue:
void signalDecodeComplete(VkSemaphore signalSemaphore, uint64_t signalValue);
```

### 3. Create `recordComputeCommands()` function:
- Records **all** compute dispatches in sequence
- Adds **pipeline barriers** (not semaphores) between stages
- Binds descriptor sets for each stage
- Returns single command buffer ready for submission

### 4. Implement `updateDescriptorSets()`:
- Updates descriptor sets with new video frame (luma/chroma images)
- Binds window-specific output images (swapchain storage images)
- One-time update per frame (before command buffer recording)

## Performance Considerations (Updated)
- **Use pipeline barriers instead of semaphores** for inter-compute synchronization
- **Batch all compute in single command buffer** for optimal GPU scheduling
- **Fine-grained pipeline barriers** with exact stage masks and access flags
- **Consider shared memory** between compute stages to avoid unnecessary copies
- **Profile with `Nsight Graphics`** to identify bottlenecks

## Testing Strategy
1. Start with single window, single compute stage (NV12toBGR)
2. Add pipeline barriers and verify memory dependencies
3. Add additional compute stages sequentially
4. Enable validation layers to catch synchronization errors
5. Use debug markers (`VK_EXT_debug_utils`) to label pipeline stages
6. Test with multiple windows once single window works

## Testing Strategy
1. Start with single window pipeline
2. Add debug markers and validation layer checks
3. Verify image layouts with `VK_LAYER_KHRONOS_validation`
4. Use `renderDoc` or `Nsight Graphics` for GPU capture

## Performance Considerations
- Use Vulkan 1.2 timeline semaphores for cleaner synchronization
- Batch compute dispatches in single command buffer
- Use pipeline barriers with fine-grained stage masks
- Consider async compute queue if available

## Next Actions
1. Implement basic synchronization objects in `Motive2D` constructor
2. Modify `Decoder` to support timeline semaphores
3. Create compute command buffer recording function
4. Update `Motive2D::run()` to use proper synchronization
5. Test with single window, then expand to multiple windows
