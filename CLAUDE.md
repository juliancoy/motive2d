# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Motive2D** is a Vulkan-based compute-driven video processing engine for real-time color grading, overlay composition, and multi-window video editing. The project processes video through a pipeline of compute shaders, supporting features like pose detection overlays, color grading, cropping, and subtitle rendering.

## Build System

### Building the Project

```bash
# Build dependencies (first time only)
python build_deps.py

# Build project (incremental)
python build.py

# Force rebuild all
python build.py --rebuild

# Build with AddressSanitizer
python build.py --asan
```

The build system (`build.py`) handles:
- Compiling all `.cpp` files to `.o` objects
- Compiling GLSL shaders (`.comp`, `.vert`, `.frag`) to SPIR-V (`.spv`)
- Linking `libengine.a` static library
- Building executables: `motive2d` and `decode_benchmark`
- Incremental builds based on file timestamps (stored in `.build_manifest.json`)

### Running the Application

```bash
# Run with default video
./motive2d

# Run with specific video
./motive2d --video path/to/video.mp4

# Enable debug logging
./motive2d --debug

# Show only input window
./motive2d --input-only

# Run with all windows
./motive2d --show-input --show-region --show-grading
```

### Shader Compilation

Shaders are automatically compiled by `build.py` using `glslangValidator`. Shaders are located in `shaders/` and compiled to `.spv` files:
- `nv12toBGR.comp` - NV12 to BGR conversion
- `color_grading_pass.comp` - Color grading compute pass
- `overlay_pose.comp` - Pose skeleton overlay
- `overlay_rect.comp` - Rectangle crop overlay
- `scrubber.comp` - Timeline scrubber UI
- `composite_bitmap.comp` - Bitmap compositing

## Architecture

### Core Components

#### `motive2d` (Main Orchestrator)
- Entry point and main application loop
- Manages multiple `Display2D` windows (input, region, grading)
- Coordinates video decoding, compute pipeline execution, and presentation
- Implements triple-buffered frame synchronization with Vulkan semaphores/fences
- Located in: `motive2d.h`, `motive2d.cpp`

#### `engine2d` (Vulkan Engine)
- Owns Vulkan instance, device, queues, and resource pools
- Provides helper functions for buffer/image creation, command buffers, descriptor sets
- Manages core Vulkan lifecycle and resource allocation
- Located in: `engine2d.h`, `engine2d.cpp`

#### `Display2D` (Window/Swapchain Manager)
- Each window has its own `Display2D` instance
- Manages per-window swapchain, image views, descriptor sets, and sync objects
- Handles window events and rendering to swapchain
- 1:1 ownership of swapchain resources
- Located in: `display2d.h`, `display2d.cpp`

#### `Decoder` (Video Decoding)
- FFmpeg-based video decoder with software and Vulkan Video support
- Produces `VideoImageSet` (luma/chroma Vulkan textures)
- Supports async decoding with frame buffering
- Located in: `decoder.h`, `decoder.cpp`

### Pipeline Architecture

The compute pipeline follows this data flow:

```
Video File
└─→ Decoder → VideoImageSet (NV12: Y/UV planes)
    └─→ nv12toBGR → BGR Image
        └─→ Crop → Cropped Region
            └─→ ColorGrading → Graded RGBA
                ├─→ PoseOverlay → Skeleton/keypoints
                ├─→ RectOverlay → Crop rectangles
                ├─→ Subtitle → Text overlays
                └─→ Scrubber + Blit → Swapchain → Display
```

### Current Pipeline Implementation

As of January 2026, the pipeline is in active development with the following status:

**Implemented:**
- Triple-buffered frame synchronization (`FrameResources` with command buffers, fences, semaphores)
- `nv12toBGR` compute pipeline with descriptor sets and push constants
- Frame loop with GPU synchronization (fence wait/reset, semaphore signaling)
- Decoder async frame acquisition
- Compute command recording infrastructure

**In Progress (see plan.md for details):**
- Descriptor set updates to bind decoder frames to compute shaders
- Pipeline barriers between compute stages
- Window presentation integration with compute semaphores
- ColorGrading and overlay dispatch implementations

**Key Synchronization Strategy:**
- All compute stages execute sequentially in a single command buffer per frame
- Pipeline barriers (not inter-stage semaphores) enforce memory dependencies
- Minimal semaphores: decode→compute (optional), compute→present (required)

### Compute Pipelines

Each compute pipeline is implemented as an independent class:

#### `nv12toBGR`
- Converts NV12 video frames to BGR/RGBA for further processing
- Descriptor bindings: Y plane (0), UV plane (1), BGR output (2)
- Push constants: frame dimensions, color space/range
- Located in: `nv12toBGR.h`, `nv12toBGR.cpp`

#### `ColorGrading`
- Applies color adjustments: exposure, contrast, saturation, three-way color, LUT curves
- Six-binding descriptor layout: swapchain storage, overlay images, luma/chroma samplers, curve UBO
- Located in: `color_grading_pass.h`, `color_grading_pass.cpp`

#### `PoseOverlay`
- Renders YOLO pose detection results (skeletons/keypoints)
- Storage image + storage buffer layout
- Reads JSON pose data from `.json` files
- Located in: `pose_overlay.h`, `pose_overlay.cpp`

#### `RectOverlay`
- Draws crop rectangles on video
- Independent layout/pipeline/pool
- Located in: `rect_overlay.h`, `rect_overlay.cpp`

#### `Subtitle`
- Renders text overlays from WhisperX JSON transcripts
- Three-binding layout: storage image + two texture samplers
- Located in: `subtitle.h`, `subtitle.cpp`

#### `Scrubber`
- Timeline UI widget for seeking/playback control
- Reuses color grading pipeline layout
- Located in: `scrubber.h`, `scrubber.cpp`

### Data Structures

#### `VideoImageSet`
- Holds YUV video frame textures (luma/chroma planes)
- Used by decoder output and compute shader input
- Includes samplers and format metadata

#### `FrameResources`
- Per-frame synchronization objects for triple buffering
- Contains: command buffer, fence, decode/compute semaphores, timeline value
- Defined in `motive2d.h`

#### `CliOptions`
- Command-line configuration for windows, overlays, debugging
- Controls which windows are shown and which features are enabled

## Development Guidelines

### Adding a New Compute Pipeline

1. Create header/source files (e.g., `new_pipeline.h`, `new_pipeline.cpp`)
2. Define descriptor set layout in pipeline constructor
3. Load SPIR-V shader from `shaders/new_pipeline.spv`
4. Create pipeline layout with push constants if needed
5. Integrate into `motive2d.cpp` render loop:
   - Add pipeline member to `Motive2D` class
   - Initialize in constructor after decoder dimensions are known
   - Add dispatch call in `recordComputeCommands()`
   - Add pipeline barrier before/after for memory dependencies
6. Update descriptor sets in `updateDescriptorSets()`

### Vulkan Synchronization Rules

- **Use pipeline barriers** (not semaphores) between compute stages in same command buffer
- **Use semaphores** only for queue-to-queue or compute-to-present synchronization
- **Use fences** for CPU-GPU synchronization (frame in flight tracking)
- Always specify correct `VkAccessFlags` and `VkPipelineStageFlags` for barriers
- Test with validation layers enabled: `VK_LAYER_KHRONOS_validation`

### Debugging

```bash
# Enable debug logging
./motive2d --debug

# Use RenderDoc for GPU capture
# Capture files saved as .rdc in project root

# Check Vulkan validation layers
export VK_LAYER_PATH=/path/to/vulkan/layers
./motive2d --debug
```

Debug logging can be controlled in code:
```cpp
#include "debug_logging.h"
setDebugLoggingEnabled(true);  // Enable general debug logging
setRenderDebugEnabled(true);   // Enable render pipeline logging
```

### Important Code Patterns

**Command Buffer Recording:**
```cpp
// In recordComputeCommands():
// 1. Bind pipeline
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

// 2. Bind descriptor sets
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

// 3. Push constants (if needed)
vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                   0, sizeof(constants), &constants);

// 4. Dispatch compute
vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

// 5. Pipeline barrier for next stage
VkImageMemoryBarrier barrier = { /* ... */ };
vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0,
                     0, nullptr, 0, nullptr, 1, &barrier);
```

**Descriptor Set Updates:**
```cpp
VkDescriptorImageInfo imageInfo = {
    .sampler = sampler,
    .imageView = imageView,
    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
};

VkWriteDescriptorSet write = {
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = descriptorSet,
    .dstBinding = bindingIndex,
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    .pImageInfo = &imageInfo
};

vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
```

### File Organization

**Core Engine:**
- `motive2d.{h,cpp}` - Main application
- `engine2d.{h,cpp}` - Vulkan engine
- `display2d.{h,cpp}` - Window/swapchain management
- `graphicsdevice.{h,cpp}` - Low-level Vulkan device wrapper

**Video Processing:**
- `decoder.{h,cpp}` - FFmpeg video decoder
- `nv12toBGR.{h,cpp}` - YUV to RGB conversion
- `crop.{h,cpp}` - Cropping utilities

**Overlays:**
- `pose_overlay.{h,cpp}` - YOLO pose rendering
- `rect_overlay.{h,cpp}` - Rectangle overlay
- `subtitle.{h,cpp}` - Text subtitles

**UI & Controls:**
- `color_grading_ui.{h,cpp}` - Color grading control window
- `scrubber.{h,cpp}` - Timeline scrubber
- `fps.{h,cpp}` - FPS counter overlay

**Utilities:**
- `utils.{h,cpp}` - Helper functions
- `debug_logging.{h,cpp}` - Debug output control
- `image_resource.{h,cpp}` - Image loading/management

### Dependencies

**Required Libraries:**
- Vulkan SDK (headers in `Vulkan-Headers/`)
- GLFW3 (window management)
- GLM (math library)
- FFmpeg (video decoding)
- FreeType (text rendering)
- NCNN (neural network inference for YOLO)
- nlohmann_json (JSON parsing)

**Build Tools:**
- Python 3
- g++ with C++17 support
- glslangValidator (SPIR-V shader compilation)

### Common Issues

**Blank screens:**
- Check descriptor sets are properly bound (`updateDescriptorSets()`)
- Verify image layouts are correct for shader access
- Ensure compute dispatches are recorded in command buffer
- Check pipeline barriers have correct stage/access masks

**Synchronization errors:**
- Enable validation layers: `export VK_LOADER_DEBUG=all`
- Check semaphore signaling/waiting pairs match
- Verify fence waiting before command buffer reuse
- Use RenderDoc to inspect command buffer execution

**Build errors:**
- Ensure dependencies are built: `python build_deps.py`
- Check FFmpeg installation in `FFmpeg/.build/install/`
- Verify Vulkan SDK is installed and headers are accessible
- Clean build with `--rebuild` flag

### Testing Strategy

1. Start with single window, single compute stage (`nv12toBGR`)
2. Add pipeline barriers and verify with validation layers
3. Progressively enable additional compute stages
4. Test multi-window scenarios after single window works
5. Profile with Nsight Graphics or RenderDoc for performance

### Key References

- `plan.md` - Detailed pipeline synchronization implementation plan
- `ARCH.md` - LLM-oriented architecture documentation
- `README.md` - Original project overview
