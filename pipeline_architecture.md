# Motive2D Pipeline Architecture

Based on analysis of the codebase, here is the comprehensive pipeline architecture showing the combination of pipelines, including source and destination of each, decoder, and render plane.

## Overall Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VIDEO DECODING PIPELINE                           │
│                                                                             │
│  Input File (MKV/H.264/H.265)                                               │
│        │                                                                    │
│        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FFmpeg Decoder (Software/Hardware)                                 │    │
│  │  - AVFormatContext → AVCodecContext → AVFrame                      │    │
│  │  - Hardware: Vulkan Video (VK_KHR_video_decode_queue)              │    │
│  │  - Software: CPU YUV conversion                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  Decoded Frame (YUV format: NV12/Planar420/Planar422/Planar444)            │
│        │                                                                    │
│        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Frame Upload (CPU or GPU)                                          │    │
│  │  - CPU: uploadImageData() to Vulkan textures                       │    │
│  │  - GPU: Vulkan Video zero-copy (external image views)              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  VideoImageSet (luma + chroma textures)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RENDERING PIPELINES (Parallel)                      │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ VIDEO BLIT PIPELINE (grading_pass)                        │    │
│  │                                                                    │    │
│  │ Source: VideoImageSet (YUV textures)                               │    │
│  │ Destination: Grading Images (RGBA intermediate)                    │    │
│  │ Shader: shaders/video_blit.comp                                    │    │
│  │ Function: YUV → RGB conversion with color space adjustment         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ OVERLAY PIPELINES (Parallel compute passes)                        │    │
│  │                                                                    │    │
│  │ 1. Pose Overlay (pose_overlay_compute)                             │    │
│  │    Source: Grading Images + YOLO pose detection data               │    │
│  │    Destination: Grading Images (modified in-place)                 │    │
│  │    Shader: shaders/overlay_pose.comp                               │    │
│  │                                                                    │    │
│  │ 2. Rectangle Overlay (rect_overlay)                                │    │
│  │    Source: Grading Images + rectangle coordinates                  │    │
│  │    Destination: Grading Images (modified in-place)                 │    │
│  │    Shader: shaders/overlay_rect.comp                               │    │
│  │                                                                    │    │
│  │ 3. Subtitle Overlay (subtitle_overlay)                             │    │
│  │    Source: Grading Images + subtitle text                          │    │
│  │    Destination: Grading Images (modified in-place)                 │    │
│  │    Shader: compute shader for text rendering                       │    │
│  │                                                                    │    │
│  │ 4. FPS Overlay (overlay.cpp)                                       │    │
│  │    Source: Generated FPS text bitmap                               │    │
│  │    Destination: Grading Images (blended)                           │    │
│  │    Function: CPU-generated overlay uploaded as texture             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ GRADING PIPELINE (Color Adjustments)                               │    │
│  │                                                                    │    │
│  │ Source: Grading Images (with overlays applied)                     │    │
│  │ Destination: Same Grading Images (modified in-place)               │    │
│  │ Operations:                                                        │    │
│  │  - Exposure adjustment                                             │    │
│  │  - Contrast adjustment                                             │    │
│  │  - Saturation adjustment                                           │    │
│  │  - Shadows/Midtones/Highlights (3-way color correction)            │
│  │  - Curve LUT (256-entry lookup table)                              │    │
│  │  - Blank screen (optional)                                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ SCRUBBER PIPELINE (scrubber_pipeline)                              │    │
│  │                                                                    │    │
│  │ Source: Grading Images (final graded image)                        │    │
│  │ Destination: Swapchain Images (final output)                       │    │
│  │ Shader: shaders/scrubber_blit.comp                                 │    │
│  │ Function: Final blit to swapchain with aspect ratio handling       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  Swapchain Images (VkImage)                                                │
│        │                                                                    │
│        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ PRESENTATION                                                       │    │
│  │                                                                    │    │
│  │ Queue: vkQueuePresentKHR                                           │    │
│  │ Destination: GLFW Window Surface                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Pipeline Connections

### 1. Video Decoding Pipeline
- **Source**: Video file (H.264/H.265 in MKV/MP4 container or raw Annex-B)
- **Decoder**: FFmpeg AVCodec with hardware acceleration (Vulkan Video) fallback to software
- **Output Formats**: 
  - NV12 (8-bit 4:2:0 interleaved)
  - Planar YUV (420/422/444, 8/10/12/16-bit)
- **Destination**: `VideoImageSet` containing:
  - `luma`: VkImageView for Y plane (R8_UNORM or R16_UNORM)
  - `chroma`: VkImageView for UV plane (R8G8_UNORM or R16G16_UNORM)

### 2. Video Blit Pipeline (YUV → RGB)
- **Source**: `VideoImageSet` (YUV textures)
- **Shader**: `shaders/video_blit.comp` (SPIR-V)
- **Destination**: Grading Images (RGBA8_UNORM intermediate buffers)
- **Function**: Performs YUV to RGB conversion with:
  - Color space conversion (BT.601/BT.709/BT.2020)
  - Color range adjustment (limited/full)
  - Chroma subsampling handling

### 3. Overlay Pipelines (Parallel)
All overlays operate on the Grading Images (RGBA intermediate):

#### 3.1 Pose Overlay
- **Source**: Grading Images + YOLO pose detection data
- **Shader**: `shaders/overlay_pose.comp`
- **Data**: 187 keypoints (from YOLO11n-pose model)
- **Function**: Draws skeleton lines and keypoints

#### 3.2 Rectangle Overlay
- **Source**: Grading Images + rectangle coordinates
- **Shader**: `shaders/overlay_rect.comp`
- **Data**: Rectangle center (987, 234), size (171.695×305.235)
- **Function**: Draws bounding boxes

#### 3.3 Subtitle Overlay
- **Source**: Grading Images + subtitle text
- **Shader**: Custom compute shader
- **Data**: Text dimensions (142×15), position (901,81)
- **Function**: Renders anti-aliased text

#### 3.4 FPS Overlay
- **Source**: CPU-generated bitmap from `buildFrameRateOverlay()`
- **Upload**: `uploadImageData()` to Vulkan texture
- **Function**: Displays frame rate counter

### 4. Grading Pipeline (Color Adjustments)
- **Source**: Grading Images (with all overlays applied)
- **Operations** (applied in sequence):
  1. Exposure adjustment (linear multiplier)
  2. Contrast adjustment (S-curve)
  3. Saturation adjustment (color intensity)
  4. 3-way color correction (shadows/midtones/highlights)
  5. Curve LUT application (256-entry lookup table)
  6. Blank screen (optional blackout)
- **Destination**: Same Grading Images (modified in-place)

### 5. Scrubber Pipeline (Final Output)
- **Source**: Grading Images (final graded RGBA)
- **Shader**: `shaders/scrubber_blit.comp`
- **Destination**: Swapchain Images (VkImage)
- **Function**:
  - Handles aspect ratio correction
  - Applies target region overrides
  - Performs final blit with bilinear filtering

### 6. Presentation Pipeline
- **Source**: Swapchain Images
- **Queue**: Graphics queue submission
- **Synchronization**: 
  - `imageAvailableSemaphores`
  - `renderFinishedSemaphores`
  - `inFlightFences`
- **Destination**: GLFW window surface

## Resource Management

### Image Resources
1. **Video Textures**: `VideoImageSet` (luma + chroma)
   - Owned by `VideoResources` in `decode.cpp`
   - Either Vulkan Video external images or CPU-uploaded textures

2. **Grading Images**: Intermediate RGBA buffers
   - Array of `VkImage` objects in `Display2D`
   - Used for all compute passes (blit, overlays, grading)

3. **Swapchain Images**: Final output
   - Managed by Vulkan swapchain
   - Presented to window surface

### Pipeline Objects
1. **Compute Pipelines**:
   - `gradingBlitPipeline`: YUV → RGB conversion
   - `overlayPipeline`: Overlay composition
   - `scrubPipeline`: Final blit to swapchain

2. **Descriptor Sets**:
   - Bound per-frame for texture access
   - Separate sets for video textures, overlays, and grading LUT

## Data Flow Summary

```
Video File → Decoder → VideoImageSet → Video Blit → Grading Images
                                                         │
                                                         ├─→ Pose Overlay
                                                         ├─→ Rectangle Overlay
                                                         ├─→ Subtitle Overlay
                                                         ├─→ FPS Overlay
                                                         │
                                                         ↓
Grading Pipeline → Scrubber Pipeline → Swapchain → Presentation → Window
```

## Multi-Window Architecture

The system supports multiple `Display2D` instances:
1. **Main Window** (1280×720): Full video + overlays
2. **Grading Window** (420×880): UI controls only (no video)
3. **Region View** (360×640): Zoomed/cropped region

Each window has its own:
- `Display2D` instance
- Swapchain and grading images
- Pipeline objects (shared shaders, unique pipelines)

## Vulkan Video Integration (WIP)

The system is transitioning from FFmpeg to pure Vulkan Video:
- **Current**: FFmpeg with Vulkan hardware acceleration fallback
- **Target**: Pure Vulkan Video decode/encode
- **Components**:
  - `annexb_demuxer`: Raw Annex-B input parsing
  - `vulkan_video_bridge`: Engine2D to Vulkan Video adapter
  - `mini_decoder`: Lightweight Vulkan Video decoder

## Key Design Patterns

1. **Compute-Only Rendering**: All image processing via compute shaders
2. **Intermediate Buffering**: Grading images isolate processing stages
3. **Descriptor Reuse**: Shared descriptor sets across pipelines
4. **Zero-Copy When Possible**: Vulkan Video direct GPU-to-GPU transfer
5. **Parallel Overlays**: Independent compute passes for each overlay type

This architecture enables real-time video processing with multiple overlays and professional color grading capabilities.
