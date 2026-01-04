# Motive Engine2D - C++ File Hierarchy

This document explains the hierarchy and relationships between the core C++ files in the Motive rendering engine.

## Overall Architecture

The Motive engine follows a hierarchical structure where the **Engine2D** class serves as the central coordinator, managing the **Display** system which in turn handles rendering and contains **Cameras** and **Models** composed of **Meshes** and **Primitives**.

## Core Class Hierarchy

```
Engine2D (engine.h/engine.cpp)
├── Display (display.h/display.cpp)
│   ├── Camera (camera.h/camera.cpp)
│   └── Models (via Engine2D's models collection)
│       ├── Model (model.h/model.cpp)
│       │   ├── Mesh (model.h/model.cpp)
│       │   │   └── Primitive (model.h/model.cpp)
│       │   └── Texture (texture.h/texture.cpp)
│       └── Material (texture.h/texture.cpp)
```

## Detailed Class Relationships

### Engine2D Class
**Files:** `engine.h`, `engine.cpp`

**Role:** Central coordinator and Vulkan context manager

**Responsibilities:**
- Vulkan instance, device, and queue management
- Memory allocation and buffer management
- Descriptor pool and layout management
- Command pool management
- Model lifecycle management
- Display system coordination

**Key Dependencies:**
- Contains `Display* display` pointer
- Manages `std::vector<Model> models`
- Provides Vulkan device context to all subsystems

### Display Class
**Files:** `display.h`, `display.cpp`

**Role:** Window management, rendering pipeline, and camera coordination

**Responsibilities:**
- GLFW window creation and management
- Vulkan swapchain and surface management
- Graphics pipeline creation
- Camera instance management and coordination
- Input event forwarding to cameras
- Frame rendering and synchronization
- Multi-camera viewport management

**Key Dependencies:**
- Contains pointer to `Engine2D* engine`
- Manages `std::vector<Camera*> cameras`
- Forwards input events to active cameras
- References models through Engine2D's model collection

### Camera Class [NEW: Separate class]
**Files:** `camera.h`, `camera.cpp`

**Role:** Camera state management, transformations, and input handling

**Responsibilities:**
- Camera position and rotation state management
- View and projection matrix calculations
- Camera UBO creation and management
- Input handling for camera movement (WASD + mouse)
- Camera descriptor set allocation
- Viewport and projection properties

**Key Dependencies:**
- References `Engine2D* engine` for Vulkan operations
- Contains `CameraTransform` UBO for view/projection matrices
- Manages `descriptorSet` for rendering
- Handles input events forwarded from Display

**Key Components:**
- `cameraPos`, `cameraRotation` state variables
- `cameraTransformUBO` for uniform buffer
- `updateCameraMatrices()` method for matrix updates
- Input handling methods for mouse and keyboard

### Model Class
**Files:** `model.h`, `model.cpp`

**Role:** 3D model container and manager

**Responsibilities:**
- Model data loading (GLTF or manual vertices)
- Mesh collection management
- Resource cleanup

**Key Dependencies:**
- Contains `std::vector<Mesh> meshes`
- Contains `std::vector<Texture*> textures`
- References `Engine2D* engine` for Vulkan operations

### Mesh Class
**Files:** `model.h`, `model.cpp`

**Role:** Mesh container for primitives

**Responsibilities:**
- Primitive collection management
- Mesh-level transformations

**Key Dependencies:**
- Contains `std::vector<Primitive> primitives`
- References parent `Model* model`

### Primitive Class
**Files:** `model.h`, `model.cpp`

**Role:** Individual renderable geometry unit

**Responsibilities:**
- Vertex buffer management
- Texture resource management
- Uniform buffer management for object transforms
- Descriptor set management

**Key Dependencies:**
- Manages `vertexBuffer`, `vertexBufferMemory`
- Contains `ObjectTransformUBO` for per-object transforms
- Contains `primitiveDescriptorSet` for rendering
- References `Engine2D* engine` for Vulkan operations

### Texture Class
**Files:** `texture.h`, `texture.cpp`

**Role:** Texture resource management

**Responsibilities:**
- Texture image creation and management
- Sampler creation
- Image view management
- Texture descriptor set updates

**Key Dependencies:**
- Manages `textureImage`, `textureImageView`, `textureSampler`
- References `Mesh* mesh` for association

### Material Class
**Files:** `texture.h`, `texture.cpp`

**Role:** Material properties container

**Responsibilities:**
- Texture collection management
- Material property definitions

**Key Dependencies:**
- Contains `std::vector<Texture*> textures`

## Data Flow

1. **Initialization:**
   - `Engine2D` creates Vulkan context
   - `Engine2D` creates `Display` system
   - `Display` creates `Camera` instances
   - `Display` creates window and graphics pipeline
   - Models are added to `Engine2D` via `addModel()`
   - Cameras allocate descriptor sets after pipeline creation

2. **Input Handling:**
   - GLFW input events captured by `Display`
   - Events forwarded to active `Camera` instances
   - `Camera` handles mouse movement and keyboard input
   - Camera state updated based on input

3. **Rendering Loop:**
   - `Display::render()` called each frame
   - All cameras updated via `camera->update()`
   - For each camera: set viewport/scissor and bind descriptor set
   - For each model: meshes → primitives rendered
   - Each primitive binds its descriptor sets and draws

4. **Multi-Camera Support:**
   - Multiple cameras can be active simultaneously
   - Each camera has its own viewport and projection
   - Scene rendered once per camera with different transforms

## Resource Management

- **Engine2D:** Owns Vulkan device, descriptor pool, command pool
- **Display:** Owns swapchain, pipeline, window resources
- **Camera:** Owns camera UBO, descriptor set, camera state
- **Model:** Owns meshes and textures
- **Primitive:** Owns vertex buffers, object UBOs, descriptor sets
- **Texture:** Owns image resources and samplers

## Key Design Patterns

- **Composition over Inheritance:** Complex objects built from simpler components
- **Resource Ownership:** Each class manages its own Vulkan resources
- **Dependency Injection:** Classes receive `Engine2D*` for Vulkan operations
- **Separation of Concerns:** Clear division between rendering, camera management, model management, and resource management
- **Event Forwarding:** Display forwards input events to Camera instances

## Pipeline Architecture

For a comprehensive diagram of all pipelines including decoder, render plane, and their connections, see `pipeline_architecture.md`. Key pipelines include:

1. **Video Decoding Pipeline**: FFmpeg/Vulkan Video decoder → VideoImageSet
2. **Video Blit Pipeline**: YUV → RGB conversion to Grading Images
3. **Overlay Pipelines**: Pose, rectangle, subtitle, and FPS overlays applied to Grading Images
4. **Grading Pipeline**: Color adjustments (exposure, contrast, saturation, 3-way color, curve LUT)
5. **Scrubber Pipeline**: Final blit to swapchain with aspect ratio handling
6. **Presentation Pipeline**: Swapchain to GLFW window

Each pipeline has specific source/destination mappings and operates on intermediate buffers (Grading Images) for isolation between processing stages.

## Headless Vulkan Video (In Progress)
- Headless Annex-B path built in `encode.cpp` using `mini_decoder*` helpers (no FFmpeg); loads Vulkan Video entry points from `Engine2D` and queries decode formats with `vkGetPhysicalDeviceVideoFormatPropertiesKHR`.
- `mini_decoder_session` creates `VkVideoSession/VkVideoSessionParameters` and allocates DPB images/views; `mini_decode_pipeline` uploads Annex-B NALs into a bitstream buffer and records `vkCmdDecodeVideoKHR`, transitioning DPB images to `GENERAL`.
- `OffscreenBlit` in `encode.cpp` copies decoded DPB images into an RGBA target for downstream blit/encode, staying GPU-only.
- Remaining integration: wire Vulkan-Video-Samples parser to emit `VkParserPerFrameDecodeParameters/VkParserDecodePictureInfo`, honor DPB/POC/display order, initialize `VulkanVideoFrameBuffer` from stream sequence info, and feed parsed decode images into the existing blit/encode pipeline.

## Vulkan Video Integration (WIP)

- A Vulkan-only encode path is being built to replace FFmpeg. New helper files: `annexb_demuxer.{h,cpp}` (raw Annex-B input), `vulkan_video_bridge.{h,cpp}` (adapts Engine2D-owned Vulkan handles to Vulkan Video), and imported sample sources under `vk_video_decoder/` and `common_vv/`.
- Current expectation: inputs are raw Annex-B elementary streams (`.h264`/`.h265`). Container demuxing (MP4/MKV) is not provided.
- Parser/decoder are instantiated; NALs are fed via `ParseByteStream`, and the display callback dequeues from `VulkanVideoFrameBuffer` to surface decoded images/headless.
- Remaining work: ensure frame buffer init matches stream format, wire decoded images into blit/NV12/encode, replace placeholder timestamps/DPB handling, and implement Vulkan Video encode + MP4 mux.
- Alternative path (planned): drop the heavy sample stack and build a minimal Vulkan Video decoder on top of Engine2D-owned instance/device/queues:
  - Query video decode capabilities/profile from the input bitstream (H.264/H.265).
  - Create `VkVideoSessionKHR`/session parameters and allocate decode/DPB images and bitstream buffers.
  - Record/submit `vkCmdDecodeVideoKHR` per frame and surface the output `VkImage`/layout to the blit/encode path.
  - Add lightweight Annex-B feeding and cleanup.

## Memory Management

- **Engine2D:** Owns Vulkan device, descriptor pool, command pool
- **Display:** Owns swapchain, pipeline, window resources
- **Camera:** Owns camera UBO, descriptor set
- **Model:** Owns meshes and textures
- **Primitive:** Owns vertex buffers, object UBOs, descriptor sets
## Next Todo: Fix Black Screen Video Rendering

**Issue:** Video playback shows black screen despite correct YUV data being decoded and uploaded.

**Root Cause Identified:** Format mismatch between decoder output (`yuv422p10le` planar 4:2:2) and requested software format (`p010le` packed 4:2:0) has been fixed.

**Remaining Issues to Investigate:**
1. **Vulkan rendering pipeline**: Verify compute shader execution, texture binding, and descriptor sets
2. **Texture format compatibility**: Ensure 16-bit UNORM textures (R16_UNORM, R16G16_UNORM) are correctly sampled by shader
3. **Swapchain presentation**: Confirm frames are being presented to the window
4. **Shader debugging**: Add more instrumentation to verify shader execution path

**Immediate Actions:**
- Enable Vulkan validation layers for error detection
- Add logging to `display2d.cpp` renderFrame method to track pipeline execution
- Verify texture creation matches shader expectations in `video_frame_utils.cpp`
- Test with 8-bit video to isolate 10-bit YUV conversion issues

This hierarchical structure allows for efficient resource management and clear separation of responsibilities while maintaining flexibility for future extensions. The separation of Camera into its own class enables better organization and potential multi-camera scenarios.
