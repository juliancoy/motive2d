# Project Architecture for LLMs

This project is a compute-driven video processing playground, where **`motive2d`** drives the high-level workflow and **`engine2d`** owns the Vulkan instance/device plus the compute pipelines. The goal is to keep each pipeline conceptually independent so LLMs can reason about contributions separately.

## 1. Core Roles

### 1.1. Main Orchestrator (`motive2d` / `motive2d.cpp`)
- Starts GLFW windows, parses command line/config, and spawns `Display2D` instances per viewport.
- Drives decoding jobs, schedules pipeline passes, and routes frame data between overlays, grading, and presentation.

### 1.2. Vulkan Engine (`engine2d` / `engine2d.cpp` + `engine2d.py`)
- Owns `VkInstance`, `VkDevice`, queue families, swapchains, descriptor pools, and synchronization primitives.
- Provides helpers for creating compute pipelines, image views, descriptor sets, and command buffers consumed by every pipeline.
- Manages Vulkan resource lifetime and exposes interfaces that `Display2D` uses for submission.

## 2. Pipeline Catalog

Each pipeline is implemented in a dedicated class or module, sharing only the Vulkan resources handed out by `engine2d`.

Every pipeline already declares or reuses a descriptor set layout so that it can be bound from any `Display2D` instance without rebuilding the bindings:
- Color grading (see `motive2d.cpp:511-611`) creates a six-binding set layout (swapchain storage image, overlay images, luma/chroma samplers, curve UBO), allocates one descriptor set per swapchain image, and reuses the same layout for the grading compute pass and the final scrubber blit.
- Rectangle overlay (`rect_overlay.cpp:208-289`) owns its own layout/pipeline/pool, which means the single `RectOverlay` kept in `Engine2D` can rebinding the layout for every window without re-creation.
- Pose overlay (`pose_overlay.cpp:672-734`) constructs a storage-image + storage-buffer layout whose pipeline layout is cached in the shared `PoseOverlay` instance.
- Subtitle overlay (`subtitle.cpp:96-134`) builds a three-binding layout (storage image plus two combined image samplers) before creating its pipeline, so any window that needs subtitles can rebind the descriptors on demand.
- Scrubber (`scrubber.cpp:8-58`) reuses the pipeline layout built alongside the color grading descriptor layout, so the same bindings and push constants are valid across every display when the final blit runs.

### 2.1. Video Decoding + Upload
- FFmpeg + `decoder.cpp` (with `annexb_demuxer` for raw streams) feed H.264/H.265 frames.
- Produces a `VideoImageSet` (luma/chroma Vulkan textures) that can be uploaded on the CPU or forwarded via Vulkan Video zero-copy.

### 2.2. Color Grading (`color_grading_pass.*`, `color_adjustments.h`)
- Converts YUV to RGB, applies exposure/contrast/saturation, three-way color correction, and LUT curves.
- Runs as a compute pass over intermediate `GradingImage` buffers before overlays or final output.

### 2.3. Color Grading UI (`color_grading_ui.*`)
- Hosts a separate control window with sliders for grading parameters.
- Communicates adjustments to the grading pass through shared state or callbacks so shaders update uniforms in real time.

### 2.4. Rectangle Overlay + Region Display (`rect_overlay.*`)
- Renders crop rectangles via `shaders/overlay_rect.comp` compute shaders on the grading buffer.
- Supports a dedicated Region Display zoom window for precise rectangle placement.

### 2.5. Pose Overlay (`pose_overlay.*`, `models/`)
- Reads YOLO pose JSON output and renders skeletons/keypoints via `shaders/overlay_pose.comp`.
- Stagers its draw calls independently while consuming the shared `GradingImage`.

### 2.6. Subtitle Overlay (`subtitle.*`)
- Loads WhisperX or similar JSON transcripts and rasterizes text onto the grading image through compute shaders.
- Integrates tightly with the same buffer chain so the text is composited after the grading pass.

### 2.7. Scrubber (`scrubber.*`)
- Draws a timeline scrubber UI beneath the playback surface.
- Blits the final graded image plus scrubber widgets to the swapchain through `shaders/scrubber_blit.comp`.

Overlays are applied before the scrubber in the `motive2d` render loop: pose, rectangle, and subtitle passes mutate `playbackState.overlay.image` in sequence (`motive2d.cpp:889-1044`), and only after that does `Display2D::renderFrame` update the shared descriptor set and run the grading/scrubber compute work (`motive2d.cpp:1240-1478`). The descriptor layout that feeds the scrubber is built alongside the grading layout (`motive2d.cpp:511-616`) and matches the bindings used by `scrubber.cpp:8-58`, so the final compute blit always sees the fully composited overlay image. Memory barriers between the grading buffer and swapchain (`motive2d.cpp:1361-1478`) enforce the transfer → compute → present order so the scrubber/ final blit cannot see stale data.

### 2.8. Supplementary Tools
- `fps.*`, `utils.*`, `display2d.*`, and asset helpers support diagnostics, overlays, and resource conversions.
- `video_editor_orchestrator.cpp` and scripts orchestrate multi-window sessions.

## 3. Data Flow Snapshot

```
Video File
└─→ Decoder → VideoImageSet (YUV textures)
    └─→ Color Grading → GradingImage (RGBA)
        ├─→ Pose Overlay
        ├─→ Rectangle Overlay
        ├─→ Subtitle Overlay
        └─→ Scrubber + Final Blit → Swapchain → GLFW Window
```

Non-blocking overlays mutate the grading buffer in a controlled sequence to avoid races.

## 4. Multi-Window Topology

### 4.1. Primary Window
- Displays video playback with overlays and scrubber controls.

### 4.2. Grading Control Window
- Ran by `color_grading_ui`, surfaces sliders/levers for real-time adjustments.

### 4.3. Region Display
- Zooms into the rectangle overlay region for precise cropping feedback.

### 4.4. Optional Preview Panels
- Pose or subtitle previews can be instantiated when needed since overlays are modular.

Each window has its own `Display2D` instance and swapchain but reuses shader modules/pipeline layouts where possible.
Each `Display2D` is responsible for its swapchain while `motive2d` owns the set of displays it creates.
Swapchains are owned 1:1 by `Display2D`: each instance keeps its own `VkSurfaceKHR`, `VkSwapchainKHR`, image views, descriptor sets, and sync objects (`display2d.h:65-100`), creates them inside the constructor/`createSwapchain`, and tears them down in the destructor (`display2d.cpp:90-140`). `Engine2D::createWindow` simply constructs a `Display2D`, stores it in the `windows` vector (`engine2d.cpp:65-86`), and returns the raw pointer, so the swapchain ownership stays with that display until `Engine2D::shutdown` destroys the window list (`engine2d.cpp:256-264`).

`motive2d` registers its displays at start-up by invoking `engine.createWindow` for every requested panel (input, region, grading) and saves those `Display2D*` handles locally (`motive2d.cpp:336-385`). It configures each display (scrubber pass, video/passthrough toggles, scroll callbacks) before the render loop so later updates towards `Display2D` ownership can refer back to these handles.

## 5. Configuration Notes for LLMs

- Requires Vulkan-capable GPU, GLFW, and FFmpeg.
- JSON artifacts (YOLO pose, Whisper subtitles) have simple arrays of structs (keypoints/text segments).
- Treat `motive2d` as the entry point; register new overlays as independent compute passes.
- When extending, register shaders/descriptors in `engine2d` and plug them into `motive2d` so per-frame descriptor updates stay coordinated.
