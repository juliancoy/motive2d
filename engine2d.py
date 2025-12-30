#!/usr/bin/env python3
"""
engine2d.py - Optimized Python interface for the Motive 2D render pipeline.

This module provides a high-level Python API for the Vulkan-based 2D video
rendering pipeline, exposing functionality for video playback, grading,
cropping, and real-time interactive controls.
"""

import ctypes
import os
import sys
import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np

# Try to load the Motive engine library
# First check for shared library, then try static library via ctypes
_lib = None
_lib_paths = [
    os.path.join(os.path.dirname(__file__), 'libengine.so'),
    os.path.join(os.path.dirname(__file__), 'libengine.a'),
    os.path.join(os.path.dirname(__file__), 'build/libengine.so'),
]

for path in _lib_paths:
    if os.path.exists(path):
        try:
            _lib = ctypes.CDLL(path)
            print(f"[Engine2D] Loaded library from {path}")
            break
        except Exception as e:
            print(f"[Engine2D] Failed to load {path}: {e}")
            continue

if _lib is None:
    # Try to load from system paths
    try:
        _lib = ctypes.CDLL('libengine.so')
    except:
        print("[Engine2D] WARNING: Could not load engine library. Some functionality will be limited.")
        _lib = None


class VideoCodec(Enum):
    """Video codec enumeration."""
    H264 = 0
    H265 = 1


class PixelFormat(Enum):
    """Pixel format enumeration."""
    NV12 = 0
    YUV420 = 1
    RGBA8 = 2
    BGRA8 = 3


@dataclass
class VideoInfo:
    """Video file information."""
    width: int
    height: int
    framerate: float
    duration: float
    codec: VideoCodec
    pixel_format: PixelFormat


@dataclass
class GradingSettings:
    """Color grading settings."""
    exposure: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    shadows: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    midtones: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    highlights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    curve_lut: List[float] = None
    
    def __post_init__(self):
        if self.curve_lut is None:
            self.curve_lut = [float(i) / 255.0 for i in range(256)]


@dataclass
class CropRegion:
    """Crop region definition."""
    x: float  # normalized 0-1
    y: float  # normalized 0-1
    width: float  # normalized 0-1
    height: float  # normalized 0-1


@dataclass
class RenderOptions:
    """Render options for a frame."""
    show_scrubber: bool = True
    show_fps: bool = True
    show_overlay: bool = True
    grading: Optional[GradingSettings] = None
    crop: Optional[CropRegion] = None
    playback_speed: float = 1.0


class Engine2D:
    """
    Main engine class for 2D video rendering.
    
    This class provides a Pythonic interface to the Motive 2D render pipeline,
    handling engine initialization, video loading, rendering, and cleanup.
    """
    
    def __init__(self, window_width: int = 1280, window_height: int = 720, 
                 window_title: str = "Motive 2D"):
        """
        Initialize the 2D engine and create a window.
        
        Args:
            window_width: Initial window width in pixels
            window_height: Initial window height in pixels
            window_title: Window title
        """
        self._window_width = window_width
        self._window_height = window_height
        self._window_title = window_title
        self._engine_ptr = None
        self._display_ptr = None
        self._video_loaded = False
        self._playing = True
        self._current_time = 0.0
        self._duration = 0.0
        self._grading_settings = GradingSettings()
        self._crop_region = None
        self._render_options = RenderOptions()
        
        # Initialize engine if library is available
        if _lib is not None:
            self._init_engine()
        else:
            print("[Engine2D] Running in simulation mode (no Vulkan backend)")
    
    def _init_engine(self):
        """Initialize the underlying C++ engine."""
        # Define function prototypes
        _lib.create_engine.restype = ctypes.c_void_p
        _lib.create_engine.argtypes = []
        
        _lib.create_engine_window.argtypes = [
            ctypes.c_void_p,  # engine
            ctypes.c_int,     # width
            ctypes.c_int,     # height
            ctypes.c_char_p   # title
        ]
        
        _lib.destroy_engine.argtypes = [ctypes.c_void_p]
        
        # Create engine instance
        self._engine_ptr = _lib.create_engine()
        if not self._engine_ptr:
            raise RuntimeError("Failed to create engine")
        
        # Create window
        title_bytes = self._window_title.encode('utf-8')
        _lib.create_engine_window(self._engine_ptr, 
                                  self._window_width, 
                                  self._window_height, 
                                  title_bytes)
        
        print(f"[Engine2D] Engine2D initialized with {self._window_width}x{self._window_height} window")
    
    def load_video(self, file_path: str) -> VideoInfo:
        """
        Load a video file for playback.
        
        Args:
            file_path: Path to video file
            
        Returns:
            VideoInfo object containing video metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        # In a real implementation, this would call C++ functions to load video
        # For now, simulate with a placeholder
        self._video_path = file_path
        self._video_loaded = True
        
        # Simulate video info (would be parsed from file)
        info = VideoInfo(
            width=1920,
            height=1080,
            framerate=30.0,
            duration=60.0,  # 1 minute
            codec=VideoCodec.H265,
            pixel_format=PixelFormat.NV12
        )
        
        self._duration = info.duration
        print(f"[Engine2D] Loaded video: {file_path}")
        print(f"  Resolution: {info.width}x{info.height}")
        print(f"  Framerate: {info.framerate} fps")
        print(f"  Duration: {info.duration:.2f}s")
        
        return info
    
    def set_grading(self, settings: GradingSettings):
        """
        Apply color grading settings.
        
        Args:
            settings: GradingSettings object
        """
        self._grading_settings = settings
        print(f"[Engine2D] Grading updated: exposure={settings.exposure}, "
              f"contrast={settings.contrast}, saturation={settings.saturation}")
    
    def set_crop(self, region: CropRegion):
        """
        Set crop region for video.
        
        Args:
            region: CropRegion object (normalized coordinates 0-1)
        """
        self._crop_region = region
        print(f"[Engine2D] Crop set: ({region.x:.2f}, {region.y:.2f}) "
              f"{region.width:.2f}x{region.height:.2f}")
    
    def play(self):
        """Start or resume playback."""
        self._playing = True
        print("[Engine2D] Playback started")
    
    def pause(self):
        """Pause playback."""
        self._playing = False
        print("[Engine2D] Playback paused")
    
    def seek(self, time_seconds: float):
        """
        Seek to specific time in video.
        
        Args:
            time_seconds: Time in seconds
        """
        self._current_time = max(0.0, min(time_seconds, self._duration))
        print(f"[Engine2D] Seek to {self._current_time:.2f}s")
    
    def render_frame(self) -> bool:
        """
        Render a single frame.
        
        Returns:
            True if rendering should continue, False if window closed
        """
        if not self._video_loaded:
            print("[Engine2D] No video loaded")
            return False
        
        # Simulate frame advancement
        if self._playing:
            self._current_time += 1.0 / 30.0  # Assume 30 fps
            if self._current_time > self._duration:
                self._current_time = 0.0
        
        # In real implementation, this would call C++ render function
        # For simulation, just print progress
        progress = self._current_time / self._duration if self._duration > 0 else 0.0
        print(f"[Engine2D] Rendering frame at {self._current_time:.2f}s "
              f"({progress*100:.1f}%)", end='\r')
        
        # Simulate window close with ESC key (in real implementation, check GLFW)
        return True
    
    def run(self):
        """
        Run the main render loop.
        
        This blocks until the window is closed.
        """
        print("[Engine2D] Starting render loop (press ESC to exit)")
        
        try:
            while self.render_frame():
                # In real implementation, this would handle events and swap buffers
                time.sleep(1.0 / 60.0)  # Simulate 60 Hz refresh
                
                # Check for exit condition
                # In real implementation, check glfwWindowShouldClose
        except KeyboardInterrupt:
            print("\n[Engine2D] Interrupted by user")
        finally:
            self.cleanup()
    
    def save_grading_preset(self, file_path: str):
        """
        Save current grading settings to a JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        data = asdict(self._grading_settings)
        # Convert tuple to list for JSON serialization
        data['shadows'] = list(data['shadows'])
        data['midtones'] = list(data['midtones'])
        data['highlights'] = list(data['highlights'])
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Engine2D] Grading preset saved to {file_path}")
    
    def load_grading_preset(self, file_path: str):
        """
        Load grading settings from a JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        if not os.path.exists(file_path):
            print(f"[Engine2D] Grading preset not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to tuples
        data['shadows'] = tuple(data['shadows'])
        data['midtones'] = tuple(data['midtones'])
        data['highlights'] = tuple(data['highlights'])
        
        self._grading_settings = GradingSettings(**data)
        print(f"[Engine2D] Grading preset loaded from {file_path}")
    
    def get_frame_data(self) -> Optional[np.ndarray]:
        """
        Get current frame as numpy array (for analysis/processing).
        
        Returns:
            RGB image as numpy array, or None if not available
        """
        # In real implementation, this would read back from GPU
        # For simulation, return a dummy array
        if not self._video_loaded:
            return None
        
        # Create a dummy gradient image
        height, width = 480, 640
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient based on playback position
        progress = self._current_time / self._duration if self._duration > 0 else 0.0
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        
        for c in range(3):
            offset = int(progress * 255)
            img[:, :, c] = np.roll(gradient, offset).reshape(1, -1)
        
        return img
    
    def cleanup(self):
        """Clean up resources."""
        print("[Engine2D] Cleaning up resources...")
        
        if _lib is not None and self._engine_ptr:
            _lib.destroy_engine(self._engine_ptr)
            self._engine_ptr = None
        
        self._video_loaded = False
        print("[Engine2D] Cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.cleanup()


# Example usage
def example():
    """Example demonstrating Engine2D usage."""
    print("=== Engine2D Example ===")
    
    # Create engine
    engine = Engine2D(window_width=1280, window_height=720)
    
    try:
        # Try to load a video (use test file if available, otherwise simulate)
        test_video = "input.h264"  # Test file that might exist
        if os.path.exists(test_video):
            video_info = engine.load_video(test_video)
        else:
            print(f"[Engine2D] Test video '{test_video}' not found, using simulation")
            # Create a simulated video info
            video_info = VideoInfo(
                width=1920,
                height=1080,
                framerate=30.0,
                duration=10.0,
                codec=VideoCodec.H264,
                pixel_format=PixelFormat.NV12
            )
            engine._video_loaded = True
            engine._duration = video_info.duration
        
        # Apply grading
        grading = GradingSettings(
            exposure=0.2,
            contrast=1.1,
            saturation=1.2,
            shadows=(0.9, 0.9, 1.0),
            midtones=(1.0, 1.0, 1.0),
            highlights=(1.1, 1.05, 1.0)
        )
        engine.set_grading(grading)
        
        # Set crop region
        crop = CropRegion(x=0.1, y=0.1, width=0.8, height=0.8)
        engine.set_crop(crop)
        
        # Save grading preset
        engine.save_grading_preset("my_grading.json")
        
        # Load it back
        engine.load_grading_preset("my_grading.json")
        
        # Run for a few seconds
        print("\nRunning simulation for 3 seconds...")
        start_time = time.time()
        frame_count = 0
        while time.time() - start_time < 3.0:
            if not engine.render_frame():
                break
            frame_count += 1
            time.sleep(0.033)  # ~30 fps
        
        print(f"\n\nRendered {frame_count} frames")
        print("Example completed successfully!")
        
        # Clean up the temporary file
        if os.path.exists("my_grading.json"):
            os.remove("my_grading.json")
            print("Cleaned up temporary grading preset file")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.cleanup()


if __name__ == "__main__":
    example()
