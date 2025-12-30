import ctypes
import os
import time

def main():
    # Load the shared library
    lib_path = os.path.join(os.path.dirname(__file__), 'libvulkanrenderer.so')
    renderer_lib = ctypes.CDLL(lib_path)

    # Define function prototypes
    renderer_lib.create_renderer.restype = ctypes.c_void_p
    renderer_lib.destroy_renderer.argtypes = [ctypes.c_void_p]
    renderer_lib.load_geometry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    renderer_lib.render.argtypes = [ctypes.c_void_p]

    # Create renderer instance
    renderer_ptr = renderer_lib.create_renderer()
    
    # Define geometry (triangle)
    vertices = [
        0.0,  0.5, 0.0,  # top
       -0.5, -0.5, 0.0,  # bottom left
        0.5, -0.5, 0.0   # bottom right
    ]
    
    # Convert to ctypes array
    vertex_array = (ctypes.c_float * len(vertices))(*vertices)
    
    # Load geometry
    renderer_lib.load_geometry(renderer_ptr, vertex_array, len(vertices))
    
    # Render and keep window open for 5 seconds
    renderer_lib.render(renderer_ptr)
    
    # Cleanup
    renderer_lib.destroy_renderer(renderer_ptr)

if __name__ == "__main__":
    main()
