#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import re
import shlex
from pathlib import Path
import textwrap


def pkg_config_exists(package):
    """Return True if pkg-config can resolve the requested package."""
    result = subprocess.run(
        ["pkg-config", "--exists", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0

def pkg_config_version(package):
    """Return the pkg-config reported version for the package, or None."""
    result = subprocess.run(
        ["pkg-config", "--modversion", package],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def version_gte(found, required):
    """Compare dotted numeric version strings."""
    def split(v):
        return [int(x) for x in re.split(r"[^\d]", v) if x]
    f_parts = split(found)
    r_parts = split(required)
    length = max(len(f_parts), len(r_parts))
    f_parts.extend([0] * (length - len(f_parts)))
    r_parts.extend([0] * (length - len(r_parts)))
    return f_parts >= r_parts

def ensure_local_vulkan_pc():
    """
    Create a pkg-config file for the Vulkan-Headers submodule so FFmpeg can pick
    up a sufficiently new Vulkan header version even if the system version is older.
    """
    header = Path("Vulkan-Headers/include/vulkan/vulkan_core.h")
    ensure_exists_or_exit(header)

    header_text = header.read_text()
    version_match = re.search(
        r"VK_MAKE_API_VERSION\(\s*0\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*VK_HEADER_VERSION\s*\)",
        header_text,
    )
    header_version_match = re.search(r"#define\s+VK_HEADER_VERSION\s+(\d+)", header_text)
    if not (version_match and header_version_match):
        print("Could not parse Vulkan header version from Vulkan-Headers.")
        sys.exit(1)

    major, minor = version_match.groups()
    header_version = header_version_match.group(1)
    full_version = f"{major}.{minor}.{header_version}"

    pc_dir = Path("Vulkan-Headers/.pkgconfig")
    pc_dir.mkdir(parents=True, exist_ok=True)
    pc_path = pc_dir / "vulkan.pc"

    # FFmpeg's configure script includes the header via an absolute path, which skips
    # our pkg-config include flags. Mirror the upstream Vulkan-Headers install layout
    # by exposing vk_video/ next to vulkan_core.h so those relative includes resolve.
    vk_video_src = Path("Vulkan-Headers/include/vk_video")
    vk_video_link = Path("Vulkan-Headers/include/vulkan/vk_video")
    try:
        if vk_video_src.exists():
            vk_video_link.parent.mkdir(parents=True, exist_ok=True)
            if vk_video_link.is_symlink() or vk_video_link.exists():
                if vk_video_link.is_symlink() and vk_video_link.resolve() != vk_video_src.resolve():
                    vk_video_link.unlink()
                else:
                    # Already present and correct; nothing to do.
                    pass
            if not vk_video_link.exists():
                vk_video_link.symlink_to(Path("..") / "vk_video")
    except OSError as exc:
        print(f"Warning: could not ensure vk_video include shim: {exc}", file=sys.stderr)

    prefix = Path("Vulkan-Headers").resolve()
    contents = textwrap.dedent(
        f"""\
        prefix={prefix}
        includedir=${{prefix}}/include

        Name: vulkan
        Description: Vulkan Headers (local)
        Version: {full_version}
        Cflags: -I${{includedir}}
        Libs: -lvulkan
        """
    )
    pc_path.write_text(contents)
    return pc_dir

def run_command(cmd, cwd=None):
    try:
        subprocess.run(cmd, check=True, shell=True, cwd=cwd)
    except subprocess.CalledProcessError:
        print(f"Command failed: {cmd}", file=sys.stderr)
        sys.exit(1)

def ensure_exists_or_exit(directory):
    if not Path(directory).exists():
        print(f"Missing required folder: {directory}")
        print("Please fetch all submodules with:")
        print("    git submodule update --init --recursive")
        sys.exit(1)

def ensure_pkg_with_apt(pkg_config_name, apt_packages, description, extra_check=None):
    """
    Ensure a dependency is available via pkg-config, installing via apt-get if
    possible. Exits with guidance if the dependency is still missing.
    """
    check_fn = extra_check or (lambda: pkg_config_exists(pkg_config_name))

    if check_fn():
        return

    pkg_list = " ".join(apt_packages)
    print(f"{description} not found via pkg-config ({pkg_config_name}).")
    if shutil.which("apt-get"):
        print(f"Attempting to install {description} via apt-get: {pkg_list}")
        run_command(f"sudo apt-get update && sudo apt-get install -y {pkg_list}")
        if check_fn():
            print(f"{description} installed.")
            return
        print(f"{description} still not found after installation attempt.")
    else:
        print("apt-get not available to auto-install dependencies.")

    print(f"Please install {description} and re-run build_deps.py. Suggested packages: {pkg_list}")
    sys.exit(1)

def is_detached_head(directory):
    """Return True if the git repo at directory is in detached HEAD state."""
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "HEAD"],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode != 0

def setup_vulkan_headers():
    ensure_exists_or_exit("Vulkan-Headers")
    print("Vulkan-Headers exists, checking for updates...")
    if is_detached_head("Vulkan-Headers"):
        print("Vulkan-Headers is detached (pinned submodule commit); skipping git pull.")
        return
    run_command("git pull", cwd="Vulkan-Headers")

def setup_glfw():
    ensure_exists_or_exit("glfw")

    build_dir = Path("glfw/build")
    build_dir.mkdir(exist_ok=True)

    print("Building GLFW...")
    cmake_flags = ["-DCMAKE_BUILD_TYPE=Release"]

    have_wayland = pkg_config_exists("wayland-client")
    have_x11 = all(pkg_config_exists(pkg) for pkg in ["x11", "xcursor", "xrandr", "xi"])

    if not have_wayland and not have_x11:
        print("Neither Wayland nor X11 development headers were found.")
        print("Attempting to install X11 dev headers via apt-get...")
        x11_packages = "libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev"
        if shutil.which("apt-get"):
            run_command(f"sudo apt-get update && sudo apt-get install -y {x11_packages}")
            have_x11 = all(pkg_config_exists(pkg) for pkg in ["x11", "xcursor", "xrandr", "xi"])
        else:
            print("apt-get not available; cannot auto-install X11 dependencies.")

        if not have_x11:
            print("Install at least one set of dev packages and rerun build_deps.py.")
            print("For X11 on Ubuntu: sudo apt-get install libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev")
            print("For Wayland on Ubuntu: sudo apt-get install libwayland-dev libxkbcommon-dev wayland-protocols")
            sys.exit(1)

    # Ubuntu 24.04 desktop images often lack Wayland dev headers by default.
    # If wayland-client is missing, disable Wayland so the build can proceed
    # with the X11 path only.
    if not have_wayland:
        print("wayland-client dev package not found; building GLFW with Wayland disabled.")
        print("To enable Wayland, install: libwayland-dev libxkbcommon-dev wayland-protocols")
        cmake_flags.append("-DGLFW_BUILD_WAYLAND=OFF")

    # If X11 (including Xcursor) headers are missing but Wayland is present,
    # build GLFW with Wayland only to avoid missing Xcursor headers.
    if have_wayland and not have_x11:
        print("X11/Xcursor dev packages not found; building GLFW with X11 disabled (Wayland only).")
        print("To enable X11, install: libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev")
        cmake_flags.append("-DGLFW_BUILD_X11=OFF")

    cmake_cmd = "cmake .. " + " ".join(cmake_flags)
    run_command(cmake_cmd, cwd=build_dir)
    run_command(f"make -j{os.cpu_count()}", cwd=build_dir)

    print("GLFW built locally in glfw/build")

def setup_glm():
    ensure_exists_or_exit("glm")
    print("GLM found (header-only library, no build required)")

def setup_ncnn():
    ensure_exists_or_exit("ncnn")
    build_dir = Path("ncnn/build")
    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir = build_dir / "install"
    lib_path = install_dir / "lib" / "libncnn.a"
    if lib_path.exists():
        print("libncnn.a already available; skipping NCNN rebuild.")
        return

    print("Configuring ncnn...")
    cmake_flags = [
        "-DNCNN_BUILD_EXAMPLES=OFF",
        "-DNCNN_BUILD_BENCHMARKS=OFF",
        "-DNCNN_BUILD_TESTS=OFF",
        "-DNCNN_VULKAN=OFF",
        "-DNCNN_SIMPLEVK=OFF",
        "-DNCNN_OPENMP=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
    ]
    cmake_cmd = "cmake .. " + " ".join(cmake_flags)
    run_command(cmake_cmd, cwd=build_dir)

    print("Building ncnn...")
    run_command(f"make -j{os.cpu_count()}", cwd=build_dir)

    print("Installing ncnn into build/install")
    install_path = shlex.quote(str(install_dir))
    run_command(f"cmake --install . --prefix {install_path}", cwd=build_dir)

    print("ncnn built and installed locally in ncnn/build/install")

def setup_ffnvcodec():
    install_dir = Path("/usr/include/ffnvcodec")
    if install_dir.exists():
        print("nv-codec-headers already installed in /usr/include/ffnvcodec; skipping.")
        return
    repo_dir = Path("nv-codec-headers")
    if not repo_dir.exists():
        print("Cloning nv-codec-headers...")
        run_command("git clone https://github.com/FFmpeg/nv-codec-headers.git nv-codec-headers")
    print("Building nv-codec-headers...")
    run_command("make", cwd=repo_dir)
    print("Installing nv-codec-headers (requires sudo)...")
    run_command("sudo make install", cwd=repo_dir)

def cuda_available():
    """Simple detection for CUDA toolkit availability."""
    if shutil.which("nvcc"):
        return True
    cuda_env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_env and Path(cuda_env).exists():
        return True
    default_path = Path("/usr/local/cuda")
    return default_path.exists()


def ffnvcodec_available():
    """Detect whether the NVIDIA ffnvcodec headers are installed."""
    custom_path = os.environ.get("FFNV_CODEC_INCLUDE")
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend(
        [
            Path("/usr/include/ffnvcodec"),
            Path("/usr/local/include/ffnvcodec"),
            Path.home() / ".local/include/ffnvcodec",
        ]
    )

    for directory in candidates:
        if (directory / "nvEncodeAPI.h").exists() or (directory / "nvDecodeAPI.h").exists():
            return True
    return False


def determine_hwaccel_flags():
    """Return configure flags for FFmpeg hardware decode features the system supports."""
    pkg_features = [
        ("--enable-libdrm", ["libdrm"], "libdrm"),
        ("--enable-vaapi", ["libva"], "VAAPI"),
        ("--enable-vdpau", ["vdpau"], "VDPAU"),
        ("--enable-opencl", ["OpenCL"], "OpenCL"),
    ]
    enabled = []
    skipped = []
    flags = []

    for flag, packages, description in pkg_features:
        if all(pkg_config_exists(pkg) for pkg in packages):
            flags.append(flag)
            enabled.append(description)
        else:
            skipped.append(description)

    # Vulkan: ffmpeg master currently requires headers >= 1.3.277
    # Enable Vulkan if we have the required version
    vulkan_version = pkg_config_version("vulkan")
    vulkan_required = "1.3.277"
    if vulkan_version and version_gte(vulkan_version, vulkan_required):
        flags.append("--enable-vulkan")
        enabled.append("Vulkan")
    else:
        msg = f"Vulkan (requires >= {vulkan_required}"
        if vulkan_version:
            msg += f", found {vulkan_version})"
        else:
            msg += ", not found)"
        skipped.append(msg)

    cuda_flags = ["--enable-cuda", "--enable-cuvid", "--enable-nvdec", "--enable-nvenc"]
    if cuda_available() and ffnvcodec_available():
        flags.extend(cuda_flags)
        enabled.append("CUDA/NVDEC")
    else:
        skipped.append("CUDA/NVDEC (requires CUDA toolkit + ffnvcodec headers)")

    return {"flags": flags, "enabled_descriptions": enabled, "skipped_descriptions": skipped}

def setup_ffmpeg():
    ffmpeg_dir = Path("FFmpeg")
    if ffmpeg_dir.exists():
        print("FFmpeg repository already present, pulling latest changes...")
        if is_detached_head(ffmpeg_dir):
            print("FFmpeg repository is detached; skipping git pull.")
            return
        run_command("git pull", cwd=ffmpeg_dir)
    else:
        print("Cloning FFmpeg repository...")
        run_command("git clone https://github.com/FFmpeg/FFmpeg.git FFmpeg")
        print("FFmpeg repository cloned.")

    # Ensure nasm exists for x86 assembly optimizations; otherwise, builds will fail.
    if shutil.which("nasm") is None:
        print("nasm assembler not found. Attempting to install via apt-get...")
        if shutil.which("apt-get"):
            run_command("sudo apt-get update && sudo apt-get install -y nasm")
        else:
            print("apt-get not available; please install nasm manually and re-run.")
            sys.exit(1)

    # Prefer the in-tree Vulkan-Headers by exporting a pkg-config file for them.
    local_vulkan_pc_dir = ensure_local_vulkan_pc()
    # Use absolute path so it works regardless of current working directory
    local_vulkan_pc_dir_abs = local_vulkan_pc_dir.resolve()
    existing_pc_path = os.environ.get("PKG_CONFIG_PATH", "")
    pc_paths = [str(local_vulkan_pc_dir_abs)]
    if existing_pc_path:
        pc_paths.append(existing_pc_path)
    pc_env = ":".join(pc_paths)
    os.environ["PKG_CONFIG_PATH"] = pc_env

    print("Configuring and building FFmpeg with static libraries...")
    build_dir = ffmpeg_dir / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    install_prefix = (build_dir / "install").resolve()
    install_prefix.mkdir(parents=True, exist_ok=True)
    hwaccel_options = determine_hwaccel_flags()
    if hwaccel_options["flags"]:
        print("Enabling FFmpeg hardware decode features:", ", ".join(hwaccel_options["enabled_descriptions"]))
    else:
        print("No optional FFmpeg hardware decode features detected.")
    if hwaccel_options["skipped_descriptions"]:
        print("Skipping unavailable hardware features:", ", ".join(hwaccel_options["skipped_descriptions"]))

    debug_enabled = os.environ.get("FFMPEG_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    debug_enabled = True
    debug_flags = []
    if debug_enabled:
        print("FFmpeg debug build enabled via FFMPEG_DEBUG (debug symbols, no stripping, optimizations off).")
        debug_flags.extend(
            [
                "--disable-optimizations",
                "--disable-stripping",
                "--enable-debug=3",
                '--extra-cflags="-g"',
                '--extra-ldflags="-g"',
            ]
        )
    else:
        print("FFmpeg debug build disabled (set FFMPEG_DEBUG=1 to enable debug symbols).")

    configure_flags = [
        f"PKG_CONFIG_PATH={pc_env}",
        "../configure",
        f"--prefix={install_prefix}",
        "--enable-static",
        "--disable-shared",
        "--enable-pic",
        "--disable-programs",
        "--disable-doc",
        "--enable-gpl",
        "--enable-version3",
    ]
    configure_flags.extend(hwaccel_options.get("flags", []))
    configure_flags.extend(debug_flags)

    configure_cmd = " ".join(configure_flags)
    run_command(configure_cmd, cwd=build_dir)

    make_cmd = f"make -j{os.cpu_count()}"
    run_command(make_cmd, cwd=build_dir)
    run_command("make install", cwd=build_dir)
    print(f"FFmpeg built and installed to {install_prefix} with static libraries")

def setup_freetype():
    freetype_dir = Path("freetype")
    if freetype_dir.exists():
        print("FreeType repository already present, pulling latest changes...")
        if is_detached_head(freetype_dir):
            print("FreeType repository is detached; skipping git pull.")
        else:
            run_command("git pull", cwd=freetype_dir)
    else:
        print("Cloning FreeType repository...")
        run_command("git clone https://github.com/freetype/freetype.git freetype")
        print("FreeType repository cloned.")

    # Ensure system build deps exist before configuring FreeType.
    ensure_pkg_with_apt("zlib", ["zlib1g-dev"], "zlib development files")
    ensure_pkg_with_apt("libpng", ["libpng-dev"], "libpng development files")

    # The bzip2 package on Debian/Ubuntu does not ship a pkg-config file.
    # Accept either pkg-config or presence of headers/libs.
    def bzip2_available():
        if pkg_config_exists("bzip2"):
            return True
        header = Path("/usr/include/bzlib.h")
        lib_candidates = [
            "/usr/lib/x86_64-linux-gnu/libbz2.so",
            "/usr/local/lib/libbz2.so",
        ]
        if header.exists() and any(Path(p).exists() for p in lib_candidates):
            return True
        return False

    ensure_pkg_with_apt(
        "bzip2",
        ["libbz2-dev"],
        "bzip2 development files",
        extra_check=bzip2_available,
    )

    # Brotli is optional in FreeType but required when WOFF2 is enabled.
    def brotli_available():
        if pkg_config_exists("libbrotlidec"):
            return True
        return Path("/usr/include/brotli/decode.h").exists()

    ensure_pkg_with_apt(
        "libbrotlidec",
        ["libbrotli-dev"],
        "Brotli development files",
        extra_check=brotli_available,
    )

    build_dir = freetype_dir / "build"
    install_dir = build_dir / "install"
    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)

    cmake_cmd = (
        f"cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF "
        f"-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX={install_dir}"
    )
    print("Configuring FreeType...")
    run_command(cmake_cmd, cwd=build_dir)

    print("Building FreeType...")
    run_command(f"make -j{os.cpu_count()}", cwd=build_dir)
    run_command("make install", cwd=build_dir)
    print(f"FreeType built and installed to {install_dir}")

def main():
    print("=== Checking and setting up development dependencies ===")
    setup_vulkan_headers()
    setup_glfw()
    setup_glm()
    setup_ncnn()
    setup_ffnvcodec()
    setup_ffmpeg()
    setup_freetype()
    print("=== Setup complete ===")

if __name__ == "__main__":
    main()
