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


def has_libvulkan():
    """
    Best-effort check that the Vulkan loader is available for linking/running.
    """
    # ldconfig is the most reliable on Linux if present.
    if shutil.which("ldconfig"):
        try:
            out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
            if "libvulkan.so" in out:
                return True
        except Exception:
            pass

    # Fallback checks.
    common = [
        Path("/usr/lib/x86_64-linux-gnu/libvulkan.so"),
        Path("/usr/lib64/libvulkan.so"),
        Path("/usr/lib/libvulkan.so"),
        Path("/usr/local/lib/libvulkan.so"),
        Path("/usr/local/lib64/libvulkan.so"),
    ]
    if any(p.exists() for p in common):
        return True

    return False


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


def setup_vulkan_headers():
    ensure_exists_or_exit("Vulkan-Headers")
    print("Vulkan-Headers exists, checking for updates...")
    if is_detached_head("Vulkan-Headers"):
        print("Vulkan-Headers is detached (pinned submodule commit); skipping git pull.")
        return
    run_command("git pull", cwd="Vulkan-Headers")


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

    # Vulkan is handled explicitly and validated separately.
    return {"flags": flags, "enabled_descriptions": enabled, "skipped_descriptions": skipped}


def require_vulkan_headers_and_loader(min_header_version="1.3.277"):
    """
    Fail fast if Vulkan headers and loader are not suitable for Vulkan Video work.
    Note: headers come from pkg-config "vulkan" (we provide a local vulkan.pc),
    loader is libvulkan (system).
    """
    # Header version (from pkg-config "vulkan")
    vulkan_version = pkg_config_version("vulkan")
    if not vulkan_version:
        print("Vulkan headers not found via pkg-config (vulkan).")
        print("Make sure Vulkan-Headers submodule is present and PKG_CONFIG_PATH includes Vulkan-Headers/.pkgconfig.")
        sys.exit(1)

    if not version_gte(vulkan_version, min_header_version):
        print(f"Vulkan headers too old for Vulkan Video work (need >= {min_header_version}, found {vulkan_version}).")
        sys.exit(1)

    # Loader (libvulkan)
    if not has_libvulkan():
        print("Vulkan loader (libvulkan) not found on this system.")
        if shutil.which("apt-get"):
            print("Attempting to install Vulkan loader dev package (requires sudo): libvulkan-dev")
            run_command("sudo apt-get update && sudo apt-get install -y libvulkan-dev")
        if not has_libvulkan():
            print("Still cannot find libvulkan after attempting install.")
            print("Please install a Vulkan loader (e.g. libvulkan-dev / vulkan-loader) and re-run.")
            sys.exit(1)

    # Optional: warn if vulkaninfo is missing (useful for runtime extension checks).
    if shutil.which("vulkaninfo") is None:
        print("Note: vulkaninfo not found; runtime Vulkan Video extension checks will be skipped.")
        if shutil.which("apt-get"):
            print("  You can install it with: sudo apt-get install -y vulkan-tools")


def warn_if_no_vulkan_video_decode_extensions():
    """
    Best-effort runtime capability check. Build can succeed without this, but decode
    will fail at runtime if the driver doesn't expose VK_KHR_video_decode_queue.
    """
    if shutil.which("vulkaninfo") is None:
        return

    try:
        out = subprocess.check_output(["vulkaninfo"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return

    needed = [
        "VK_KHR_video_decode_queue",
        # Codec-specific extensions vary by driver; these are common.
        "VK_KHR_video_decode_h264",
        "VK_KHR_video_decode_h265",
        "VK_KHR_video_decode_av1",
    ]

    present = {k: (k in out) for k in needed}
    if not present["VK_KHR_video_decode_queue"]:
        print("WARNING: Your Vulkan driver does NOT report VK_KHR_video_decode_queue in vulkaninfo.")
        print("         FFmpeg may build with Vulkan enabled, but Vulkan Video *decode* will not work at runtime.")
    else:
        # Helpful info only.
        codecs = [k for k in needed[1:] if present.get(k)]
        if codecs:
            print("Vulkan Video decode extensions detected:", ", ".join(codecs))
        else:
            print("Vulkan decode queue detected, but no codec-specific decode extensions were found (may still be limited).")


def assert_ffmpeg_config_has_vulkan(build_dir: Path):
    """
    Ensure configure actually enabled Vulkan.
    (Do not trust passing --enable-vulkan alone; verify the generated config.)
    """
    config_mak = build_dir / "ffbuild" / "config.mak"
    config_h = build_dir / "config.h"
    config_log = build_dir / "ffbuild" / "config.log"

    missing = [p for p in (config_mak, config_h, config_log) if not p.exists()]
    if missing:
        print("Configure did not produce expected files:", ", ".join(str(p) for p in missing))
        sys.exit(1)

    mak_txt = config_mak.read_text(errors="ignore")
    h_txt = config_h.read_text(errors="ignore")
    log_txt = config_log.read_text(errors="ignore")

    # config.mak is the most consistent place to check CONFIG_* toggles.
    if "CONFIG_VULKAN=yes" not in mak_txt:
        print("FFmpeg configure did not enable Vulkan (CONFIG_VULKAN is not 'yes').")
        # Show a small hint from config.log if present.
        for needle in ["vulkan", "Vulkan"]:
            idx = log_txt.find(needle)
            if idx != -1:
                start = max(0, idx - 400)
                end = min(len(log_txt), idx + 800)
                snippet = log_txt[start:end]
                print("\n--- config.log excerpt (around 'vulkan') ---")
                print(snippet)
                print("--- end excerpt ---\n")
                break
        sys.exit(1)

    # Secondary checks: these macros vary across versions; treat as advisory.
    if ("HAVE_VULKAN" not in h_txt) and ("CONFIG_VULKAN" not in h_txt):
        print("Note: Vulkan enabled in config.mak, but expected Vulkan markers weren't found in config.h (may be normal).")


def assert_ffmpeg_tree_has_vulkan_decode_sources(ffmpeg_dir: Path):
    """
    Ensure the source tree contains Vulkan Video decode implementation source.
    This is a strong indicator you're on a recent-enough FFmpeg snapshot.
    """
    vulkan_decode = ffmpeg_dir / "libavcodec" / "vulkan_decode.c"
    if not vulkan_decode.exists():
        print("FFmpeg source tree is missing libavcodec/vulkan_decode.c")
        print("This usually means your FFmpeg revision does not include Vulkan Video decode support.")
        print("Update FFmpeg (git pull) or switch to a revision/newer release that includes it.")
        sys.exit(1)


def setup_ffmpeg():
    ffmpeg_dir = Path("FFmpeg")
    if ffmpeg_dir.exists():
        print("FFmpeg repository already present, pulling latest changes...")
        if is_detached_head(ffmpeg_dir):
            print("FFmpeg repository is detached; skipping git pull.")
        else:
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
    local_vulkan_pc_dir_abs = local_vulkan_pc_dir.resolve()
    existing_pc_path = os.environ.get("PKG_CONFIG_PATH", "")
    pc_paths = [str(local_vulkan_pc_dir_abs)]
    if existing_pc_path:
        pc_paths.append(existing_pc_path)
    pc_env = ":".join(pc_paths)
    os.environ["PKG_CONFIG_PATH"] = pc_env

    vulkan_include = str(Path("Vulkan-Headers/include").resolve())

    # Fail fast on Vulkan prerequisites for Vulkan Video decode.
    require_vulkan_headers_and_loader(min_header_version="1.3.277")
    warn_if_no_vulkan_video_decode_extensions()

    # Ensure FFmpeg tree is new enough to have Vulkan decode sources.
    assert_ffmpeg_tree_has_vulkan_decode_sources(ffmpeg_dir)

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

    # Debug settings
    debug_enabled = os.environ.get("FFMPEG_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    # Your original script forced this on; keeping the behavior but making it explicit.
    debug_enabled = True

    # IMPORTANT: only pass --extra-cflags/--extra-ldflags ONCE, otherwise later flags override earlier ones.
    extra_cflags = f"-I{vulkan_include}"
    extra_cxxflags = f"-I{vulkan_include}"
    extra_ldflags = ""

    debug_flags = []
    if debug_enabled:
        print("FFmpeg debug build enabled via FFMPEG_DEBUG (debug symbols, no stripping, optimizations off).")
        debug_flags.extend(
            [
                "--disable-optimizations",
                "--disable-stripping",
                "--enable-debug=3",
            ]
        )
        extra_cflags += " -g"
        extra_cxxflags += " -g"
        extra_ldflags += " -g"
    else:
        print("FFmpeg debug build disabled (set FFMPEG_DEBUG=1 to enable debug symbols).")

    configure_flags = [
        f"PKG_CONFIG_PATH={pc_env}",
        "../configure",
        f"--prefix={install_prefix}",
        "--enable-static",
        "--disable-shared",
        "--enable-pic",
        #"--disable-programs",
        "--enable-vulkan",
        "--enable-decoders",
        "--enable-decoder=h265_vulkan",
        "--enable-decoder=h264_vulkan",
        "--enable-decoder=hevc_vulkan",
        "--enable-decoder=av1_vulkan",
        "--enable-decoder=vp9_vulkan",
        "--disable-doc",
        "--enable-gpl",
        "--enable-version3",
        f"--extra-cflags={shlex.quote(extra_cflags)}",
        f"--extra-cxxflags={shlex.quote(extra_cxxflags)}",
    ]

    if extra_ldflags.strip():
        configure_flags.append(f"--extra-ldflags={shlex.quote(extra_ldflags.strip())}")

    configure_flags.extend(hwaccel_options.get("flags", []))
    configure_flags.extend(debug_flags)

    configure_cmd = " ".join(configure_flags)

    build_dir.mkdir(parents=True, exist_ok=True)

    run_command(configure_cmd, cwd=build_dir)

    # Post-configure assertion: ensure Vulkan is truly enabled.
    assert_ffmpeg_config_has_vulkan(build_dir)

    make_cmd = f"make -j{os.cpu_count()}"
    run_command(make_cmd, cwd=build_dir)
    run_command("make install", cwd=build_dir)
    print(f"FFmpeg built and installed to {install_prefix} with static libraries"
          )
    ffmpeg_bin = install_prefix / "bin" / "ffmpeg"
    ffprobe_bin = install_prefix / "bin" / "ffprobe"
    print(f"FFmpeg built and installed to {install_prefix} (static libs + CLI)")
    if ffmpeg_bin.exists():
        print(f"ffmpeg:  {ffmpeg_bin}")
    else:
        print("Warning: ffmpeg binary not found after build/install.")
    if ffprobe_bin.exists():
        print(f"ffprobe: {ffprobe_bin}")
    else:
        print("Warning: ffprobe binary not found after build/install.")



if __name__ == "__main__":
    setup_ffmpeg()
