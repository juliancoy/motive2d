#!/usr/bin/env python3
"""Orchestrate GPU-capable YOLO jobs inside the prebuilt docker image."""

import argparse
import atexit
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from docker.types import DeviceRequest
from docker.errors import ContainerError

from docker_utils import run_container

HOST_WORKDIR = Path.cwd()
CONTAINER_WORKDIR = Path("/work")


def get_model_tag(args) -> str:
    if args.pose:
        return "pose"
    if args.model:
        return Path(args.model).stem
    return "detect"


def map_to_container_path(
    host_path: Path, volumes: Dict[str, Dict[str, str]], write: bool = False
) -> Path:
    try:
        rel_path = host_path.relative_to(HOST_WORKDIR)
    except ValueError:
        host_str = str(host_path)
        entry = volumes.get(host_str)
        mode = "rw" if write else "ro"
        if entry:
            if write and entry["mode"] != "rw":
                entry["mode"] = "rw"
        else:
            volumes[host_str] = {"bind": host_str, "mode": mode}
        return Path(host_str)
    return CONTAINER_WORKDIR / rel_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference inside docker with optional segmentation.")
    parser.add_argument("videos", nargs="+", help="Input video(s) to process.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent docker jobs (default: 1).",
    )
    parser.add_argument(
        "--pose", action="store_true", help="Force the pose model for every job."
    )
    parser.add_argument(
        "--model",
        help="Override the YOLO model to use for every job (default uses pose model when --pose is set).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=int(os.environ.get("BATCH", "16")),
        help="Batch size for inference (sets BATCH env).",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "0"),
        help="CUDA device to target (sets DEVICE env).",
    )
    parser.add_argument(
        "--coords",
        help="Custom coords output path (sets COORDS env) per job.",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("YOLO_IMAGE", "yolo11:latest"),
        help="Docker image that contains the baked YOLO environment.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Log host GPU utilization (runs `nvidia-smi` alongside jobs).",
    )
    parser.add_argument(
        "--profile-log",
        default=os.environ.get("PROFILE_LOG", "yolo11_gpu_profile.csv"),
        help="Path to write GPU profile CSV (default: yolo11_gpu_profile.csv).",
    )
    return parser.parse_args()


def start_profile(log_path: Path):
    if not shutil.which("nvidia-smi"):
        raise SystemExit("--profile requires nvidia-smi on the host")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[profile] logging GPU stats to {log_path}")
    log_handle = log_path.open("w")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,pstate",
            "--format=csv",
            "-lms",
            "1000",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return proc, log_handle


def stop_profile(proc_handle):
    proc, handle = proc_handle
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if handle and not handle.closed:
        handle.close()


def probe_video(video: Path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    stream = info.get("streams", [{}])[0]
    duration = float(stream.get("duration") or 0.0)
    nb_frames = stream.get("nb_frames")
    fps_raw = stream.get("avg_frame_rate") or "0/1"
    fps = parse_fps(fps_raw)
    if not fps:
        fps = 30.0
    if nb_frames and nb_frames not in ("N/A", "0"):
        frames = int(float(nb_frames))
    elif duration > 0:
        frames = int(max(1, duration * fps))
    else:
        raise SystemExit("Unable to determine frame count for segmentation")
    return frames, fps


def parse_fps(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den = float(den)
            if den == 0:
                return 0.0
            return float(num) / den
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def compute_segments(video: Path, workers: int):
    frames, fps = probe_video(video)
    per_chunk = max(1, frames // workers + (1 if frames % workers else 0))
    segments = []
    start = 0
    while start < frames:
        end = min(frames - 1, start + per_chunk - 1)
        count = end - start + 1
        segments.append((start, end, count))
        start = end + 1
    return segments, fps


def create_segment_files(video: Path, segments, fps, tmp_dir: Path):
    jobs = []

    ext = "mp4"
    for idx, (start, end, count) in enumerate(segments):
        out_path = tmp_dir / f"segment_{start}_{end}.{ext}"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select='between(n,{start},{end})'",
            "-frames:v",
            str(count),
            "-loglevel",
            "error",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-reset_timestamps",
            "1",
            "-avoid_negative_ts",
            "make_zero",
            "-an",
            "-r",
            f"{fps:.6f}",
            str(out_path),
        ]
        # Reuse existing exports when possible instead of re-running ffmpeg.
        if out_path.exists():
            print(f"[segment] Skipping existing {out_path} (frames {start}-{end})")
            jobs.append(out_path)
            continue
        log_path = tmp_dir / f"segment_{start}_{end}.log"
        print(f"[segment] Creating {out_path} (frames {start}-{end})")
        print(f"[segment] command: {shlex.join(ffmpeg_cmd)}")
        with log_path.open("w") as log_file:
            proc = subprocess.run(ffmpeg_cmd, stdout=log_file, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not out_path.exists():
            print(f"[segment] ffmpeg failed for {out_path} (see {log_path})")
            if out_path.exists():
                out_path.unlink()
            continue
        size = out_path.stat().st_size
        print(f"[segment] Created {out_path} ({size} bytes)")
        jobs.append(out_path)
    return jobs


def run_job(
    idx: int,
    video: Path,
    args,
    output_dir: Path,
    coords_host: Path,
    model_tag: str,
    device_requests=None,
):
    env = {
        "BATCH": str(args.batch),
        "DEVICE": args.device,
        "SKIP_INSTALL": "1",
    }

    command = []
    if args.pose:
        command.append("--pose")
    volumes = {str(HOST_WORKDIR): {"bind": str(CONTAINER_WORKDIR), "mode": "rw"}}
    container_video = map_to_container_path(video, volumes)
    command.append(str(container_video))
    if args.model:
        command.append(args.model)

    output_dir.mkdir(parents=True, exist_ok=True)
    project_path = map_to_container_path(output_dir, volumes, write=True)
    env["PROJECT"] = str(project_path)
    env["NAME"] = f"{video.stem}_{model_tag}"
    out_path = output_dir / f"{video.stem}_{model_tag}_yolo11.mp4"
    env["OUT"] = str(map_to_container_path(out_path, volumes, write=True))
    env["COORDS"] = str(map_to_container_path(coords_host, volumes, write=True))

    config = {
        "image": args.image,
        "name": f"yolo11_job_{idx}_{int(time.time()*1000)}",
        "command": command,
        "volumes": volumes,
        "working_dir": "/work",
        "environment": env,
        "remove": True,
        "detach": False,
    }
    if device_requests:
        config["device_requests"] = device_requests

    print(f"[job {idx}] running {command}")
    try:
        result = run_container(config)
    except ContainerError as err:
        stderr = getattr(err, "stderr", None)
        stdout = getattr(err, "stdout", None)
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else "<no stderr>"
        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else "<no stdout>"
        print(
            f"[job {idx}] container error: {err}. stderr:\n{stderr_text}\nstdout:\n{stdout_text}"
        )
        raise
    if isinstance(result, bytes):
        try:
            output = result.decode("utf-8")
        except UnicodeDecodeError:
            output = str(result)
    else:
        output = str(result)
    print(f"[job {idx}] finished {video.name}\n{output}")


def main():
    args = parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    job_videos = [Path(v).resolve() for v in args.videos]
    input_videos = list(job_videos)
    primary_input_dir = job_videos[0].parent

    segment_dir = None
    if len(job_videos) == 1 and args.workers > 1:
        segment_dir = primary_input_dir / "run_yolo11_segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        segments, fps = compute_segments(job_videos[0], args.workers)
        created = create_segment_files(job_videos[0], segments, fps, segment_dir)
        if not created:
            raise SystemExit("Unable to create any segments for the input video")
        job_videos = created
        args.workers = min(args.workers, len(job_videos))

    profile_proc = None
    if args.profile:
        profile_proc = start_profile(Path(args.profile_log))
        atexit.register(lambda: stop_profile(profile_proc))

    device_requests = [
        DeviceRequest(count=-1, capabilities=[["gpu"]]),
    ]

    concurrency = min(args.workers, len(job_videos))
    futures = []
    model_tag = get_model_tag(args)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for idx, video in enumerate(job_videos):
            if not video.exists():
                print(f"[job {idx}] skipping missing video {video}")
                continue
            coords_host = (
                Path(args.coords).resolve()
                if args.coords
                else video.with_name(f"{video.stem}_{model_tag}_coords.txt")
            )
            if coords_host.exists():
                print(
                    f"[job {idx}] found existing coords {coords_host}; skipping inference for {video}"
                )
                continue
            output_dir = video.parent
            futures.append(
                executor.submit(
                    run_job,
                    idx,
                    video,
                    args,
                    output_dir,
                    coords_host,
                    model_tag,
                    device_requests=device_requests,
                )
            )
        for future in as_completed(futures):
            if future.exception():
                print(f"[pipeline] job raised: {future.exception()}", file=sys.stderr)

    if args.pose:
        combine_pose_outputs(input_videos, model_tag, segment_dir)

    if profile_proc:
        stop_profile(profile_proc)


POSE_DIR_SUFFIX = "_pose"
SEGMENT_DIR_RE = re.compile(r"segment_(\d+)_(\d+)$")


def combine_pose_outputs(videos, model_tag, segment_root):
    """Write combined pose entries for each source video to JSON."""
    for video in videos:
        pose_dirs = collect_pose_dirs(video, segment_root)
        if not pose_dirs:
            print(f"[pose] No pose outputs found for {video.name}")
            continue

        entries = []
        for pose_dir in pose_dirs:
            entries.extend(read_pose_entries(pose_dir))

        if not entries:
            print(f"[pose] No pose entries discovered for {video.name}")
            continue

        entries.sort(key=lambda entry: entry[0])
        output_path = video.with_name(f"{video.stem}_{model_tag}_coords.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        entries_with_frame = [
            {"frame": int(entry[0]), "pose": entry[1]} for entry in entries
        ]
        with output_path.open("w", encoding="utf-8") as out_file:
            json.dump(entries_with_frame, out_file)
        print(f"[pose] Combined {len(entries)} entries into {output_path}")


def collect_pose_dirs(video, segment_root):
    """Return pose directories relevant to the provided video."""
    candidates = []
    seen = set()

    if segment_root and segment_root.is_dir():
        for candidate in segment_root.iterdir():
            if not candidate.is_dir() or not candidate.name.endswith(POSE_DIR_SUFFIX):
                continue
            start = segment_start_offset(candidate)
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append((start, candidate))

    default_dir = video.parent / f"{video.stem}{POSE_DIR_SUFFIX}"
    if default_dir.is_dir():
        resolved = default_dir.resolve()
        if resolved not in seen:
            candidates.append((0, default_dir))

    candidates.sort(key=lambda pair: pair[0])
    return [path for _, path in candidates]


def segment_start_offset(pose_dir):
    """Parse the starting frame from a segment pose directory."""
    base_name = pose_dir.name
    if base_name.endswith(POSE_DIR_SUFFIX):
        base_name = base_name[: -len(POSE_DIR_SUFFIX)]
    match = SEGMENT_DIR_RE.match(base_name)
    if match:
        return int(match.group(1))
    return 0


def read_pose_entries(pose_dir):
    """Load pose label files and return (frame, instance) pairs."""
    base_name = pose_dir.name
    if base_name.endswith(POSE_DIR_SUFFIX):
        base_name = base_name[: -len(POSE_DIR_SUFFIX)]

    start_offset = segment_start_offset(pose_dir)
    label_dir = pose_dir / "labels"
    if not label_dir.is_dir():
        return []

    prefix = f"{base_name}_"
    label_files = []
    for label_file in label_dir.glob("*.txt"):
        stem = label_file.stem
        if not stem.startswith(prefix):
            continue
        frame_part = stem[len(prefix) :]
        if not frame_part.isdigit():
            continue
        label_files.append((int(frame_part), label_file))

    label_files.sort(key=lambda pair: pair[0])

    entries = []
    for frame_index, label_file in label_files:
        abs_frame = start_offset + frame_index
        try:
            with label_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = line.strip()
                    if not payload:
                        continue
                    tokens = payload.split()
                    try:
                        instance = [float(token) for token in tokens]
                    except ValueError:
                        print(f"[pose] Skipping malformed line in {label_file}")
                        continue
                    entries.append([abs_frame, instance])
        except OSError as exc:
            print(f"[pose] Failed to read {label_file}: {exc}")

    return entries


if __name__ == "__main__":
    main()
