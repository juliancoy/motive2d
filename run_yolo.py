#!/usr/bin/env python3
"""Lightweight wrapper that runs ultralytics YOLO for pose detection and writes newline JSON output."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO pose inference and stream coordinates.")
    parser.add_argument("--source", required=True, help="Path to the source video inside the container.")
    parser.add_argument("--model", required=True, help="YOLO model name or path inside the container.")
    parser.add_argument("--project", required=True, help="Directory where YOLO stores intermediate outputs.")
    parser.add_argument("--name", required=True, help="Name prefix for the YOLO run.")
    parser.add_argument(
        "--batch",
        type=int,
        default=int(os.environ.get("BATCH", "16")),
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "0"),
        help="Device string that is passed to YOLO.",
    )
    parser.add_argument("--coords", required=True, help="Path where pose coordinates will be written.")
    return parser.parse_args()


def probe_video_dimensions(source: str) -> Tuple[float, float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                source,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffprobe failed: {exc}") from exc

    info = json.loads(result.stdout or "{}")
    stream = info.get("streams", [{}])[0]
    width = float(stream.get("width") or 0.0)
    height = float(stream.get("height") or 0.0)
    if width <= 0.0 or height <= 0.0:
        raise SystemExit(f"Unable to determine video dimensions for {source}")
    return width, height


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def main():
    args = parse_args()

    source = args.source
    model_name = args.model
    project = Path(args.project)
    name = args.name
    batch_size = args.batch
    device = args.device
    coords_path = Path(args.coords)
    video_width, video_height = probe_video_dimensions(source)

    print(f"Model:  {model_name}")
    print(f"Source: {source}")
    print(f"Device: {device} (batch={batch_size})")

    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    predict_iter = model.predict(
        source=source,
        project=str(project),
        name=name,
        device=device,
        batch=batch_size,
        save=False,
        save_txt=False,
        stream=True,
        exist_ok=True,
    )

    entries_written = 0
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    with coords_path.open("w", encoding="utf-8") as coords_file:
        for frame_index, result in enumerate(predict_iter):
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xywh is None:
                continue
            coords_xywh = boxes.xywh.cpu().tolist()
            cls_values = (
                boxes.cls.cpu().tolist()
                if getattr(boxes, "cls", None) is not None
                else [0.0] * len(coords_xywh)
            )
            keypoints = getattr(result, "keypoints", None)
            kp_values = None
            if keypoints is not None:
                kp_tensor = getattr(keypoints, "xy", None)
                if kp_tensor is None:
                    kp_tensor = getattr(keypoints, "data", None)
                if kp_tensor is not None:
                    kp_values = kp_tensor.cpu().tolist()

            for idx, box_data in enumerate(coords_xywh):
                cls_val = cls_values[idx] if idx < len(cls_values) else 0.0
                width_inv = 1.0 / video_width
                height_inv = 1.0 / video_height
                norm_x = clamp01(float(box_data[0]) * width_inv)
                norm_y = clamp01(float(box_data[1]) * height_inv)
                norm_w = clamp01(float(box_data[2]) * width_inv)
                norm_h = clamp01(float(box_data[3]) * height_inv)
                pose_entry = [float(cls_val), norm_x, norm_y, norm_w, norm_h]
                kp_list = kp_values[idx] if kp_values and idx < len(kp_values) else []
                if kp_list:
                    for kp_point in kp_list:
                        if len(kp_point) >= 3:
                            pose_entry.extend(
                                [
                                    clamp01(float(kp_point[0]) * width_inv),
                                    clamp01(float(kp_point[1]) * height_inv),
                                    float(kp_point[2]),
                                ]
                            )
                        else:
                            pose_entry.extend(
                                [
                                    clamp01(float(kp_point[0]) * width_inv),
                                    clamp01(float(kp_point[1]) * height_inv),
                                    1.0,
                                ]
                            )
                else:
                    pose_entry.extend([0.0, 0.0, 0.0] * 17)
                entry = {"frame": frame_index, "pose": pose_entry}
                coords_file.write(json.dumps(entry) + "\n")
                entries_written += 1

    if entries_written == 0:
        if coords_path.exists():
            coords_path.unlink()
        raise SystemExit("No results returned by the YOLO model.")

    print(f"✅ Saved pose coords: {coords_path.resolve()}")


if __name__ == "__main__":
    main()
