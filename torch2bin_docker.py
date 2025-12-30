#!/usr/bin/env python3
"""
Host orchestrator: SINGLE-step conversion
  .pt -> NCNN (.param/.bin) using Ultralytics CLI inside ultralytics/ultralytics:latest

Assumptions:
- Input .pt and outputs should live in the SAME directory as the input file.
- Container mounts that directory at /work and runs:
    yolo export model=/work/<file>.pt format=ncnn imgsz=<imgsz>

Idempotency:
- If <stem>.param and <stem>.bin exist next to the input, skip (unless --force).
- After export, Ultralytics writes into <stem>_ncnn_model/; we normalize by copying/moving
  model.ncnn.param/bin to <stem>.param/bin in the input directory.

Notes:
- This script does NOT require torch2pnnx.py or pnnx2bin.py.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import docker_utils


def die(msg: str, code: int = 1) -> None:
    print(msg, flush=True)
    raise SystemExit(code)


def decode_logs(logs) -> str:
    if isinstance(logs, (bytes, bytearray)):
        return logs.decode("utf-8", errors="replace")
    return str(logs)


def find_ncnn_pair(export_dir: Path) -> tuple[Path, Path] | None:
    """
    Ultralytics exports NCNN as:
      <export_dir>/model.ncnn.param
      <export_dir>/model.ncnn.bin
    but we also fall back to searching for any *.ncnn.param/*.ncnn.bin pair.
    """
    p1 = export_dir / "model.ncnn.param"
    b1 = export_dir / "model.ncnn.bin"
    if p1.exists() and b1.exists():
        return p1, b1

    # fallback: search
    params = sorted(export_dir.rglob("*.ncnn.param"))
    bins = sorted(export_dir.rglob("*.ncnn.bin"))
    if not params or not bins:
        return None

    bins_by_stem = {b.stem: b for b in bins}
    for p in params:
        b = bins_by_stem.get(p.stem)
        if b:
            return p, b

    # last resort: largest
    params.sort(key=lambda x: -x.stat().st_size)
    bins.sort(key=lambda x: -x.stat().st_size)
    return params[0], bins[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Path to .pt file on host")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--image", default="ultralytics/ultralytics:latest")
    p.add_argument("--selfrun", action="store_true", help="Print docker command instead of running")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if they already exist")
    p.add_argument("--keep-export-dir", action="store_true", help="Keep <stem>_ncnn_model dir after normalization")
    args = p.parse_args()

    input_pt = Path(args.input).resolve()
    if not input_pt.exists():
        die(f"Input not found: {input_pt}")

    work_dir = input_pt.parent
    stem = input_pt.stem

    dst_param = work_dir / f"{stem}.param"
    dst_bin = work_dir / f"{stem}.bin"

    def outputs_exist() -> bool:
        return dst_param.exists() and dst_bin.exists()

    if outputs_exist() and not args.force:
        print("✅ Outputs already exist; skipping (use --force to rerun).")
        print(dst_param)
        print(dst_bin)
        return

    if args.force:
        dst_param.unlink(missing_ok=True)
        dst_bin.unlink(missing_ok=True)

    export_dir = work_dir / f"{stem}_ncnn_model"

    # One-shot container run.
    # Using bash -lc so the CLI resolves properly in the conda env.
    cmd = (
        f"yolo export model=/work/{input_pt.name} format=ncnn imgsz={args.imgsz}"
    )

    config = {
        "name": "ultra_export_ncnn",
        "image": args.image,
        "command": ["bash", "-lc", cmd],
        "volumes": {
            str(work_dir): {"bind": "/work", "mode": "rw"},
        },
        "remove": True,
        "detach": False,
    }

    if args.selfrun:
        # Best-effort printable docker run command
        docker_cmd = (
            f"docker run --rm --name ultra_export_ncnn "
            f"-v {work_dir}:/work:rw {args.image} bash -lc {cmd!r}"
        )
        print(docker_cmd)
        return

    out = docker_utils.run_container(config)
    logs = decode_logs(out)
    if logs.strip():
        print(logs)

    # Locate export dir + files
    if not export_dir.exists():
        # Sometimes Ultralytics might place results elsewhere; scan as fallback.
        # But your run shows it uses /work/<stem>_ncnn_model.
        die(f"Expected export dir missing: {export_dir}")

    found = find_ncnn_pair(export_dir)
    if not found:
        die(f"Could not find NCNN outputs under: {export_dir}")

    src_param, src_bin = found

    # Normalize into input directory
    shutil.copy2(src_param, dst_param)
    shutil.copy2(src_bin, dst_bin)

    if not outputs_exist():
        die(f"Normalization failed; missing:\n  {dst_param}\n  {dst_bin}")

    if not args.keep_export_dir:
        shutil.rmtree(export_dir, ignore_errors=True)

    print("✅ Done:")
    print(f"  {dst_param}")
    print(f"  {dst_bin}")
    if args.keep_export_dir:
        print(f"  (Kept export directory: {export_dir})")


if __name__ == "__main__":
    main()
