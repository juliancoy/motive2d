#!/usr/bin/env python3
"""
Direct NCNN export using Ultralytics CLI inside ultralytics/ultralytics:latest.

Assumption:
- Input model (.pt) lives in some directory on the host.
- Outputs should be written into that SAME directory (no temp dirs, no separate out dirs).
- We run ONE container and call: yolo export model=/work/<file>.pt format=ncnn imgsz=<N>

Idempotency:
- If <stem>.param and <stem>.bin already exist next to the input, we skip (unless --force).
- After export, we locate the produced .param/.bin (Ultralytics may put them in a subfolder)
  and normalize them to:
    <input_dir>/<stem>.param
    <input_dir>/<stem>.bin
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def die(msg: str, code: int = 1) -> None:
    print(msg, flush=True)
    raise SystemExit(code)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def find_param_bin(root: Path) -> tuple[Path, Path] | None:
    """Find a .param/.bin pair somewhere under root. Prefer same-stem pairs, else largest."""
    params = sorted(root.rglob("*.param"))
    bins = sorted(root.rglob("*.bin"))
    if not params or not bins:
        return None

    bins_by_stem = {}
    for b in bins:
        bins_by_stem.setdefault(b.stem, []).append(b)

    # Prefer a param/bin with matching stem
    for p in params:
        cands = bins_by_stem.get(p.stem)
        if cands:
            # If multiple, pick largest
            cands.sort(key=lambda x: -x.stat().st_size)
            return p, cands[0]

    # Fallback: largest of each (best-effort)
    params.sort(key=lambda x: -x.stat().st_size)
    bins.sort(key=lambda x: -x.stat().st_size)
    return params[0], bins[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to .pt model on host")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--image", default="ultralytics/ultralytics:latest")
    ap.add_argument("--force", action="store_true", help="Re-export even if outputs exist")
    ap.add_argument("--selfrun", action="store_true", help="Print docker command only")
    ap.add_argument(
        "--name",
        default="ultra_export_ncnn",
        help="Container name (set empty string to omit --name)",
    )
    args = ap.parse_args()

    input_pt = Path(args.input).resolve()
    if not input_pt.exists():
        die(f"Input not found: {input_pt}")
    if input_pt.suffix.lower() != ".pt":
        print(f"Warning: input does not end with .pt: {input_pt}", flush=True)

    work_dir = input_pt.parent
    stem = input_pt.stem

    # Final normalized outputs (next to input)
    dst_param = work_dir / f"{stem}.param"
    dst_bin = work_dir / f"{stem}.bin"

    # Idempotency
    if dst_param.exists() and dst_bin.exists() and not args.force:
        print(f"✅ Outputs already exist; skipping:")
        print(f"  {dst_param}")
        print(f"  {dst_bin}")
        return

    if args.force:
        dst_param.unlink(missing_ok=True)
        dst_bin.unlink(missing_ok=True)

    # Docker command: mount the input directory as /work and export from /work/<file>
    container_model_path = f"/work/{input_pt.name}"

    docker_cmd = ["docker", "run", "--rm"]
    if args.name:
        docker_cmd += ["--name", args.name]
    docker_cmd += [
        "-v",
        f"{work_dir}:/work:rw",
        args.image,
        "bash",
        "-lc",
        # Use bash -lc so the image's environment/entrypoints are respected.
        f"yolo export model={container_model_path} format=ncnn imgsz={args.imgsz}",
    ]

    if args.selfrun:
        print(" ".join(docker_cmd))
        return

    # Run export
    run(docker_cmd)

    # Ultralytics may put outputs in a subfolder (e.g., *_ncnn_model). Find them and normalize.
    found = find_param_bin(work_dir)
    if not found:
        die(f"No .param/.bin outputs found anywhere under: {work_dir}")

    src_param, src_bin = found

    # If it already produced exactly the target names in-place, great.
    # Otherwise, move/rename into the same directory as input with normalized names.
    if src_param.resolve() != dst_param.resolve():
        if dst_param.exists() and not args.force:
            die(f"Refusing to overwrite {dst_param} (use --force).")
        shutil.move(str(src_param), str(dst_param))

    if src_bin.resolve() != dst_bin.resolve():
        if dst_bin.exists() and not args.force:
            die(f"Refusing to overwrite {dst_bin} (use --force).")
        shutil.move(str(src_bin), str(dst_bin))

    if not (dst_param.exists() and dst_bin.exists()):
        die(f"Expected outputs missing:\n  {dst_param}\n  {dst_bin}")

    print("✅ Done:")
    print(f"  {dst_param}")
    print(f"  {dst_bin}")


if __name__ == "__main__":
    main()
