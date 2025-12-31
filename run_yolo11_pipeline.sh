#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_yolo11_pipeline.sh [--workers N] [--pose] [--model <path>] <video1> [video2 ...]
Example: run_yolo11_pipeline.sh --workers 3 videoA.MOV videoB.MOV

Options:
  --workers N     Maximum number of concurrent docker runs (default: 2)
  --pose          Force the pose model for every job
  --model PATH    Override the model used for every job
EOF
}

WORKERS="${WORKERS:-2}"
POSE=false
MODEL_OVERRIDE=""
SEGMENT_DIR=""

cleanup_segments() {
  if [[ -n "$SEGMENT_DIR" && -d "$SEGMENT_DIR" ]]; then
    rm -rf "$SEGMENT_DIR"
  fi
}
trap cleanup_segments EXIT

while [[ $# -gt 0 && "${1:-}" == --* ]]; do
  case "$1" in
    --workers)
      shift
      if [[ $# -eq 0 ]]; then
        echo "ERROR: --workers requires a value" >&2
        usage
        exit 2
      fi
      WORKERS="$1"
      if ! [[ "$WORKERS" =~ ^[0-9]+$ && "$WORKERS" -ge 1 ]]; then
        echo "ERROR: --workers value must be a positive integer" >&2
        exit 2
      fi
      shift
      ;;
    --pose)
      POSE=true
      shift
      ;;
    --model)
      shift
      if [[ $# -eq 0 ]]; then
        echo "ERROR: --model requires a path" >&2
        usage
        exit 2
      fi
      MODEL_OVERRIDE="$1"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

INPUT_VIDEOS=("$@")

jobs_running() {
  jobs -pr | wc -l
}

wait_for_slot() {
  while true; do
    local running
    running="$(jobs_running)"
    if (( running < WORKERS )); then
      break
    fi
    sleep 1
  done
}

VIDEO_JOBS=("${INPUT_VIDEOS[@]}")
if [[ ${#INPUT_VIDEOS[@]} -eq 1 && "$WORKERS" -gt 1 ]]; then
  VIDEO_TO_SPLIT="${INPUT_VIDEOS[0]}"
  if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "ERROR: ffmpeg and ffprobe are required for segmenting the video" >&2
    exit 1
  fi

  SEGMENT_PLAN=$(python3 "$VIDEO_TO_SPLIT" "$WORKERS" <<'PY'
import json, math, subprocess, sys

video = sys.argv[1]
workers = int(sys.argv[2])
if workers <= 0:
    sys.exit(0)

info = subprocess.run([
    "ffprobe",
    "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=duration,nb_frames,avg_frame_rate",
    "-of", "json",
    video,
], capture_output=True, text=True, check=True)

data = json.loads(info.stdout)
stream = data.get("streams", [{}])[0]
duration = float(stream.get("duration") or 0.0)
nb_frames = stream.get("nb_frames")
avg_frame_rate = stream.get("avg_frame_rate") or "0/1"

def parse_rate(rate):
    if rate and "/" in rate:
        num, den = rate.split("/")
        if float(den) == 0:
            return 0.0
        return float(num) / float(den)
    if rate:
        try:
            return float(rate)
        except ValueError:
            return 0.0
    return 0.0

fps = parse_rate(avg_frame_rate)

frames = None
if nb_frames and nb_frames not in ("N/A", "0"):
    frames = int(float(nb_frames))
elif duration > 0 and fps > 0:
    frames = int(math.ceil(duration * fps))

if not frames or frames <= 0:
    raise SystemExit("Unable to determine frame count for segmentation")

if fps <= 0:
    fps = 30.0

delta = max(1, math.ceil(frames / workers))
segments = []
for i in range(workers):
    start = i * delta
    if start >= frames:
        break
    end = min(frames - 1, start + delta - 1)
    segments.append((start, end, end - start + 1))

print(f"frames {frames}")
print(f"segments {len(segments)}")
print(f"fps {fps:.6f}")
for start, end, count in segments:
    print(f"{start}:{end}:{count}")
PY
)

  if [[ -z "$SEGMENT_PLAN" ]]; then
    echo "ERROR: failed to compute segment timings" >&2
    exit 1
  fi

  SEGMENT_DIR=$(mktemp -d run_yolo11_segments.XXXX)
  ext="${VIDEO_TO_SPLIT##*.}"
  segments=()
  index=0
  SEGMENT_FPS=30
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$line" == fps* ]]; then
      SEGMENT_FPS="${line#fps }"
      continue
    fi
    IFS=':' read -r start_frame end_frame count <<< "$line"
    if [[ -z "$start_frame" || -z "$end_frame" || -z "$count" ]]; then
      continue
    fi
    segment_path="$SEGMENT_DIR/segment_${index}.${ext}"
    ffmpeg_cmd=(ffmpeg -y -i "$VIDEO_TO_SPLIT"
      -vf "select='between(n,$start_frame,$end_frame)'"
      -vsync 0 -frames:v "$count"
      -loglevel error
      -c:v libx264 -preset ultrafast -crf 24 -pix_fmt yuv420p
      -movflags +faststart -reset_timestamps 1 -avoid_negative_ts make_zero
      -an -r "$SEGMENT_FPS" "$segment_path")
    echo "[segment] Creating $segment_path (frames $start_frame-$end_frame count=$count)"
    log_file="$SEGMENT_DIR/segment_${index}.log"
    if ! "${ffmpeg_cmd[@]}" >"$log_file" 2>&1; then
      echo "[segment] ffmpeg failed ($segment_path):"
      cat "$log_file"
      rm -f "$segment_path"
      continue
    fi
    echo "[segment] Created $segment_path (size=$(stat -c '%s' "$segment_path") bytes)"
    segments+=("$segment_path")
    index=$((index + 1))
  done <<<"$SEGMENT_PLAN"

  if [[ ${#segments[@]} -eq 0 ]]; then
    echo "ERROR: no segments could be created from $VIDEO_TO_SPLIT" >&2
    exit 1
  fi

  VIDEO_JOBS=("${segments[@]}")
  if [[ ${#VIDEO_JOBS[@]} -lt WORKERS ]]; then
    WORKERS="${#VIDEO_JOBS[@]}"
  fi
fi

for video in "${VIDEO_JOBS[@]}"; do
  if [[ ! -f "$video" ]]; then
    echo "Skipping missing video: $video" >&2
    continue
  fi

  wait_for_slot

  {
    CMD=(./run_yolo11_docker.sh)
    if [[ "$POSE" == true ]]; then
      CMD+=(--pose)
    fi
    if [[ -n "$MODEL_OVERRIDE" ]]; then
      CMD+=("$video" "$MODEL_OVERRIDE")
    else
      CMD+=("$video")
    fi
    echo "[pipeline] Starting: ${CMD[*]}"
    set +e
    "${CMD[@]}"
    CODE=$?
    set -e
    if [[ $CODE -ne 0 ]]; then
      echo "[pipeline] Job failed for $video with exit code $CODE" >&2
    fi
  } &
done

wait
