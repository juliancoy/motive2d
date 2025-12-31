#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--profile] [--pose] <video.mp4> [yolo_model.pt]"
  echo "Example: $0 --pose --profile video.mp4"
  echo
  echo "Flags:"
  echo "  --profile    log GPU utilization while the container runs"
  echo "  --pose       request the pose model (overrides the default inside the container)"
}

PROFILE=false
PROFILE_LOG="${PROFILE_LOG:-yolo11_gpu_profile.csv}"
PROFILE_PID=""
POSE=false

terminate_profile() {
  if [[ -n "${PROFILE_PID:-}" ]]; then
    echo "[profile] Stopping GPU logging."
    kill "$PROFILE_PID" >/dev/null 2>&1 || true
    wait "$PROFILE_PID" >/dev/null 2>&1 || true
    PROFILE_PID=""
  fi
}

while [[ $# -gt 0 && "${1:-}" == --* ]]; do
  case "$1" in
    --profile)
      PROFILE=true
      shift
      ;;
    --pose)
      POSE=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

VIDEO="$1"
shift

MODEL_ARG=""
if [[ $# -gt 0 ]]; then
  MODEL_ARG="$1"
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "ERROR: unexpected arguments: $*"
  usage
  exit 2
fi

BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0}"
YOLO_IMAGE="${YOLO_IMAGE:-yolo11:latest}"

if [[ "$PROFILE" == true ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: --profile requires nvidia-smi to be installed on the host." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${PROFILE_LOG}")"
  echo "[profile] Logging GPU usage to ${PROFILE_LOG}"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,pstate --format=csv -lms 1000 > "${PROFILE_LOG}" &
  PROFILE_PID=$!
  trap 'terminate_profile' EXIT
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: video not found: $VIDEO"
  exit 1
fi

if [[ ! -f "./run_yolo11.sh" ]]; then
  echo "ERROR: ./run_yolo11.sh not found (expected in current directory)"
  exit 1
fi

# Ensure the inner script is executable on the host; container will see the same perms via bind mount.
chmod +x ./run_yolo11.sh

DOCKER_ENV=(-e "BATCH=$BATCH" -e "DEVICE=$DEVICE" -e "SKIP_INSTALL=1")
if [[ -n "${COORDS:-}" ]]; then
  DOCKER_ENV+=(-e "COORDS=$COORDS")
fi

INNER_ARGS=()
if [[ "$POSE" == true ]]; then
  INNER_ARGS+=(--pose)
fi
INNER_ARGS+=("$VIDEO")
if [[ -n "${MODEL_ARG:-}" ]]; then
  INNER_ARGS+=("$MODEL_ARG")
fi

docker run --rm \
  -v "$PWD:/work" -w /work \
  --gpus all \
  "${DOCKER_ENV[@]}" \
  "$YOLO_IMAGE" \
  "${INNER_ARGS[@]}"

terminate_profile
