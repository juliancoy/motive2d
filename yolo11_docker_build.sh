#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-yolo11:latest}"
DOCKERFILE="${2:-YOLO11.dockerfile}"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: dockerfile not found: $DOCKERFILE"
  exit 1
fi

echo "Building YOLO11 image as $TAG using $DOCKERFILE"
docker build -t "$TAG" -f "$DOCKERFILE" .
