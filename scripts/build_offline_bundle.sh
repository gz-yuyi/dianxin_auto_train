#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 [--output DIR]

Options:
  -o, --output DIR   Absolute or relative path for the staging directory.
                     Defaults to ./offline_bundle_<timestamp> under the repo root.
  -h, --help         Show this message.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH" >&2
    exit 1
  fi
}

timestamp="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_DIR="$REPO_ROOT/offline_bundle_${timestamp}"
CUSTOM_OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      CUSTOM_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

TARGET_DIR="${CUSTOM_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

if [[ "$TARGET_DIR" != /* ]]; then
  TARGET_DIR="$REPO_ROOT/$TARGET_DIR"
fi

DX_IMAGE_TAG="${DX_IMAGE_TAG:-latest}"
APP_IMAGE="${APP_IMAGE:-crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/dianxin_auto_train:${DX_IMAGE_TAG}}"
REDIS_IMAGE="${REDIS_IMAGE:-m.daocloud.io/docker.io/redis:7-alpine}"
MODEL_NAME="${MODEL_NAME:-google-bert/bert-base-chinese}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-bert-base-chinese}"

require_cmd docker
require_cmd tar

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run python)
else
  require_cmd python
  PYTHON_RUNNER=(python)
fi

sanitize_image_name() {
  echo "$1" | sed 's#/#_#g; s#:#-#g'
}

mkdir -p "$TARGET_DIR"/{images,models,compose,data,artifacts}

cp "$REPO_ROOT/docker-compose.yml" "$TARGET_DIR/compose/docker-compose.yml"
cp "$REPO_ROOT/.env.offline.example" "$TARGET_DIR/compose/.env.example"

IMAGE_LIST=("$APP_IMAGE" "$REDIS_IMAGE")

for image in "${IMAGE_LIST[@]}"; do
  echo "Pulling ${image}..."
  docker pull "$image"
  archive_name="$(sanitize_image_name "$image").tar"
  echo "Exporting ${image} to ${archive_name}..."
  docker save "$image" -o "$TARGET_DIR/images/${archive_name}"
done

MODEL_TARGET_DIR="$TARGET_DIR/models/$MODEL_DIR_NAME"
rm -rf "$MODEL_TARGET_DIR"
mkdir -p "$MODEL_TARGET_DIR"

echo "Downloading model '$MODEL_NAME' from '$MODEL_SOURCE'..."
(
  cd "$REPO_ROOT"
  "${PYTHON_RUNNER[@]}" main.py download-model \
    --model-name "$MODEL_NAME" \
    --output-dir "$MODEL_TARGET_DIR" \
    --source "$MODEL_SOURCE"
)

cp "$REPO_ROOT/docs/offline_deployment.md" "$TARGET_DIR/README_OFFLINE.md"

MANIFEST="$TARGET_DIR/bundle_manifest.txt"
cat >"$MANIFEST" <<EOF
Dianxin Auto Train offline bundle
Generated: ${timestamp}

Application image: ${APP_IMAGE}
Redis image: ${REDIS_IMAGE}
Model source: ${MODEL_SOURCE}
Model name: ${MODEL_NAME}
Model directory name: ${MODEL_DIR_NAME}

Contents:
- images/ (docker save archives)
- models/${MODEL_DIR_NAME} (pre-downloaded model)
- compose/docker-compose.yml
- compose/.env.example
- data/ (drop training data here)
- artifacts/ (model outputs)
EOF

ARCHIVE_PATH="${TARGET_DIR}.tar.gz"
tar -czf "$ARCHIVE_PATH" -C "$(dirname "$TARGET_DIR")" "$(basename "$TARGET_DIR")"

cat <<EOF
Offline bundle ready:
- Staging directory: $TARGET_DIR
- Archive: $ARCHIVE_PATH

Copy the archive to the offline machine, extract it, run
'docker load' for each tarball under images/, adjust .env based on compose/.env.example,
and start the stack with 'docker compose up -d'.
EOF
