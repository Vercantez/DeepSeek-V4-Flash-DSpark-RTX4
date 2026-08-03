#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.rtx4}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${DSPARK_MODEL:=deepseek-ai/DeepSeek-V4-Flash-0731}"
: "${DSPARK_VLLM_IMAGE:=vllm-dspark-runtime:rtx4-nvfp4-port-v3}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"
: "${HF_DOWNLOAD_WORKERS:=8}"

mkdir -p "$HF_CACHE"
: "${PULL_IMAGE:=0}"

if [ "$PULL_IMAGE" = "1" ]; then
  docker pull "$DSPARK_VLLM_IMAGE"
fi

docker run --rm -i \
  -v "${HF_CACHE}:/cache/huggingface" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  -e HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
  -e DSPARK_MODEL="$DSPARK_MODEL" \
  -e HF_DOWNLOAD_WORKERS="$HF_DOWNLOAD_WORKERS" \
  --entrypoint /opt/venv/bin/python \
  "$DSPARK_VLLM_IMAGE" \
  - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model = os.environ["DSPARK_MODEL"]
path = Path(snapshot_download(
    model,
    max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
))
index_path = path / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
needed = sorted(set(index["weight_map"].values()))
missing = [name for name in needed if not (path / name).exists()]
print(f"snapshot={path}")
print(f"safetensor_shards={len(needed)}")
print(f"missing_shards={len(missing)}")
if missing:
    for name in missing[:20]:
        print(f"missing {name}")
    raise SystemExit(1)
PY
