#!/usr/bin/env bash
set -euo pipefail

# Build a resumable immutable S3 release from an official Hugging Face commit.
# Run on a host with roughly 200 GiB free disk, AWS credentials, and Python.

: "${ARTIFACT_URI:?Set the full regional S3 release URI}"
: "${MODEL_ID:=deepseek-ai/DeepSeek-V4-Flash-0731}"
: "${MODEL_REVISION:=7872f01b1d1fe23eabc4c98b48bffcef5a386062}"
: "${WORK_DIR:=/opt/deepseek-artifact-stage}"
: "${HF_DOWNLOAD_WORKERS:=16}"

release_name=${ARTIFACT_URI%/}
release_name=${release_name##*/}
stage="$WORK_DIR/$release_name"
model_dir="$stage/hf/model"
venv="$WORK_DIR/.venv"

if aws s3 ls "$ARTIFACT_URI/artifact.json" >/dev/null 2>&1; then
  echo "Refusing to overwrite published release: $ARTIFACT_URI" >&2
  exit 1
fi

mkdir -p "$WORK_DIR" "$stage/hf"
if [ ! -x "$venv/bin/python" ]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install --upgrade pip huggingface_hub
fi

MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" MODEL_DIR="$model_dir" \
  HF_DOWNLOAD_WORKERS="$HF_DOWNLOAD_WORKERS" "$venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_DIR"],
    max_workers=int(os.environ["HF_DOWNLOAD_WORKERS"]),
)
PY

# local_dir bookkeeping is unnecessary at serve time; model files remain.
rm -rf "$model_dir/.cache"
test -f "$model_dir/config.json"
test -f "$model_dir/model.safetensors.index.json"

printf '{"release":"%s","model_id":"%s","revision":"%s","model_rel":"model"}\n' \
  "$release_name" "$MODEL_ID" "$MODEL_REVISION" >"$stage/artifact.json"
(
  cd "$stage/hf"
  find . -type f -print0 | sort -z | xargs -0 -r sha256sum >"$stage/manifest.sha256"
)
sha256sum "$stage/manifest.sha256" >"$stage/manifest.sha256.sha256"

# artifact.json is published last. Workers therefore never consume a partial
# upload as a valid release.
aws s3 sync "$stage/hf/" "$ARTIFACT_URI/hf/" --only-show-errors
aws s3 cp "$stage/manifest.sha256" "$ARTIFACT_URI/manifest.sha256" --only-show-errors
aws s3 cp "$stage/manifest.sha256.sha256" "$ARTIFACT_URI/manifest.sha256.sha256" --only-show-errors
aws s3 cp "$stage/artifact.json" "$ARTIFACT_URI/artifact.json" --only-show-errors

echo "Published $ARTIFACT_URI"
