#!/usr/bin/env bash
set -euo pipefail

# Publish one immutable archive of compiled artifacts from a fully warmed
# worker. Run this on the worker after /v1/models reports ready.

: "${ARTIFACT_URI:?Set the full regional S3 release URI}"
: "${RUNTIME_CACHE_OBJECT:?Set a unique object name, for example runtime-cache/0731-sm120-v1.tar.zst}"
: "${HF_CACHE:=/opt/dlami/nvme/deepseek-model/hf}"
: "${VLLM_CACHE_DIR:=$HF_CACHE/vllm-cache-stock-sm120-v0.25.1-fi0.6.14}"

case "$VLLM_CACHE_DIR" in
  "$HF_CACHE"/*) cache_rel=${VLLM_CACHE_DIR#"$HF_CACHE"/} ;;
  *)
    echo "VLLM_CACHE_DIR must be below HF_CACHE" >&2
    exit 2
    ;;
esac
test -d "$VLLM_CACHE_DIR"
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models >/dev/null

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
archive="$work_dir/$(basename "$RUNTIME_CACHE_OBJECT")"

tar --zstd -cf "$archive" -C "$HF_CACHE" \
  --exclude="$cache_rel/tmp" \
  "$cache_rel"
(
  cd "$work_dir"
  sha256sum "$(basename "$archive")" >"$(basename "$archive").sha256"
)

if aws s3 ls "$ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" >/dev/null 2>&1; then
  echo "Refusing to overwrite runtime cache: $ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" >&2
  exit 1
fi

aws s3 cp "$archive" "$ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" --only-show-errors
aws s3 cp "$archive.sha256" "$ARTIFACT_URI/$RUNTIME_CACHE_OBJECT.sha256" --only-show-errors
echo "Published $ARTIFACT_URI/$RUNTIME_CACHE_OBJECT"
