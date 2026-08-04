#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ARTIFACT_URI:?Set the regional S3 release URI}"

systemctl stop deepseek-rtx4.service 2>/dev/null || true

NVME_ROOT=${NVME_ROOT:-/opt/dlami/nvme/deepseek-model}
HF_CACHE="$NVME_ROOT/hf"
ARTIFACT_DIR="$NVME_ROOT/artifact"
MANIFEST="$ARTIFACT_DIR/manifest.sha256"
VERIFY_JOBS=${VERIFY_JOBS:-8}
RUNTIME_CACHE_OBJECT=${RUNTIME_CACHE_OBJECT:-}

if ! command -v aws >/dev/null; then
  apt-get update -y
  apt-get install -y curl unzip
  curl --fail --location --silent --show-error \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip \
    --output /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install --update
fi

mkdir -p "$HF_CACHE" "$ARTIFACT_DIR"
chown -R ubuntu:ubuntu "$NVME_ROOT"

# The cache is immutable per release. High concurrency matters because the
# model is stored as independent safetensor shards rather than one archive.
aws configure set default.s3.max_concurrent_requests 64
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.preferred_transfer_client crt
aws configure set default.s3.target_bandwidth 40Gb/s

aws s3 cp "$MODEL_ARTIFACT_URI/artifact.json" "$ARTIFACT_DIR/artifact.json" --only-show-errors
aws s3 cp "$MODEL_ARTIFACT_URI/manifest.sha256" "$MANIFEST" --only-show-errors
aws s3 cp "$MODEL_ARTIFACT_URI/manifest.sha256.sha256" "$MANIFEST.sha256" --only-show-errors

cd "$ARTIFACT_DIR"
expected_manifest_hash=$(awk '{print $1}' "$MANIFEST.sha256")
actual_manifest_hash=$(sha256sum "$MANIFEST" | awk '{print $1}')
test "$actual_manifest_hash" = "$expected_manifest_hash"

model_rel=$(sed -n 's/.*"model_rel":"\([^"]*\)".*/\1/p' "$ARTIFACT_DIR/artifact.json")
if [ -z "$model_rel" ] || [ "$model_rel" = null ]; then
  echo "artifact.json is missing a valid model_rel" >&2
  exit 1
fi

# Keep vLLM's runtime cache out of the immutable artifact sync. It can contain
# Unix sockets while the previous engine is stopped, which makes a broad sync
# return a nonzero status even after all model files have copied.
AWS_CRT_S3_MEMORY_LIMIT_IN_GIB=8 \
  aws s3 sync "$MODEL_ARTIFACT_URI/hf/$model_rel/" "$HF_CACHE/$model_rel/" \
    --exclude 'vllm-cache/*' --only-show-errors

# Compiled SM120 artifacts are release- and runtime-specific. Restore the
# explicitly selected immutable archive before vLLM starts; a missing or
# corrupt configured archive is a bootstrap failure rather than a slow,
# silently recompiling worker.
if [ -n "$RUNTIME_CACHE_OBJECT" ]; then
  runtime_cache_archive="$ARTIFACT_DIR/$(basename "$RUNTIME_CACHE_OBJECT")"
  aws s3 cp "$MODEL_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" "$runtime_cache_archive" --only-show-errors
  aws s3 cp "$MODEL_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT.sha256" "$runtime_cache_archive.sha256" --only-show-errors
  (
    cd "$ARTIFACT_DIR"
    sha256sum --quiet -c "$(basename "$runtime_cache_archive").sha256"
  )
  tar --zstd -xf "$runtime_cache_archive" -C "$HF_CACHE"
fi
chown -R ubuntu:ubuntu "$HF_CACHE"

cd "$HF_CACHE"
verify_dir=$(mktemp -d "$ARTIFACT_DIR/manifest-parts.XXXXXX")
trap 'rm -rf "$verify_dir"' EXIT
split --number="l/$VERIFY_JOBS" --additional-suffix=.sha256 "$MANIFEST" "$verify_dir/part-"
find "$verify_dir" -type f -print0 | xargs -0 -r -n 1 -P "$VERIFY_JOBS" sha256sum --quiet -c
rm -rf "$verify_dir"
trap - EXIT

model_dir_rel="$model_rel"
if [ ! -f "$HF_CACHE/$model_dir_rel/config.json" ]; then
  model_dir_rel=$(find "$HF_CACHE/$model_rel" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/config.json' \; -print -quit | sed "s#^$HF_CACHE/##")
fi
test -n "$model_dir_rel" && test -f "$HF_CACHE/$model_dir_rel/config.json"

repo=/opt/deepseek/mia-dspark-rtx4
env_file="$repo/.env.rtx4"
sudo -u ubuntu git -C "$repo" fetch --all --prune
sudo -u ubuntu git -C "$repo" checkout main
sudo -u ubuntu git -C "$repo" pull --ff-only

# Begin watching IMDS before vLLM accepts traffic. On a Spot interruption or
# rebalance recommendation, the watcher rejects new connections while allowing
# established streams to use the remaining notice window.
"$repo/infra/aws/install-spot-interruption-watcher.sh"

sed -i '/^HF_CACHE=/d; /^MODEL_DIR=/d; /^KV_CACHE_DTYPE=/d; /^KV_OFFLOAD_GB=/d; /^KV_OFFLOAD_DISK_DIR=/d; /^DSPARK_MODEL=/d; /^DSPARK_DRAFT_MODEL=/d; /^MTP_NUM_TOKENS=/d; /^DSPARK_NUM_TOKENS=/d; /^DSPARK_SAMPLE=/d' "$env_file"
printf '%s\n' \
  "HF_CACHE=$HF_CACHE" \
  "MODEL_DIR=/cache/huggingface/$model_dir_rel" \
  "KV_CACHE_DTYPE=fp8_ds_mla" \
  "DSPARK_NUM_TOKENS=5" \
  "DSPARK_SAMPLE=greedy" >>"$env_file"

# Give new RTX4 workers a durable overflow tier for long-context sessions. The
# primary offload tier is host memory; local NVMe is used after it fills.
: "${KV_OFFLOAD_GB:=256}"
: "${KV_OFFLOAD_DISK_DIR:=/opt/dlami/nvme/kv-offload}"
if [ "$KV_OFFLOAD_GB" -gt 0 ]; then
  printf '%s\n' \
    "KV_OFFLOAD_GB=$KV_OFFLOAD_GB" \
    "KV_OFFLOAD_DISK_DIR=$KV_OFFLOAD_DISK_DIR" >>"$env_file"
  install -d -o ubuntu -g ubuntu "$KV_OFFLOAD_DISK_DIR"
fi

install -d /etc/systemd/system/deepseek-rtx4.service.d
cat >/etc/systemd/system/deepseek-rtx4.service.d/cleanup-offload.conf <<'EOF'
[Service]
# vLLM's host-IPC offload mmap can outlive an interrupted container.
ExecStartPre=+/bin/sh -c 'rm -f /dev/shm/vllm_offload_*.mmap'
EOF

systemctl daemon-reload
systemctl enable deepseek-rtx4.service
systemctl start deepseek-rtx4.service
