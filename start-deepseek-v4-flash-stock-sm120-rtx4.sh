#!/usr/bin/env bash
set -euo pipefail

# Isolated stock-vLLM profile for Hermia's validated 4x RTX PRO 6000 SM120
# recipe. Keep this separate from start-deepseek-v4-flash-dspark-rtx4.sh: the
# production launcher intentionally carries custom B12X/DCP/DSpark knobs that
# are neither required nor validated on the stock path.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.rtx4.stock-sm120}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${DSPARK_MODEL:=deepseek-ai/DeepSeek-V4-Flash-DSpark}"
: "${SERVED_MODEL_NAME:=deepseek-v4-flash-dspark}"
: "${DSPARK_VLLM_IMAGE:=vllm-dspark-runtime:stock-sm120-vllm-0.25.1}"
: "${CONTAINER_NAME:=deepseek-v4-flash-stock-sm120-rtx4}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"
: "${VLLM_CACHE_DIR:=$HF_CACHE/vllm-cache-stock-sm120-v0.25.1-fi0.6.14}"
: "${VLLM_HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${GPUS:=0,1,2,3}"
: "${TP_SIZE:=4}"
: "${DCP_SIZE:=1}"
: "${DSPARK_NUM_TOKENS:=0}"
: "${KV_CACHE_DTYPE:=fp8}"
: "${MAX_MODEL_LEN:=262144}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${KV_OFFLOAD_GB:=256}"
: "${KV_OFFLOAD_DISK_DIR:=/opt/dlami/nvme/kv-offload}"
: "${KV_OFFLOAD_REQUIRED_MOUNT:=/opt/dlami/nvme/kv-offload}"
: "${KV_OFFLOAD_MIN_FREE_GB:=1024}"
: "${KV_OFFLOAD_READ_THREADS:=32}"
: "${KV_OFFLOAD_WRITE_THREADS:=16}"
: "${PULL_IMAGE:=0}"

if [ "$TP_SIZE" != "4" ]; then
  echo "stock-sm120 is validated only with TP_SIZE=4, got $TP_SIZE" >&2
  exit 2
fi
if [ "$DCP_SIZE" != "1" ]; then
  echo "stock-sm120 does not support DCP; set DCP_SIZE=1" >&2
  exit 2
fi
if [ "$DSPARK_NUM_TOKENS" != "0" ]; then
  echo "stock-sm120 does not support DSpark/speculative decode; set DSPARK_NUM_TOKENS=0" >&2
  exit 2
fi
if [ "$KV_CACHE_DTYPE" != "fp8" ]; then
  echo "stock-sm120 requires KV_CACHE_DTYPE=fp8, got $KV_CACHE_DTYPE" >&2
  exit 2
fi
if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || (( MAX_MODEL_LEN > 262144 )); then
  echo "stock-sm120 requires MAX_MODEL_LEN to be an integer at or below 262144, got $MAX_MODEL_LEN" >&2
  exit 2
fi
if ! [[ "$KV_OFFLOAD_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "stock-sm120 requires KV_OFFLOAD_GB to be a positive integer" >&2
  exit 2
fi
if ! [[ "$KV_OFFLOAD_MIN_FREE_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "stock-sm120 requires KV_OFFLOAD_MIN_FREE_GB to be a positive integer" >&2
  exit 2
fi
if ! [[ "$KV_OFFLOAD_READ_THREADS" =~ ^[1-9][0-9]*$ ]] ||
  ! [[ "$KV_OFFLOAD_WRITE_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "stock-sm120 requires positive KV offload thread counts" >&2
  exit 2
fi
if [ "${KV_OFFLOAD_REQUIRED_MOUNT#/}" = "$KV_OFFLOAD_REQUIRED_MOUNT" ] ||
  [ "${KV_OFFLOAD_DISK_DIR#/}" = "$KV_OFFLOAD_DISK_DIR" ]; then
  echo "stock-sm120 requires absolute KV offload mount and directory paths" >&2
  exit 2
fi
case "$KV_OFFLOAD_DISK_DIR/" in
  "$KV_OFFLOAD_REQUIRED_MOUNT"/*) ;;
  *)
    echo "KV offload directory must be below $KV_OFFLOAD_REQUIRED_MOUNT" >&2
    exit 2
    ;;
esac
case "$KV_OFFLOAD_DISK_DIR" in
  *[!A-Za-z0-9_./-]*)
    echo "KV offload directory contains unsupported characters" >&2
    exit 2
    ;;
esac

mkdir -p "$KV_OFFLOAD_DISK_DIR"
KV_OFFLOAD_ACTUAL_MOUNT="$(findmnt -n -o TARGET -T "$KV_OFFLOAD_DISK_DIR")"
if [ "$KV_OFFLOAD_ACTUAL_MOUNT" != "$KV_OFFLOAD_REQUIRED_MOUNT" ]; then
  echo "Refusing KV offload: $KV_OFFLOAD_DISK_DIR resolves to $KV_OFFLOAD_ACTUAL_MOUNT, expected $KV_OFFLOAD_REQUIRED_MOUNT" >&2
  exit 2
fi
KV_OFFLOAD_AVAILABLE_KB="$(df -Pk "$KV_OFFLOAD_DISK_DIR" | awk 'NR == 2 {print $4}')"
if ! [[ "$KV_OFFLOAD_AVAILABLE_KB" =~ ^[0-9]+$ ]] ||
  (( KV_OFFLOAD_AVAILABLE_KB < KV_OFFLOAD_MIN_FREE_GB * 1024 * 1024 )); then
  echo "Refusing KV offload: $KV_OFFLOAD_DISK_DIR has less than ${KV_OFFLOAD_MIN_FREE_GB} GiB free" >&2
  exit 2
fi

KV_TRANSFER_CONFIG="$(printf '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":%s,\"eviction_policy\":\"lru\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"%s\",\"n_read_threads\":%s,\"n_write_threads\":%s}]}}' "$((KV_OFFLOAD_GB * 1024 * 1024 * 1024))" "$KV_OFFLOAD_DISK_DIR" "$KV_OFFLOAD_READ_THREADS" "$KV_OFFLOAD_WRITE_THREADS")"

mkdir -p "$HF_CACHE" "$VLLM_CACHE_DIR"
MODEL_ARG="${MODEL_DIR:-$DSPARK_MODEL}"

if [ "$PULL_IMAGE" = "1" ]; then
  docker pull "$DSPARK_VLLM_IMAGE"
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Failed to remove existing container: $CONTAINER_NAME" >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --runtime nvidia \
  --ipc host \
  --network host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v "${HF_CACHE}:/cache/huggingface" \
  -v "${VLLM_CACHE_DIR}:/root/.cache" \
  -v "${KV_OFFLOAD_DISK_DIR}:${KV_OFFLOAD_DISK_DIR}" \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_HOME=/usr/local/cuda \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_P2P_DISABLE=1 \
  -e PYTHONHASHSEED=0 \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
  --entrypoint /bin/bash \
  "$DSPARK_VLLM_IMAGE" \
  -lc 'export PATH="$CUDA_HOME/bin:$PATH"; exec vllm serve "$@"' \
  -- "$MODEL_ARG" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$VLLM_HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --block-size 256 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-offloading-size "$KV_OFFLOAD_GB" \
  --kv-offloading-backend native \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kernel-config '{"moe_backend":"marlin"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --trust-remote-code

echo "$CONTAINER_NAME $SERVED_MODEL_NAME stock-sm120 TP=$TP_SIZE EP=1 GPUS=$GPUS PORT=$PORT KV=$KV_CACHE_DTYPE MAX_MODEL_LEN=$MAX_MODEL_LEN KV_OFFLOAD=${KV_OFFLOAD_GB}GiB+fs:$KV_OFFLOAD_DISK_DIR"
