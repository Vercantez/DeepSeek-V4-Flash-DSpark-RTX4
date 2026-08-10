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
: "${SERVED_MODEL_NAME:=deepseek-v4-flash-dspark}"
: "${DSPARK_VLLM_IMAGE:=vllm-dspark-runtime:rtx4-nvfp4-port-v3}"
: "${CONTAINER_NAME:=deepseek-v4-flash-dspark-rtx4}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"
: "${VLLM_HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${GPUS:=0,1,2,3}"
: "${TP_SIZE:=4}"
: "${BACKEND:=lucifer-cutlass}"
: "${KV_CACHE_DTYPE:=fp8_ds_mla}"
: "${MAX_MODEL_LEN:=}"
: "${MAX_NUM_SEQS:=64}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${CUDA_GRAPH_CAPTURE_SIZE:=512}"
: "${DSPARK_NUM_TOKENS:=${MTP_NUM_TOKENS:-5}}"
: "${DSPARK_SAMPLE:=greedy}"
: "${DSPARK_DRAFT_MODEL:=}"
: "${PREFIX_CACHE:=1}"
: "${SCHEDULING_POLICY:=priority}"
: "${PULL_IMAGE:=0}"
: "${KV_OFFLOAD_GB:=}"
: "${KV_OFFLOAD_DISK_DIR:=}"

mkdir -p \
  "$HF_CACHE" \
  "$HF_CACHE/vllm-cache" \
  "$HF_CACHE/vllm-cache/tilelang/tmp" \
  "$HF_CACHE/vllm-cache/triton" \
  "$HF_CACHE/vllm-cache/torchinductor" \
  "$HF_CACHE/vllm-cache/torch_extensions" \
  "$HF_CACHE/vllm-cache/flashinfer" \
  "$HF_CACHE/vllm-cache/tmp"

# S3/NVMe workers stage a Hugging Face snapshot under the cache mount. A stale
# cache-root MODEL_DIR is not a model directory, so resolve the snapshot before
# handing the path to vLLM.
if [[ "${MODEL_DIR:-}" == /cache/huggingface/* ]] && \
  [ ! -f "$HF_CACHE/${MODEL_DIR#/cache/huggingface/}/config.json" ]; then
  staged_snapshot=$(find "$HF_CACHE/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots" \
    -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/config.json' \; -print -quit 2>/dev/null || true)
  if [ -n "$staged_snapshot" ]; then
    MODEL_DIR="/cache/huggingface${staged_snapshot#"$HF_CACHE"}"
  fi
fi

MODEL_ARG="${MODEL_DIR:-$DSPARK_MODEL}"
# Flash-0731 stores the DSpark drafter in the target checkpoint. The older
# preview package used a separate draft model, which remains opt-in only for
# historical checkpoints.
SPECULATIVE_CONFIG="$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "$DSPARK_NUM_TOKENS" "$DSPARK_SAMPLE")"
if [ -n "$DSPARK_DRAFT_MODEL" ]; then
  SPECULATIVE_CONFIG="$(printf '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "$DSPARK_DRAFT_MODEL" "$DSPARK_NUM_TOKENS" "$DSPARK_SAMPLE")"
fi
GENERATION_CONFIG_JSON="$(printf '{"temperature":%s,"top_p":%s,"top_k":%s,"repetition_penalty":%s,"max_tokens":%s}' \
  "${GENERATION_TEMPERATURE:-0.0}" \
  "${GENERATION_TOP_P:-1.0}" \
  "${GENERATION_TOP_K:-40}" \
  "${GENERATION_REPETITION_PENALTY:-1.05}" \
  "${GENERATION_MAX_TOKENS:-384000}")"

MODEL_LEN_ARGS=()
if [ -n "$MAX_MODEL_LEN" ]; then
  MODEL_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

BACKEND_ARGS=()
BACKEND_ENV=()
case "$BACKEND" in
  b12x)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
    BACKEND_ENV=(
      -e VLLM_USE_B12X_WO_PROJECTION=1
      -e VLLM_USE_B12X_MHC=1
      -e VLLM_USE_B12X_FP8_GEMM=1
      -e VLLM_USE_B12X_MOE=1
      -e VLLM_USE_B12X_SPARSE_INDEXER=1
      -e VLLM_ENABLE_PCIE_ALLREDUCE=1
      -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
      -e B12X_MLA_SM120_UNIFIED=1
      -e B12X_MHC_MAX_TOKENS=16384
      -e B12X_DENSE_SPLITK_TURBO=1
      -e B12X_W4A16_TC_DECODE=1
      -e B12X_MOE_FORCE_A16=1
    )
    ;;
  lucifer-cutlass)
    BACKEND_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend flashinfer_cutlass --disable-custom-all-reduce)
    BACKEND_ENV=(
      -e VLLM_ENABLE_PCIE_ALLREDUCE=0
      -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp
    )
    ;;
  lucifer-default)
    BACKEND_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --disable-custom-all-reduce)
    BACKEND_ENV=(
      -e VLLM_ENABLE_PCIE_ALLREDUCE=0
      -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp
    )
    ;;
  *)
    echo "Unknown BACKEND=$BACKEND" >&2
    exit 2
    ;;
esac

case "$SCHEDULING_POLICY" in
  fcfs|priority) ;;
  *)
    echo "SCHEDULING_POLICY must be fcfs or priority, got $SCHEDULING_POLICY" >&2
    exit 2
    ;;
esac

PREFIX_ARGS=(--enable-prefix-caching)
if [ "$PREFIX_CACHE" != "1" ]; then
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

KV_OFFLOAD_ARGS=()
KV_OFFLOAD_ENV=()
KV_OFFLOAD_MOUNTS=()
CUDA_ALLOCATOR_ARGS=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
SHM_SIZE=64g
if [ -n "$KV_OFFLOAD_GB" ]; then
  if [ -z "$KV_OFFLOAD_DISK_DIR" ]; then
    echo "KV_OFFLOAD_DISK_DIR is required when KV_OFFLOAD_GB is set" >&2
    exit 2
  fi
  mkdir -p "$KV_OFFLOAD_DISK_DIR"
  KV_OFFLOAD_MOUNTS=(-v "${KV_OFFLOAD_DISK_DIR}:${KV_OFFLOAD_DISK_DIR}")
  KV_TRANSFER_CONFIG="$(printf '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":%s,\"eviction_policy\":\"lru\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"%s\",\"n_read_threads\":32,\"n_write_threads\":16}]}}' "$((KV_OFFLOAD_GB * 1024 * 1024 * 1024))" "$KV_OFFLOAD_DISK_DIR")"
  KV_OFFLOAD_ARGS=(
    --kv-offloading-size "$KV_OFFLOAD_GB"
    --kv-offloading-backend native
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
  )
  # OffloadingConnector pins KV allocations, which is incompatible with
  # PyTorch CUDA VMM's expandable segments allocator.
  CUDA_ALLOCATOR_ARGS=()
  KV_OFFLOAD_ENV=(-e PYTHONHASHSEED=0)
  SHM_SIZE="$((KV_OFFLOAD_GB + 1))g"
fi

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
  --shm-size "$SHM_SIZE" \
  --network host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v "${HF_CACHE}:/cache/huggingface" \
  "${KV_OFFLOAD_MOUNTS[@]}" \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  "${CUDA_ALLOCATOR_ARGS[@]}" \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  -e HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
  -e TMPDIR=/cache/huggingface/vllm-cache/tmp \
  -e XDG_CACHE_HOME=/cache/huggingface/vllm-cache \
  -e VLLM_CACHE_DIR=/cache/huggingface/vllm-cache/vllm \
  -e TILELANG_CACHE_DIR=/cache/huggingface/vllm-cache/tilelang \
  -e TILELANG_TMP_DIR=/cache/huggingface/vllm-cache/tilelang/tmp \
  -e TRITON_CACHE_DIR=/cache/huggingface/vllm-cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/huggingface/vllm-cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/huggingface/vllm-cache/torch_extensions \
  -e FLASHINFER_WORKSPACE_BASE=/cache/huggingface/vllm-cache/flashinfer \
  "${KV_OFFLOAD_ENV[@]}" \
  "${BACKEND_ENV[@]}" \
  "$DSPARK_VLLM_IMAGE" \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve "$@"' \
  -- "$MODEL_ARG" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$VLLM_HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --block-size 256 \
  --load-format auto \
  --tensor-parallel-size "$TP_SIZE" \
  --decode-context-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${MODEL_LEN_ARGS[@]}" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --scheduling-policy "$SCHEDULING_POLICY" \
  --max-cudagraph-capture-size "$CUDA_GRAPH_CAPTURE_SIZE" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs.thinking=false \
  --generation-config vllm \
  --override-generation-config "$GENERATION_CONFIG_JSON" \
  --speculative-config "$SPECULATIVE_CONFIG" \
  "${KV_OFFLOAD_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${PREFIX_ARGS[@]}"

echo "$CONTAINER_NAME $SERVED_MODEL_NAME $BACKEND TP=$TP_SIZE GPUS=$GPUS PORT=$PORT KV=$KV_CACHE_DTYPE SCHEDULER=$SCHEDULING_POLICY"
