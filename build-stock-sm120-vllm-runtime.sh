#!/usr/bin/env bash
set -euo pipefail

# Build the pinned stock SM120 runtime from vLLM source so the Marlin MoE
# kernel fix is compiled into the extension. A derived image cannot patch the
# precompiled _C extension in vllm/vllm-openai, so this script deliberately
# rebuilds the exact v0.25.1 source commit before applying our small overlay.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VLLM_REPO_URL=${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}
VLLM_SOURCE_COMMIT=${VLLM_SOURCE_COMMIT:-752a3a504485790a2e8491cacbb35c137339ad34}
PATCHER=${PATCHER:-$SCRIPT_DIR/recipe/rtx4/patch_sm120_marlin.py}
PATCHED_BASE_IMAGE=${PATCHED_BASE_IMAGE:-vllm-dspark-runtime:vllm-0.25.1-marlin-c-tmp-v1-base}
FINAL_IMAGE=${FINAL_IMAGE:-vllm-dspark-runtime:stock-sm120-vllm-0.25.1-marlin-c-tmp-v1}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
MAX_JOBS=${MAX_JOBS:-32}
NVCC_THREADS=${NVCC_THREADS:-4}

if [ ! -f "$PATCHER" ]; then
  echo "missing Marlin patcher: $PATCHER" >&2
  exit 1
fi

build_root=$(mktemp -d "${TMPDIR:-/tmp}/vllm-sm120-marlin.XXXXXX")
cleanup() {
  rm -rf "$build_root"
}
trap cleanup EXIT

git -C "$build_root" init --quiet
git -C "$build_root" remote add origin "$VLLM_REPO_URL"
git -C "$build_root" fetch --quiet --depth 1 origin "$VLLM_SOURCE_COMMIT"
git -C "$build_root" checkout --quiet --detach FETCH_HEAD

actual_commit=$(git -C "$build_root" rev-parse HEAD)
if [ "$actual_commit" != "$VLLM_SOURCE_COMMIT" ]; then
  echo "vLLM source drift: expected $VLLM_SOURCE_COMMIT, got $actual_commit" >&2
  exit 1
fi

source_file="$build_root/csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu"
python3 "$PATCHER" "$source_file" --check
python3 "$PATCHER" "$source_file"
test "$(python3 "$PATCHER" "$source_file" --check)" = patched
grep -Fq 'long max_c_tmp_size =' "$source_file"
grep -Fq '(long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;' "$source_file"
if grep -A3 -F 'long max_c_tmp_size =' "$source_file" | grep -Fq 'min('; then
  echo "Marlin c_tmp clamp is still present after patching" >&2
  exit 1
fi
grep -Fq 'kernel<<<blocks, num_threads, sh_cache_size, stream>>>' "$source_file"

docker build --pull=false \
  --target vllm-openai \
  --build-arg "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST" \
  --build-arg "max_jobs=$MAX_JOBS" \
  --build-arg "nvcc_threads=$NVCC_THREADS" \
  --label "ai.camel.vllm.source_commit=$VLLM_SOURCE_COMMIT" \
  --label "ai.camel.vllm.marlin_fix=c-tmp-v1" \
  -f "$build_root/docker/Dockerfile" \
  -t "$PATCHED_BASE_IMAGE" \
  "$build_root"

docker build --pull=false \
  --build-arg "BASE_IMAGE=$PATCHED_BASE_IMAGE" \
  --build-arg "VLLM_SOURCE_COMMIT=$VLLM_SOURCE_COMMIT" \
  --build-arg MARLIN_FIX=c-tmp-v1 \
  -f "$SCRIPT_DIR/recipe/rtx4/Dockerfile.stock-sm120" \
  -t "$FINAL_IMAGE" \
  "$SCRIPT_DIR"

docker run --rm --entrypoint python3 "$FINAL_IMAGE" -c \
  'from importlib.metadata import version; assert version("vllm") == "0.25.1"; assert version("flashinfer-python") == "0.6.14"; assert version("flashinfer-cubin") == "0.6.13"'

echo "built $FINAL_IMAGE from vLLM $VLLM_SOURCE_COMMIT with Marlin c_tmp fix"
