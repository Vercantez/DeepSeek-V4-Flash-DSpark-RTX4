"""Install the Python-only DeepSeek-V4 DCP PR overlay into a vLLM image."""

from pathlib import Path
import py_compile
import shutil
import sys

import vllm


FILES = (
    "model_executor/layers/attention/mla_attention.py",
    "models/deepseek_v4/attention.py",
    "models/deepseek_v4/common/ops/__init__.py",
    "models/deepseek_v4/common/ops/cache_utils.py",
    "models/deepseek_v4/common/ops/dcp.py",
    "models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
    "models/deepseek_v4/compressor.py",
    "models/deepseek_v4/nvidia/flashmla.py",
    "models/deepseek_v4/sparse_mla.py",
    "v1/attention/backends/flash_attn.py",
    "v1/attention/backends/flashinfer.py",
    "v1/attention/backends/mla/compressor_utils.py",
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/core/kv_cache_coordinator.py",
    "v1/core/kv_cache_utils.py",
    "v1/core/single_type_kv_cache_manager.py",
    "v1/kv_cache_interface.py",
    "v1/worker/cp_utils.py",
    "v1/worker/gpu_model_runner.py",
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: install_dsv4_dcp_pr.py SOURCE_CHECKOUT")

    source_root = Path(sys.argv[1]) / "vllm"
    target_root = Path(vllm.__file__).resolve().parent
    missing = [path for path in FILES if not (source_root / path).is_file()]
    if missing:
        raise RuntimeError(f"DCP PR checkout is missing files: {missing}")

    for relative_path in FILES:
        source = source_root / relative_path
        target = target_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        py_compile.compile(str(target), doraise=True)

    print(f"Installed {len(FILES)} DeepSeek-V4 DCP PR files into {target_root}")


if __name__ == "__main__":
    main()
