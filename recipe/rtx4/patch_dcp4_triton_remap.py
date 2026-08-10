"""Allow the DCP global-top-k remap to use its PyTorch fallback.

The Triton candidate-pack/remap kernels can spin indefinitely on one rank on
the 4x RTX PRO 6000 Blackwell path.  The implementation already contains an
equivalent PyTorch path; expose a narrow environment switch so production can
prefer correctness and liveness while retaining TP4+DCP4.
"""

from pathlib import Path


INDEXER_PATH = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/"
    "model_executor/layers/sparse_attn_indexer.py"
)

source = INDEXER_PATH.read_text()
needle = """def _use_triton_dcp_remap(topk_indices: torch.Tensor) -> bool:
    return HAS_TRITON and current_platform.is_cuda() and topk_indices.is_cuda
"""
replacement = """def _use_triton_dcp_remap(topk_indices: torch.Tensor) -> bool:
    enabled = os.environ.get(\"VLLM_DCP_TRITON_REMAP\", \"1\").lower() in (
        \"1\",
        \"true\",
        \"yes\",
        \"on\",
    )
    return (
        enabled
        and HAS_TRITON
        and current_platform.is_cuda()
        and topk_indices.is_cuda
    )
"""

if replacement in source:
    print("DCP Triton-remap switch already applied")
elif source.count(needle) != 1:
    raise RuntimeError("expected exactly one DCP Triton-remap predicate")
else:
    INDEXER_PATH.write_text(source.replace(needle, replacement, 1))
    print("Applied DCP Triton-remap switch")
