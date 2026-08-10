"""Add the missing B12X DCP global-top-k scratch-buffer allocation.

This is a minimal port of local-inference-lab/vllm PR #72 commit 6785ad5.
It intentionally changes only the fallback for callers that supply an index
buffer but omit the matching score buffer.
"""

from pathlib import Path


INDEXER_PATH = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/"
    "model_executor/layers/sparse_attn_indexer.py"
)

source = INDEXER_PATH.read_text()
needle = """        self.topk_indices_buffer = topk_indices_buffer
        self.topk_scores_buffer = topk_scores_buffer
        self.output_physical_slots = bool(output_physical_slots)
"""
replacement = """        self.topk_indices_buffer = topk_indices_buffer
        if topk_scores_buffer is None and topk_indices_buffer is not None:
            try:
                from vllm.distributed.parallel_state import get_dcp_group

                dcp_world_size = get_dcp_group().world_size
            except Exception:
                dcp_world_size = 1
            if dcp_world_size > 1 and use_b12x_sparse_indexer():
                topk_scores_buffer = torch.empty(
                    topk_indices_buffer.shape,
                    dtype=torch.float32,
                    device=topk_indices_buffer.device,
                )
        self.topk_scores_buffer = topk_scores_buffer
        self.output_physical_slots = bool(output_physical_slots)
"""

if replacement in source:
    print("B12X DCP top-k score-buffer patch already applied")
elif source.count(needle) != 1:
    raise RuntimeError("expected exactly one sparse-indexer buffer assignment")
else:
    INDEXER_PATH.write_text(source.replace(needle, replacement, 1))
    print("Applied B12X DCP top-k score-buffer patch")
