from pathlib import Path


path = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/warmup/"
    "flashinfer_sparse_mla_warmup.py"
)
source = path.read_text()
needle = "    _warmup_deepseek_v4_prefill_metadata(worker)\n"
replacement = needle + (
    "    if runner.vllm_config.parallel_config.decode_context_parallel_size > 1:\n"
    "        logger.warning(\"Skipping synthetic sparse-MLA mixed warmup for "
    "DCP > 1.\")\n"
    "        return\n"
)

if source.count(needle) != 1:
    raise RuntimeError("expected exactly one sparse-MLA metadata warmup call")

path.write_text(source.replace(needle, replacement))
