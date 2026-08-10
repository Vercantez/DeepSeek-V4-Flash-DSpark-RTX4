from pathlib import Path


mla_warmup_path = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/warmup/"
    "flashinfer_sparse_mla_warmup.py"
)
source = mla_warmup_path.read_text()
needle = "    _warmup_deepseek_v4_prefill_metadata(worker)\n"
replacement = needle + (
    "    if runner.vllm_config.parallel_config.decode_context_parallel_size > 1:\n"
    "        logger.warning(\"Skipping synthetic sparse-MLA mixed warmup for "
    "DCP > 1.\")\n"
    "        return\n"
)

if source.count(needle) != 1:
    raise RuntimeError("expected exactly one sparse-MLA metadata warmup call")

mla_warmup_path.write_text(source.replace(needle, replacement))

gpu_worker_path = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py"
)
source = gpu_worker_path.read_text()
needle = (
    "        if self.use_v2_model_runner:\n"
    "            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.\n"
    "            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)\n"
)
replacement = (
    "        if self.use_v2_model_runner:\n"
    "            if self.vllm_config.parallel_config.decode_context_parallel_size > 1:\n"
    "                logger.warning(\"Skipping synthetic V2 post-capture warmup for "
    "DCP > 1.\")\n"
    "            else:\n"
    "                # Run full execute_model + sample_tokens to JIT compile kernels.\n"
    "                warmup_kernels(\n"
    "                    self.model_runner, self.execute_model, self.sample_tokens\n"
    "                )\n"
)

if source.count(needle) != 1:
    raise RuntimeError("expected exactly one V2 post-capture warmup call")

gpu_worker_path.write_text(source.replace(needle, replacement))
