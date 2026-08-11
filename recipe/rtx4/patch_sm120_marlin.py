#!/usr/bin/env python3
"""Fail-closed vLLM 0.25.1 Marlin MoE SM120 source patcher."""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = (
    (
        """  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  STD_TORCH_CHECK(max_shared_mem > 0);
""",
        """  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  STD_TORCH_CHECK(max_shared_mem > 0);
  int device_max_shared_mem = max_shared_mem;
""",
    ),
    (
        """  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
""",
        """  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       device_max_shared_mem);
""",
    ),
    (
        "kernel<<<blocks, num_threads, max_shared_mem, stream>>>(",
        "kernel<<<blocks, num_threads, sh_cache_size, stream>>>(",
    ),
    (
        """    long max_c_tmp_size = min(
        (long)size_n * sorted_token_ids.size(0),
        (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
""",
        """    long max_c_tmp_size =
        (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
""",
    ),
)


def source_state(source: str) -> str:
    states: list[str] = []
    for old, new in REPLACEMENTS:
        old_count = source.count(old)
        new_count = source.count(new)
        if old_count == 1 and new_count == 0:
            states.append("original")
        elif new_count == 1 and old_count == new.count(old):
            states.append("patched")
        else:
            raise RuntimeError(
                "Marlin source drift: expected exactly one original or patched "
                f"anchor, got original={old_count} patched={new_count}"
            )
    if len(set(states)) != 1:
        raise RuntimeError(f"partially applied Marlin patch: {states}")
    return states[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = args.source.read_text()
    state = source_state(source)
    if args.check:
        print(state)
        return
    if state == "original":
        for old, new in REPLACEMENTS:
            source = source.replace(old, new, 1)
        args.source.write_text(source)
        assert source_state(source) == "patched"
        print("patched")
    else:
        print("already patched")


if __name__ == "__main__":
    main()
