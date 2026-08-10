#!/usr/bin/env python3
"""Make DeepSeek-V4 compressed-cache slot mapping DCP-aware.

This is a companion to the DCP compressor partial-stat reduction/finalize
implementation.  It is intentionally not a standalone correctness fix: the
old fused compressor gates work on the DCP-sharded *source* slot mapping and
therefore still needs the cross-rank softmax reduction before owner-only KV
insertion.

With no arguments, patch the vLLM file in the RTX4 base image.  An explicit
path is accepted so the exact transformation can be checked against an
unpacked image or source tree.  ``--check`` validates anchors without writing.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/"
    "models/deepseek_v4/sparse_mla.py"
)

INIT_OLD = '''        assert hasattr(self.kv_cache_spec, "compress_ratio")
        self.compress_ratio = self.kv_cache_spec.compress_ratio

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
'''

INIT_NEW = '''        assert hasattr(self.kv_cache_spec, "compress_ratio")
        self.compress_ratio = self.kv_cache_spec.compress_ratio
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
        )

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
'''

CALL_OLD = '''                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
'''

CALL_NEW = '''                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(
                    self.cp_kv_cache_interleave_size
                ),
            )
'''


def _replace_exact(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {label} anchor in old vLLM source; found {count}"
        )
    return source.replace(old, new, 1)


def patch_source(source: str) -> tuple[str, bool]:
    """Return ``(patched_source, changed)`` or fail closed on source drift."""
    init_is_new = INIT_NEW in source
    call_is_new = CALL_NEW in source
    if init_is_new and call_is_new:
        return source, False
    if init_is_new != call_is_new:
        raise RuntimeError(
            "compressed-slot mapping patch is only partially applied; refusing "
            "to guess"
        )

    source = _replace_exact(source, INIT_OLD, INIT_NEW, "builder-init")
    source = _replace_exact(source, CALL_OLD, CALL_NEW, "slot-mapping call")
    return source, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"sparse_mla.py to patch (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate exact anchors/idempotency without writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.path.read_text()
    patched, changed = patch_source(source)
    if args.check:
        state = "already applied" if not changed else "ready to apply"
        print(f"DCP compressed-slot mapping companion patch: {state}")
        return
    if changed:
        args.path.write_text(patched)
        print("Applied DCP compressed-slot mapping companion patch")
    else:
        print("DCP compressed-slot mapping companion patch already applied")


if __name__ == "__main__":
    main()
