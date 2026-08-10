#!/usr/bin/env python3
"""Port exact cross-rank compressor reduction to the pinned B12X runtime.

The 45c1582-era B12X implementation shards compressor state under DCP but
normalizes each rank's partial window independently. This patch preserves its
attention and sparse-indexer backends while replacing only the compressor's
DCP>1 path with rank-local softmax statistics, one exact all-gather merge, and
owner-local compressed-cache writes.

The edit is fail-closed against exact 45c1582 source anchors and idempotent.
Set VLLM_PACKAGE_ROOT only for local fixture testing.
"""

from __future__ import annotations

import os
from pathlib import Path
import py_compile
import shutil


PACKAGE_ROOT = Path(
    os.environ.get(
        "VLLM_PACKAGE_ROOT",
        "/opt/venv/lib/python3.12/site-packages/vllm",
    )
)
COMPRESSOR = PACKAGE_ROOT / "models/deepseek_v4/compressor.py"
KV_INTERFACE = PACKAGE_ROOT / "v1/kv_cache_interface.py"
TARGET_KERNELS = (
    PACKAGE_ROOT / "models/deepseek_v4/common/ops/dcp4_compressor_kernels.py"
)
KERNEL_SOURCE = Path(__file__).with_name("dcp4_compressor_kernels.py")

PATCH_MARKER = "# DCP4 exact cross-rank compressor port (45c1582 base)."

OLD_IMPORT = '''from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
'''

NEW_IMPORT = OLD_IMPORT + '''# DCP4 exact cross-rank compressor port (45c1582 base).
from vllm.models.deepseek_v4.common.ops.dcp4_compressor_kernels import (
    dcp_softmax_reduce,
    dsv4_dcp_compressor_partial_stats_kernel,
    dsv4_dcp_finalize_indexer_attn_kernel,
    dsv4_dcp_finalize_indexer_mxfp4_attn_kernel,
    dsv4_dcp_finalize_sparse_attn_kernel,
)
'''

OLD_TRITON_IMPORT_ANCHOR = "from vllm.platforms import current_platform\n"
NEW_TRITON_IMPORT_ANCHOR = OLD_TRITON_IMPORT_ANCHOR + (
    "from vllm.triton_utils import triton\n"
)

OLD_TOKEN_METADATA = '''        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
'''

NEW_TOKEN_METADATA = '''        token_to_req_indices = state_metadata.token_to_req_indices
        assert token_to_req_indices is not None
        slot_mapping = state_metadata.slot_mapping
'''

OLD_DCP_GROUP = '''        dcp_rank = 0
        if dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_rank = get_dcp_group().rank_in_group
'''

NEW_DCP_GROUP = '''        dcp_rank = 0
        dcp_group = None
        if dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_group = get_dcp_group()
            dcp_rank = dcp_group.rank_in_group
'''

OLD_FORWARD_INSERTION = '''        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
'''

NEW_FORWARD_INSERTION = '''        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        if dcp_world_size > 1:
            assert dcp_group is not None
            # Every rank processes the same real global-token rows. Do not use
            # rank-local slot_mapping length or skip a rank with no contribution:
            # the cross-rank collective sequence and shapes must stay identical.
            self._dcp_compress_and_insert(
                state_cache=state_cache,
                num_actual=token_to_req_indices.shape[0],
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                block_table=block_table,
                block_size=block_size,
                state_width=state_width,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                k_cache_metadata=k_cache_metadata,
                pdl_kwargs=pdl_kwargs,
                dcp_group=dcp_group,
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=(
                    parallel_config.cp_kv_cache_interleave_size
                ),
            )
            return

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
'''

DCP_METHOD = '''

    def _dcp_compress_and_insert(
        self,
        state_cache: torch.Tensor,
        num_actual: int,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        state_width: int,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata: Any,
        pdl_kwargs: dict,
        dcp_group: Any,
        dcp_world_size: int,
        dcp_rank: int,
        cp_kv_cache_interleave_size: int,
    ) -> None:
        partial_m = torch.empty(
            (num_actual, self.head_dim),
            dtype=torch.float32,
            device=state_cache.device,
        )
        partial_s = torch.empty_like(partial_m)
        partial_v = torch.empty_like(partial_m)

        dsv4_dcp_compressor_partial_stats_kernel[(num_actual,)](
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            token_to_req_indices,
            positions,
            block_table,
            block_table.stride(0),
            block_size,
            partial_m,
            partial_s,
            partial_v,
            partial_m.stride(0),
            HEAD_SIZE=self.head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            DCP_WORLD_SIZE=dcp_world_size,
            DCP_RANK=dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
            num_warps=4 if self.head_dim == 512 else 1,
            **pdl_kwargs,
        )

        compressed_kv = dcp_softmax_reduce(
            partial_m,
            partial_s,
            partial_v,
            dcp_group,
        )

        if self.head_dim == 512:
            dsv4_dcp_finalize_sparse_attn_kernel[(num_actual,)](
                compressed_kv,
                compressed_kv.stride(0),
                token_to_req_indices,
                positions,
                k_cache_metadata.block_table,
                k_cache_metadata.block_table.stride(0),
                k_cache_metadata.block_size // self.compress_ratio,
                self.norm.weight,
                self.rms_norm_eps,
                cos_sin_cache,
                cos_sin_cache.stride(0),
                kv_cache,
                HEAD_SIZE=self.head_dim,
                TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
                COMPRESS_RATIO=self.compress_ratio,
                ROPE_HEAD_DIM=self.rope_head_dim,
                FP8_MAX=448.0,
                QUANT_BLOCK=self._quant_block,
                TOKEN_STRIDE=self._token_stride,
                SCALE_DIM=self._scale_dim,
                KV_BLOCK_STRIDE=kv_cache.stride(0),
                DCP_WORLD_SIZE=dcp_world_size,
                DCP_RANK=dcp_rank,
                CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
                num_warps=4,
                **pdl_kwargs,
            )
        elif self.use_fp4_cache:
            dsv4_dcp_finalize_indexer_mxfp4_attn_kernel[(num_actual,)](
                compressed_kv,
                compressed_kv.stride(0),
                positions,
                k_cache_metadata.slot_mapping,
                self.norm.weight,
                self.rms_norm_eps,
                cos_sin_cache,
                cos_sin_cache.stride(0),
                kv_cache,
                kv_cache.shape[1],
                HEAD_SIZE=self.head_dim,
                TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
                COMPRESS_RATIO=self.compress_ratio,
                ROPE_HEAD_DIM=self.rope_head_dim,
                QUANT_BLOCK=self._quant_block,
                TOKEN_STRIDE=self._token_stride,
                SCALE_DIM=self._scale_dim,
                KV_BLOCK_STRIDE=kv_cache.stride(0),
                num_warps=1,
                **pdl_kwargs,
            )
        else:
            dsv4_dcp_finalize_indexer_attn_kernel[(num_actual,)](
                compressed_kv,
                compressed_kv.stride(0),
                positions,
                k_cache_metadata.slot_mapping,
                self.norm.weight,
                self.rms_norm_eps,
                cos_sin_cache,
                cos_sin_cache.stride(0),
                kv_cache,
                kv_cache.shape[1],
                HEAD_SIZE=self.head_dim,
                TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
                COMPRESS_RATIO=self.compress_ratio,
                ROPE_HEAD_DIM=self.rope_head_dim,
                FP8_MAX=448.0,
                QUANT_BLOCK=self._quant_block,
                TOKEN_STRIDE=self._token_stride,
                SCALE_DIM=self._scale_dim,
                KV_BLOCK_STRIDE=kv_cache.stride(0),
                num_warps=1,
                **pdl_kwargs,
            )
'''


def read_required(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required pinned-runtime source is missing: {path}")
    return path.read_text()


def replace_once(source: str, old: str, new: str, path: Path) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one pinned 45c1582 source anchor in {path}, "
            f"found {count}: {old[:120]!r}"
        )
    return source.replace(old, new, 1)


def verify_base_contract(source: str) -> None:
    interface_source = read_required(KV_INTERFACE)
    required_compressor = (
        "dcp_sharded=True",
        '"dcp_world_size": dcp_world_size',
        "compress_norm_rope_store_fn(",
    )
    for needle in required_compressor:
        if source.count(needle) != 1:
            raise RuntimeError(
                f"expected exactly one {needle!r} in pinned compressor source"
            )
    if "def _dcp_compress_and_insert(" in source:
        raise RuntimeError(
            "runtime already has a different DCP compressor method but lacks "
            "this patch's marker; refusing to combine implementations"
        )
    if interface_source.count("dcp_sharded: bool = False") != 1:
        raise RuntimeError(
            "pinned SlidingWindowMLASpec.dcp_sharded contract is missing"
        )


def verify_patched_contract(source: str) -> None:
    expected_counts = {
        PATCH_MARKER: 1,
        "def _dcp_compress_and_insert(": 1,
        "num_actual=token_to_req_indices.shape[0]": 1,
        "compressed_kv = dcp_softmax_reduce(": 1,
        # Each kernel name appears once in the import and once at its launch.
        "dsv4_dcp_compressor_partial_stats_kernel": 2,
        "dsv4_dcp_finalize_sparse_attn_kernel": 2,
        "dsv4_dcp_finalize_indexer_attn_kernel": 2,
        "dsv4_dcp_finalize_indexer_mxfp4_attn_kernel": 2,
    }
    for needle, expected in expected_counts.items():
        if source.count(needle) != expected:
            raise RuntimeError(
                f"patched compressor contract expected {expected} {needle!r}, "
                f"found {source.count(needle)}"
            )


def patch_compressor() -> None:
    source = read_required(COMPRESSOR)
    if PATCH_MARKER in source:
        verify_patched_contract(source)
        print(f"DCP4 compressor correctness port already present in {COMPRESSOR}")
        return

    verify_base_contract(source)
    source = replace_once(source, OLD_IMPORT, NEW_IMPORT, COMPRESSOR)
    source = replace_once(
        source,
        OLD_TRITON_IMPORT_ANCHOR,
        NEW_TRITON_IMPORT_ANCHOR,
        COMPRESSOR,
    )
    source = replace_once(
        source,
        OLD_TOKEN_METADATA,
        NEW_TOKEN_METADATA,
        COMPRESSOR,
    )
    source = replace_once(source, OLD_DCP_GROUP, NEW_DCP_GROUP, COMPRESSOR)
    source = replace_once(
        source,
        OLD_FORWARD_INSERTION,
        NEW_FORWARD_INSERTION,
        COMPRESSOR,
    )
    if not source.endswith("\n"):
        raise RuntimeError("pinned compressor.py must end with a newline")
    # DCP_METHOD starts with two newlines; the source already supplies one.
    source += DCP_METHOD[1:]
    verify_patched_contract(source)
    COMPRESSOR.write_text(source)
    print(f"Applied exact DCP4 compressor correctness port to {COMPRESSOR}")


def main() -> None:
    if not KERNEL_SOURCE.is_file():
        raise RuntimeError(f"bundled DCP compressor kernels missing: {KERNEL_SOURCE}")
    TARGET_KERNELS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(KERNEL_SOURCE, TARGET_KERNELS)
    patch_compressor()
    for path in (TARGET_KERNELS, COMPRESSOR, KV_INTERFACE):
        py_compile.compile(str(path), doraise=True)
    print("Verified DCP4 compressor port and Python syntax")


if __name__ == "__main__":
    main()
