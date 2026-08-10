#!/usr/bin/env python3
"""Enable recycled-block zeroing for DeepSeek V4 compressor state caches."""

from pathlib import Path


SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
INTERFACE = SITE_PACKAGES / "vllm/v1/kv_cache_interface.py"
SINGLE_MANAGER = SITE_PACKAGES / "vllm/v1/core/single_type_kv_cache_manager.py"
KV_MANAGER = SITE_PACKAGES / "vllm/v1/core/kv_cache_manager.py"
SCHEDULER_OUTPUT = SITE_PACKAGES / "vllm/v1/core/sched/output.py"
SCHEDULER = SITE_PACKAGES / "vllm/v1/core/sched/scheduler.py"
MODEL_RUNNER = SITE_PACKAGES / "vllm/v1/worker/gpu/model_runner.py"
WORKER_UTILS = SITE_PACKAGES / "vllm/v1/worker/utils.py"


def replace_exact(path: Path, old: str, new: str, expected: int) -> None:
    source = path.read_text()
    actual = source.count(old)
    if actual != expected:
        raise RuntimeError(
            f"expected {expected} occurrence(s) in {path}, found {actual}"
        )
    path.write_text(source.replace(old, new))


replace_exact(
    INTERFACE,
    """    def needs_kv_cache_zeroing(self) -> bool:
        return self.has_mamba_layers
""",
    """    def needs_kv_cache_zeroing(self) -> bool:
        # DeepSeek V4's compressor state is a recurrent cache. Recycled blocks
        # must be cleared just like Mamba state or later requests consume stale
        # compressor values and produce corrupted logits.
        return self.has_mamba_layers or any(
            isinstance(g.kv_cache_spec, SlidingWindowMLASpec)
            for g in self.kv_cache_groups
        )
""",
    expected=1,
)

replace_exact(
    SINGLE_MANAGER,
    """            MLAAttentionSpec,
            HiddenStateCacheSpec,
""",
    """            MLAAttentionSpec,
            SlidingWindowMLASpec,
            HiddenStateCacheSpec,
""",
    expected=1,
)

replace_exact(
    SINGLE_MANAGER,
    """                MLAAttentionSpec,
                HiddenStateCacheSpec,
""",
    """                MLAAttentionSpec,
                SlidingWindowMLASpec,
                HiddenStateCacheSpec,
""",
    expected=1,
)

replace_exact(
    KV_MANAGER,
    """    def take_new_block_ids(self) -> list[int]:
        \"\"\"Drain and return new attention block IDs for zeroing.\"\"\"
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids
""",
    """    def take_new_block_ids(self) -> list[int]:
        \"\"\"Drain and return new attention block IDs for zeroing.\"\"\"
        by_group = self.take_new_block_ids_by_group()
        return [block_id for ids in by_group.values() for block_id in ids]

    def take_new_block_ids_by_group(self) -> dict[int, list[int]]:
        \"\"\"Drain new block IDs without losing KV-cache group ownership.\"\"\"
        by_group: dict[int, list[int]] = {}
        for mgr in self.coordinator.single_type_managers:
            ids = mgr.take_new_block_ids()
            if ids:
                by_group.setdefault(mgr.kv_cache_group_id, []).extend(ids)
        return by_group
""",
    expected=1,
)

replace_exact(
    SCHEDULER_OUTPUT,
    """    new_block_ids_to_zero: list[int] | None = None

    # Dynamic speculative decoding: optimal K chosen by scheduler.
""",
    """    new_block_ids_to_zero: list[int] | None = None

    # Group-scoped form used when independently recycled cache groups cannot
    # safely share a flat block-ID namespace (DeepSeek V4 compressor caches).
    new_block_ids_to_zero_by_group: dict[int, list[int]] | None = None

    # Dynamic speculative decoding: optimal K chosen by scheduler.
""",
    expected=1,
)

replace_exact(
    SCHEDULER,
    """        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )
""",
    """        new_block_ids_to_zero_by_group = (
            (self.kv_cache_manager.take_new_block_ids_by_group() or None)
            if self.needs_kv_cache_zeroing
            else None
        )
        # A flat list makes equal numeric IDs from independently recycled KV
        # groups alias each other. Workers consume the scoped mapping instead.
        new_block_ids_to_zero = None
""",
    expected=1,
)

replace_exact(
    SCHEDULER,
    """            new_block_ids_to_zero=new_block_ids_to_zero,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
""",
    """            new_block_ids_to_zero=new_block_ids_to_zero,
            new_block_ids_to_zero_by_group=new_block_ids_to_zero_by_group,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
""",
    expected=1,
)

replace_exact(
    WORKER_UTILS,
    """    MambaSpec,
    UniformTypeKVCacheSpecs,
""",
    """    MambaSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
""",
    expected=1,
)

replace_exact(
    WORKER_UTILS,
    """                BLOCK_SIZE=blk_size,
            )


@dataclass
""",
    """                BLOCK_SIZE=blk_size,
            )

    def warmup(self, num_kv_blocks: int) -> None:
        \"\"\"Compile the zeroing kernels before the first real request.\"\"\"
        if not self._metas:
            raise RuntimeError(\"KVBlockZeroer has no cache segments\")
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


@dataclass
""",
    expected=1,
)

replace_exact(
    WORKER_UTILS,
    """        static_forward_context: dict[str, Any],
        runner_only_attn_layers: set[str] | None = None,
""",
    """        static_forward_context: dict[str, Any],
        runner_only_attn_layers: set[str] | None = None,
        include_sliding_window_mla: bool = False,
""",
    expected=1,
)

replace_exact(
    WORKER_UTILS,
    """            if not isinstance(spec, FullAttentionSpec):
                continue
""",
    """            if not isinstance(spec, FullAttentionSpec) and not (
                include_sliding_window_mla
                and isinstance(spec, SlidingWindowMLASpec)
            ):
                continue
""",
    expected=1,
)

replace_exact(
    MODEL_RUNNER,
    """from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
""",
    """from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    SlidingWindowMLASpec,
)
""",
    expected=1,
)

replace_exact(
    MODEL_RUNNER,
    """        self.kv_block_zeroer: KVBlockZeroer | None = None

        self.vocab_size = self.model_config.get_vocab_size()
""",
    """        self.kv_block_zeroer: KVBlockZeroer | None = None
        self.group_kv_block_zeroers: dict[int, KVBlockZeroer] | None = None

        self.vocab_size = self.model_config.get_vocab_size()
""",
    expected=1,
)

replace_exact(
    MODEL_RUNNER,
    """    def _init_kv_zero_meta(self) -> None:
        \"\"\"Build KV-block zeroing metadata; invoked from gpu_worker.\"\"\"
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            pin_memory=PIN_MEMORY,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
        )
""",
    """    def _init_kv_zero_meta(self) -> None:
        \"\"\"Build KV-block zeroing metadata; invoked from gpu_worker.\"\"\"
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            pin_memory=PIN_MEMORY,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
        )
        self._init_group_kv_zero_meta()
        assert self.group_kv_block_zeroers is not None
        for zeroer in self.group_kv_block_zeroers.values():
            zeroer.warmup(self.kv_cache_config.num_blocks)

    def _init_group_kv_zero_meta(self) -> None:
        \"\"\"Build one zeroer per independently allocated KV cache group.\"\"\"
        group_ids = {
            group.kv_cache_group_id
            for groups in self.attn_groups
            for group in groups
            if isinstance(
                group.kv_cache_spec,
                (FullAttentionSpec, SlidingWindowMLASpec),
            )
        }
        self.group_kv_block_zeroers = {
            group_id: KVBlockZeroer(
                self.device,
                pin_memory=PIN_MEMORY,
                attn_groups_iter=(
                    group
                    for groups in self.attn_groups
                    for group in groups
                    if group.kv_cache_group_id == group_id
                ),
                kernel_block_sizes=self.kernel_block_sizes,
                cache_dtype=self.cache_config.cache_dtype,
                static_forward_context=self.compilation_config.static_forward_context,
                include_sliding_window_mla=True,
            )
            for group_id in group_ids
        }
""",
    expected=1,
)

replace_exact(
    MODEL_RUNNER,
    """from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    SlidingWindowMLASpec,
)
""",
    """from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SlidingWindowMLASpec,
)
""",
    expected=1,
)

replace_exact(
    MODEL_RUNNER,
    """        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)
""",
    """        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)
        if scheduler_output.new_block_ids_to_zero_by_group:
            assert self.group_kv_block_zeroers is not None
            for group_id, block_ids in (
                scheduler_output.new_block_ids_to_zero_by_group.items()
            ):
                try:
                    zeroer = self.group_kv_block_zeroers[group_id]
                except KeyError as exc:
                    raise RuntimeError(
                        f\"No KV zeroer for cache group {group_id}\"
                    ) from exc
                zeroer.zero_block_ids(block_ids)
""",
    expected=1,
)
