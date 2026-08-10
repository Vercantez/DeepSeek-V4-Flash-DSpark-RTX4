# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP-safe DeepSeek-V4 compressor kernels for the pinned B12X runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

from .fused_indexer_q import _fp32x2_to_fp4x2

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def dcp_softmax_reduce(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Merge stable partial softmax statistics with one exact collective."""
    valid = local_sum > 0
    local_max = torch.where(
        valid,
        local_max,
        torch.full_like(local_max, -float("inf")),
    )
    gathered = group.all_gather(
        torch.stack((local_max, local_sum, local_weighted_value)),
        dim=0,
    ).reshape((group.world_size, 3) + local_max.shape)
    gathered_max = gathered[:, 0]
    gathered_sum = gathered[:, 1]
    gathered_weighted_value = gathered[:, 2]
    gathered_valid = gathered_sum > 0
    global_max = gathered_max.max(dim=0).values

    scale = torch.exp(gathered_max - global_max.unsqueeze(0))
    scale = torch.where(gathered_valid, scale, torch.zeros_like(scale))
    global_sum = (gathered_sum * scale).sum(dim=0)
    global_weighted_value = (gathered_weighted_value * scale).sum(dim=0)
    return torch.where(
        global_sum > 0,
        global_weighted_value / global_sum,
        torch.zeros_like(global_weighted_value),
    )


@triton.jit
def _cp_is_local_pos(
    pos,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    return (
        pos // CP_KV_CACHE_INTERLEAVE_SIZE
    ) % DCP_WORLD_SIZE == DCP_RANK


@triton.jit
def cp_global_to_local_block(
    pos,
    block_size,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    """Map a global logical position to a rank-local physical block address."""
    virtual_block_size = block_size * DCP_WORLD_SIZE
    block_idx = pos // virtual_block_size
    virtual_offset = pos - block_idx * virtual_block_size
    block_offset = (
        virtual_offset
        // (DCP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
        virtual_offset % CP_KV_CACHE_INTERLEAVE_SIZE
    )
    is_local = _cp_is_local_pos(
        virtual_offset,
        DCP_WORLD_SIZE,
        DCP_RANK,
        CP_KV_CACHE_INTERLEAVE_SIZE,
    )
    return block_idx, block_offset, is_local


@triton.jit
def dsv4_dcp_compressor_partial_stats_kernel(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── outputs ──
    partial_m_ptr,
    partial_s_ptr,
    partial_v_ptr,
    partial_stride0,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    out_offsets = token_idx * partial_stride0 + block

    tl.store(partial_m_ptr + out_offsets, -float("inf"), mask=mask)
    tl.store(partial_s_ptr + out_offsets, 0.0, mask=mask)
    tl.store(partial_v_ptr + out_offsets, 0.0, mask=mask)

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices, local_block_offsets, is_local = cp_global_to_local_block(
        pos,
        block_size,
        DCP_WORLD_SIZE,
        DCP_RANK,
        CP_KV_CACHE_INTERLEAVE_SIZE,
    )
    mask_pos = mask_pos & is_local

    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + local_block_offsets * state_cache_stride1
        + head_offset
    )
    combined_mask = mask_pos[:, None] & mask[None, :]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=-float("inf"),
    )
    local_m = tl.max(score, axis=0)
    safe_m = tl.where(local_m == -float("inf"), 0.0, local_m)
    weight = tl.exp(score - safe_m[None, :])
    weight = tl.where(combined_mask, weight, 0.0)
    local_s = tl.sum(weight, axis=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )
    local_v = tl.sum(kv * weight, axis=0)
    local_m = tl.where(local_s > 0, local_m, -float("inf"))

    tl.store(partial_m_ptr + out_offsets, local_m, mask=mask)
    tl.store(partial_s_ptr + out_offsets, local_s, mask=mask)
    tl.store(partial_v_ptr + out_offsets, local_v, mask=mask)


@triton.jit
def dsv4_dcp_finalize_sparse_attn_kernel(
    compressed_kv_ptr,
    compressed_stride0,
    token_to_req_indices_ptr,
    positions_ptr,
    kv_block_table_ptr,
    kv_block_table_stride,
    kv_block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    compressed_pos = position // COMPRESS_RATIO
    kv_block_idx, local_block_offset, is_local = cp_global_to_local_block(
        compressed_pos,
        kv_block_size,
        DCP_WORLD_SIZE,
        DCP_RANK,
        CP_KV_CACHE_INTERLEAVE_SIZE,
    )
    if not is_local:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    block_number = tl.load(
        kv_block_table_ptr + req_idx * kv_block_table_stride + kv_block_idx
    )
    kv_slot_idx = block_number * kv_block_size + local_block_offset

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    compressed_kv = tl.load(
        compressed_kv_ptr + token_idx * compressed_stride0 + block,
        mask=mask,
        other=0.0,
    )

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    cache_block_idx = kv_slot_idx // kv_block_size
    kv_pos_in_block = kv_slot_idx % kv_block_size
    cache_block_ptr = k_cache_ptr + cache_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr + kv_block_size * TOKEN_STRIDE + (kv_pos_in_block * SCALE_DIM)
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    block_absmax = tl.max(tl.abs(quant_2d), axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)
    exponents = tl.ceil(tl.log2(block_absmax * INV_FP8_MAX))
    inv_scales = tl.exp2(-exponents)
    x_scaled = quant_2d * tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_uint8 = tl.reshape(
        x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
        (TRITON_BLOCK_SIZE,),
    )
    tl.store(fp8_ptr + block, x_uint8, mask=block < NOPE_HEAD_DIM)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_rope_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_rope_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    result = tl.interleave(even * cos_v - odd * sin_v, odd * cos_v + even * sin_v)

    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    tl.store(
        bf16_ptr + rope_local,
        result.to(tl.bfloat16),
        mask=(block >= NOPE_HEAD_DIM) & mask,
    )


@triton.jit
def dsv4_dcp_finalize_indexer_attn_kernel(
    compressed_kv_ptr,
    compressed_stride0,
    positions_ptr,
    kv_slot_mapping_ptr,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    compressed_kv = tl.load(
        compressed_kv_ptr + token_idx * compressed_stride0 + block,
        mask=mask,
        other=0.0,
    )

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    result = tl.interleave(even * cos_v - odd * sin_v, odd * cos_v + even * sin_v)

    tl.static_assert(
        TRITON_BLOCK_SIZE == QUANT_BLOCK,
        "Indexer expects one quant block (QUANT_BLOCK == TRITON_BLOCK_SIZE)",
    )
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(result_bf16), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    exponent = tl.ceil(tl.log2(absmax * INV_FP8_MAX))
    inv_scale = tl.exp2(-exponent)

    x_scaled = result_bf16 * inv_scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_uint8 = x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(fp8_ptr + block, x_uint8, mask=mask)

    scale_val = tl.exp2(exponent)
    tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale_val)


@triton.jit
def dsv4_dcp_finalize_indexer_mxfp4_attn_kernel(
    compressed_kv_ptr,
    compressed_stride0,
    positions_ptr,
    kv_slot_mapping_ptr,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    compressed_kv = tl.load(
        compressed_kv_ptr + token_idx * compressed_stride0 + block,
        mask=mask,
        other=0.0,
    )

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = (even * cos_v - odd * sin_v).to(tl.bfloat16).to(tl.float32)
    new_odd = (odd * cos_v + even * sin_v).to(tl.bfloat16).to(tl.float32)

    N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // QUANT_BLOCK
    HALF_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    tl.static_assert(TRITON_BLOCK_SIZE == HEAD_SIZE)
    tl.static_assert(HEAD_SIZE % QUANT_BLOCK == 0)
    tl.static_assert(TOKEN_STRIDE == HEAD_SIZE // 2)
    tl.static_assert(SCALE_DIM == N_QUANT_BLOCKS)

    even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, HALF_BLOCK))
    odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, HALF_BLOCK))
    amax = tl.maximum(
        tl.max(tl.abs(even_2d), axis=1),
        tl.max(tl.abs(odd_2d), axis=1),
    )
    amax = tl.maximum(amax, 6.0 * (2**-126))

    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    packed = _fp32x2_to_fp4x2(even_2d * inv_scale_col, odd_2d * inv_scale_col)
    packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))

    tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), packed_flat)
    tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)
