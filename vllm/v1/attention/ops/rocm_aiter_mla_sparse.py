# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
import json
import math
import os
import sys
from importlib.util import find_spec

import torch
import torch.nn.functional as F

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950
else:
    _ON_GFX942 = False
    _ON_GFX950 = False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_int_or_none(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_bool_default(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value not in ("0", "false", "False", "no", "No")


def _sparse_indexer_debug_enabled() -> bool:
    return os.getenv("DSV4_SPARSE_INDEXER_DEBUG", "0") == "1"


def _sparse_prefill_mem_metrics_enabled() -> bool:
    return os.getenv("DSV4_SPARSE_PREFILL_MEM_METRICS", "0") == "1"


def _sparse_prefill_mem_snapshot(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free": int(free),
        "total": int(total),
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
    }


def _log_sparse_prefill_mem_metrics(
    event: str,
    device: torch.device,
    **kwargs: int | str | float,
) -> None:
    if not _sparse_prefill_mem_metrics_enabled():
        return
    record: dict[str, int | str | float] = {
        "event": event,
        **_sparse_prefill_mem_snapshot(device),
        **kwargs,
    }
    print(
        "DSV4_SPARSE_PREFILL_MEM " + json.dumps(record, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _log_sparse_indexer_debug(message: str, device: torch.device) -> None:
    if not _sparse_indexer_debug_enabled():
        return
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"DSV4_SPARSE_INDEXER {message} "
        f"free_gib={free / 2**30:.3f} total_gib={total / 2**30:.3f}",
        file=sys.stderr,
        flush=True,
    )


def _select_sparse_decode_config(
    num_queries: int,
    head_dim: int,
    extra_indices: torch.Tensor,
) -> tuple[int, int, int]:
    block_h_override = _env_int_or_none("DSV4_SPARSE_ATTN_DECODE_BLOCK_H")
    block_k_override = _env_int_or_none("DSV4_SPARSE_ATTN_DECODE_BLOCK_K")
    num_warps_override = _env_int_or_none("DSV4_SPARSE_ATTN_DECODE_NUM_WARPS")

    block_h = 16
    block_k = 16 if head_dim >= 256 else 32
    num_warps = 4

    if (
        block_h_override is None
        and block_k_override is None
        and num_warps_override is None
        and not _env_bool_default("DSV4_SPARSE_ATTN_DECODE_DYNAMIC_DEFAULT", False)
    ):
        return block_h, block_k, num_warps

    extra_per_query = (
        extra_indices.numel() // num_queries if num_queries > 0 else 0
    )
    if extra_per_query <= 8:
        if num_queries >= 256:
            block_h, block_k = 64, 16
            if extra_per_query > 0:
                num_warps = 8
        elif num_queries >= 80:
            block_h, block_k = 32, 16
        elif num_queries == 32:
            block_h, block_k = 16, 16
        else:
            block_h, block_k = 16, 32
    else:
        if num_queries < 32:
            block_h, block_k = 4, 64
        elif num_queries >= 256:
            block_h, block_k = 64, 16
            num_warps = 8
        elif num_queries >= 80:
            block_h, block_k = 32, 16
        else:
            block_h, block_k = 16, 32

    if block_h_override is not None:
        block_h = block_h_override
    if block_k_override is not None:
        block_k = block_k_override
    if num_warps_override is not None:
        num_warps = num_warps_override
    return block_h, block_k, num_warps


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    kv_cache_value = kv_cache[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        "SHUFFLE",
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    valid_tid = tid < NUM_TOKENS
    batch_id = tl.load(token_to_seq_ptr + tid, mask=valid_tid, other=-1)
    valid_batch = (batch_id >= 0) & (batch_id < NUM_BATCHES)
    safe_batch_id = tl.where(valid_batch, batch_id, 0)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=valid_batch, other=0)
    batch_end = tl.load(cu_seqlen_ptr + safe_batch_id + 1, mask=valid_batch, other=0)
    batch_offset = tid - batch_start
    valid_token = valid_tid & valid_batch & (tid >= batch_start) & (tid < batch_end)
    if not valid_token:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    valid_block_table = (
        valid_token
        & (block_table_id >= 0)
        & (block_table_id < BLOCK_TABLE_WIDTH)
        & (block_offset >= 0)
        & (block_offset < block_size)
    )
    safe_block_table_id = tl.where(valid_block_table, block_table_id, 0)
    block_table_offset = safe_batch_id * block_table_stride + safe_block_table_id
    block_id = tl.load(
        block_table_ptr + block_table_offset, mask=valid_block_table, other=-1
    )
    valid_block = valid_block_table & (block_id >= 0) & (block_id < NUM_BLOCKS)
    safe_block_id = tl.where(valid_block, block_id, 0)
    safe_block_offset = tl.where(valid_block, block_offset, 0)
    tiled_block_offset = safe_block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            safe_block_id * kv_cache_stride
            + (safe_block_offset // BLOCK_TILE_SIZE) * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = (
            safe_block_id * kv_cache_stride + safe_block_offset * HEAD_DIM
        )
    src_scale_offset = safe_block_id * kv_cache_scale_stride + safe_block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr, mask=valid_block, other=0.0)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val, mask=valid_block)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        "SHUFFLE",
        head_dim,
        block_tile_size,
        head_tile_size,
        num_tokens,
        cu_seqlen.shape[0] - 1,
        block_table.shape[1],
        num_blocks,
    )


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        max_blocks = cdiv(max_model_len, block_size)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        pages = block_tables[:, :max_blocks].clamp_min(0)
        cache = kv_cache_flat[pages]
        scale_offset = block_size * dim
        cache_value = cache[..., :scale_offset].view(dtype=fp8_dtype)
        cache_value = cache_value.to(torch.float32).view(batch_size, -1, dim)
        cache_scale = cache[..., scale_offset:].view(dtype=torch.float32)
        cache_scale = cache_scale.contiguous().view(batch_size, -1)

        q = q[:, 0].to(torch.float32)
        scores = torch.einsum("bhd,btd->bht", q, cache_value)
        scores = F.relu(scores) * weights[:, :, None]
        logits = scores.sum(dim=1) * cache_scale
        logits = logits[:, :max_model_len]
        positions = torch.arange(max_model_len, device=q.device)
        return logits.masked_fill(
            positions[None, :] >= context_lens[:, None],
            float("-inf"),
        )

    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=q.device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=q.device
            )
        else:
            context_limit = context_len.to(device=q.device, dtype=torch.int32)
            q_offsets = context_limit - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@functools.lru_cache
def paged_mqa_logits_module():
    paged_mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

    if paged_mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(paged_mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_paged_mqa_logits_module = None
    # if rocm_aiter_ops.is_enabled():
    batch_size, next_n, heads, head_dim = q_fp8.shape
    num_blocks, block_size, _, _ = kv_cache_fp8.shape

    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        if _ON_GFX942 or _ON_GFX950:
            deepgemm_fp8_paged_mqa_logits = (
                aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits
            )
            batch_size, next_n, heads, _ = q_fp8.shape
            out_logits = torch.full(
                [batch_size * next_n, max_model_len],
                float("-inf"),
                device="cuda",
                dtype=torch.float32,
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=256,
                Preshuffle=block_size == 64,
                KVBlockSize=block_size,
                WavePerEU=2,
            )
            return out_logits
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ=heads,
        )
        return out_qk.sum(dim=0)
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    if scale.ndim == 2:
        if scale.shape[1] == 1:
            scale = scale.squeeze(1)
        else:
            scale = scale.reshape(-1).repeat_interleave(128)[:seq_len_kv]
    k = k * scale.reshape(seq_len_kv, 1).to(k.dtype)
    q = q.to(torch.bfloat16)
    device = q.device

    mask_lo = (
        torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float()
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@functools.lru_cache
def mqa_logits_module():
    mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

    if mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # AITER's current fp8_mqa_logits Triton kernel can request 96 KiB LDS for
    # long prefill windows on MI300X/gfx942, above the 64 KiB per-block limit.
    # Keep the fast path for shorter windows and fall back to the torch
    # reference for longer ones until the kernel is retiled.
    max_aiter_seq_len = _env_int("DSV4_AITER_PREFILL_MQA_LOGITS_MAX_N", 0)
    module = (
        mqa_logits_module()
        if os.getenv("DSV4_USE_AITER_PREFILL_MQA_LOGITS", "0") == "1"
        and kv[0].shape[0] <= max_aiter_seq_len
        else None
    )
    if module is not None:
        k_fp8, scale = kv
        return module.fp8_mqa_logits(
            q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke
        )

    return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def _topk_indices_torch(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    if row_starts is not None:
        # Match the CUDA top_k_per_row_prefill contract: indices are local to
        # each row's valid [row_start, row_end) range, not columns in the
        # concatenated chunk logits matrix.
        starts = row_starts.to(dtype=torch.int32).view(-1, 1)
        indices = torch.where(indices < 0, indices, indices - starts)
    if k == topk_tokens:
        return indices
    padded = torch.full(
        (logits.shape[0], topk_tokens),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    padded[:, :k] = indices
    return padded


# topk_tokens values with dedicated fused C++ kernel support.
_TOPK_FAST_PATH_VALUES = frozenset({2048})


def _sparse_prefill_chunk_size() -> int:
    return max(0, _env_int("DSV4_SPARSE_PREFILL_CHUNK_SIZE", 0))


def _sparse_prefill_logits_chunk_size() -> int:
    return max(1, _env_int("DSV4_SPARSE_PREFILL_LOGITS_CHUNK_SIZE", 512))


@triton.jit
def _fill_full_window_topk_prefill_kernel(
    topk_out_ptr,
    topk_out_stride,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    topk_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_start = tl.load(cu_seqlen_ks_ptr + row_id)
    row_end = tl.load(cu_seqlen_ke_ptr + row_id)
    row_len = row_end - row_start
    values = tl.where(offsets < row_len, offsets, -1)
    tl.store(
        topk_out_ptr + row_id * topk_out_stride + offsets,
        values,
        mask=offsets < topk_tokens,
    )


def _fill_full_window_topk_prefill(
    topk_out: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
) -> None:
    block = 256
    grid = (topk_out.shape[0], triton.cdiv(topk_tokens, block))
    _fill_full_window_topk_prefill_kernel[grid](
        topk_out,
        topk_out.stride(0),
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_tokens,
        BLOCK_SIZE=block,
    )


@triton.jit
def _fill_full_window_topk_decode_kernel(
    topk_out_ptr,
    topk_out_stride,
    seq_lens_ptr,
    topk_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_len = tl.load(seq_lens_ptr + row_id)
    values = tl.where(offsets < seq_len, offsets, -1)
    tl.store(
        topk_out_ptr + row_id * topk_out_stride + offsets,
        values,
        mask=offsets < topk_tokens,
    )


def _fill_full_window_topk_decode(
    topk_out: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_tokens: int,
) -> None:
    block = 256
    seq_lens = seq_lens.reshape(-1)
    grid = (topk_out.shape[0], triton.cdiv(topk_tokens, block))
    _fill_full_window_topk_decode_kernel[grid](
        topk_out,
        topk_out.stride(0),
        seq_lens,
        topk_tokens,
        BLOCK_SIZE=block,
    )


def _topk_indices_prefill(
    logits: torch.Tensor,
    topk_tokens: int,
    topk_out: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> None:
    """Top-k indices for the prefill path.

    Writes ``logits.shape[0]`` rows into ``topk_out``; caller must size the
    view accordingly.
    """
    if topk_tokens in _TOPK_FAST_PATH_VALUES:
        torch.ops._C.top_k_per_row_prefill(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk_out,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
    else:
        topk_out.copy_(
            _topk_indices_torch(logits, topk_tokens, row_starts=cu_seqlen_ks)
        )


def _topk_indices_decode(
    logits: torch.Tensor,
    topk_tokens: int,
    topk_out: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
) -> None:
    """Top-k indices for the decode path.

    Writes ``logits.shape[0] == batch_size * next_n`` rows into ``topk_out``;
    caller must size the view to ``num_padded_tokens``.
    """
    if topk_tokens in _TOPK_FAST_PATH_VALUES:
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            topk_out,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
    else:
        topk_out.copy_(_topk_indices_torch(logits, topk_tokens))


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


@eager_break_during_capture
def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import _resolve_layer_name

    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None
    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens
    device = hidden_states.device if k is None else k.device

    # during speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")

    if not skip_k_cache_insert:
        if _ON_GFX942:
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )
        else:
            indexer_k_quant_and_cache_triton(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None
        prefill_chunk_size = _sparse_prefill_chunk_size()
        for chunk_idx, chunk in enumerate(prefill_metadata.chunks):
            _log_sparse_prefill_mem_metrics(
                "before_gather",
                device,
                chunk=chunk_idx,
                M=chunk.token_end - chunk.token_start,
                N=chunk.total_seq_lens,
                topk=topk_tokens,
            )
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=device,
                dtype=fp8_dtype,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=device,
                dtype=torch.uint8,
            )
            if _ON_GFX942:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            else:
                cp_gather_indexer_k_quant_cache_triton(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                    token_to_seq=chunk.token_to_seq,
                )
            _log_sparse_prefill_mem_metrics(
                "after_gather",
                device,
                chunk=chunk_idx,
                M=chunk.token_end - chunk.token_start,
                N=chunk.total_seq_lens,
                topk=topk_tokens,
            )

            chunk_tokens = chunk.token_end - chunk.token_start
            chunk_max_seq_len = getattr(chunk, "max_seq_len", chunk.total_seq_lens)
            if prefill_chunk_size:
                row_step = prefill_chunk_size
            elif chunk_max_seq_len <= topk_tokens:
                row_step = chunk_tokens
            else:
                row_step = min(chunk_tokens, _sparse_prefill_logits_chunk_size())
            for row_start in range(0, chunk_tokens, row_step):
                row_end = min(row_start + row_step, chunk_tokens)
                token_start = chunk.token_start + row_start
                token_end = chunk.token_start + row_end
                cu_seqlen_ks = chunk.cu_seqlen_ks[row_start:row_end]
                cu_seqlen_ke = chunk.cu_seqlen_ke[row_start:row_end]
                topk_indices = topk_indices_buffer[
                    token_start:token_end, :topk_tokens
                ]
                _log_sparse_prefill_mem_metrics(
                    "before_topk_select",
                    device,
                    chunk=chunk_idx,
                    row_start=row_start,
                    rows=row_end - row_start,
                    N=chunk.total_seq_lens,
                    topk=topk_tokens,
                )
                if chunk_max_seq_len <= topk_tokens:
                    _fill_full_window_topk_prefill(
                        topk_indices,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_tokens,
                    )
                    _log_sparse_prefill_mem_metrics(
                        "after_full_window_fill",
                        device,
                        chunk=chunk_idx,
                        row_start=row_start,
                        rows=row_end - row_start,
                        N=chunk.total_seq_lens,
                        topk=topk_tokens,
                    )
                else:
                    logits = rocm_fp8_mqa_logits(
                        q_fp8[token_start:token_end],
                        (k_fp8, k_scale.view(torch.float32)),
                        weights[token_start:token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                    _log_sparse_indexer_debug(
                        "after_logits "
                        f"M={row_end - row_start} "
                        f"N={chunk.total_seq_lens}",
                        device,
                    )
                    _log_sparse_prefill_mem_metrics(
                        "after_logits",
                        device,
                        chunk=chunk_idx,
                        row_start=row_start,
                        rows=row_end - row_start,
                        N=chunk.total_seq_lens,
                        topk=topk_tokens,
                    )
                    _topk_indices_prefill(
                        logits,
                        topk_tokens,
                        topk_indices,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                    _log_sparse_prefill_mem_metrics(
                        "after_topk",
                        device,
                        chunk=chunk_idx,
                        row_start=row_start,
                        rows=row_end - row_start,
                        N=chunk.total_seq_lens,
                        topk=topk_tokens,
                    )
                _log_sparse_indexer_debug(
                    "after_topk "
                    f"M={row_end - row_start} "
                    f"N={chunk.total_seq_lens}",
                    device,
                )

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        max_seq_len = getattr(decode_metadata, "max_seq_len", None)
        if max_seq_len is None:
            max_seq_len = int(decode_metadata.seq_lens.max().item())

        if max_seq_len <= topk_tokens:
            _fill_full_window_topk_decode(
                topk_indices,
                decode_metadata.seq_lens,
                topk_tokens,
            )
            if decode_metadata.requires_padding:
                # if padded, we need to unpack
                # the topk indices removing padded tokens
                topk_indices = unpack_seq_triton(
                    topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
                    decode_lens,
                )
                topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                    topk_indices
                )
            return topk_indices_buffer

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        _topk_indices_decode(
            logits,
            topk_tokens,
            topk_indices,
            decode_metadata.seq_lens,
            next_n,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _apply_gptj_inv_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0 or x.numel() == 0:
        return x
    half_rot = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    dtype = x.dtype
    x = x.to(torch.float32)
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[:, :half_rot].to(torch.float32)
    sin = cache[:, half_rot : 2 * half_rot].to(torch.float32)
    view_shape = (positions.shape[0],) + (1,) * (x.dim() - 2) + (half_rot,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    rope = x[..., nope_dim:]
    y_even = rope[..., 0::2]
    y_odd = rope[..., 1::2]
    rope_out = torch.stack(
        (y_even * cos + y_odd * sin, y_odd * cos - y_even * sin),
        dim=-1,
    ).flatten(-2)
    x = x.clone()
    x[..., nope_dim:] = rope_out
    return x.to(dtype)


def _apply_inv_rope_ref(
    rotary_emb: torch.nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if hasattr(rotary_emb, "forward_native"):
        try:
            query, _ = rotary_emb.forward_native(
                positions,
                x.clone(),
                None,
                inverse=True,
            )
            return query
        except TypeError:
            pass
    return _apply_gptj_inv_rope_ref(x, positions, rotary_emb.cos_sin_cache, rope_dim)


def rocm_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Reference inverse-RoPE + WO_A einsum path used on ROCm."""
    o_ref = _apply_inv_rope_ref(rotary_emb, o, positions, rope_head_dim).to(
        torch.bfloat16
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    hidden_dim = o_ref.shape[-1]
    use_cache = _env_bool_default("DSV4_ROCM_WOA_BF16_WEIGHT_CACHE_DEFAULT", False)

    if hasattr(wo_a, "weight_scale_inv"):
        cache_key = (
            n_local_groups,
            o_lora_rank,
            hidden_dim,
            wo_a.weight.data_ptr(),
            wo_a.weight_scale_inv.data_ptr(),
        )
        cached_key = getattr(wo_a, "_vllm_rocm_bf16_weight_key", None)
        wo_a_weight = getattr(wo_a, "_vllm_rocm_bf16_weight_cache", None)
        if (not use_cache) or cached_key != cache_key or wo_a_weight is None:
            weight_fp32 = wo_a.weight.view(
                n_local_groups, o_lora_rank, hidden_dim
            ).to(torch.float32)
            wo_a_scale = _expand_2d_block_scales(
                wo_a.weight_scale_inv.view(
                    n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
                ),
                o_lora_rank,
                hidden_dim,
            )
            wo_a_weight = (weight_fp32 * wo_a_scale).to(torch.bfloat16)
            if use_cache:
                wo_a._vllm_rocm_bf16_weight_cache = wo_a_weight
                wo_a._vllm_rocm_bf16_weight_key = cache_key
    else:
        cache_key = (n_local_groups, o_lora_rank, hidden_dim, wo_a.weight.data_ptr())
        cached_key = getattr(wo_a, "_vllm_rocm_bf16_weight_key", None)
        wo_a_weight = getattr(wo_a, "_vllm_rocm_bf16_weight_cache", None)
        if (not use_cache) or cached_key != cache_key or wo_a_weight is None:
            wo_a_weight = wo_a.weight.view(
                n_local_groups, o_lora_rank, hidden_dim
            ).to(torch.bfloat16)
            if use_cache:
                wo_a._vllm_rocm_bf16_weight_cache = wo_a_weight
                wo_a._vllm_rocm_bf16_weight_key = cache_key

    return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)


_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    )
    assert (
        nope_head_dim == _DSV4_SPARSE_NOPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    ), (
        f"{op_name} expects {_DSV4_SPARSE_NOPE_DIM} NoPE dims and "
        f"{_DSV4_SPARSE_ROPE_DIM} RoPE dims"
    )


_DSV4_SPARSE_DECODE_SHAPE_CALLS = 0


def _shape_dump_limit() -> int:
    return _env_int("DSV4_SPARSE_DECODE_SHAPE_DUMP_LIMIT", 0)


def _shape_dump_stride() -> int:
    return max(1, _env_int("DSV4_SPARSE_DECODE_SHAPE_DUMP_STRIDE", 1))


def _tensor_shape(x: torch.Tensor | None) -> list[int] | None:
    return None if x is None else list(x.shape)


def _tensor_stride(x: torch.Tensor | None) -> list[int] | None:
    return None if x is None else list(x.stride())


def _length_summary(x: torch.Tensor | None) -> dict[str, object] | None:
    if x is None:
        return None
    flat = x.detach().to("cpu", dtype=torch.int64).reshape(-1)
    if flat.numel() == 0:
        return {
            "numel": 0,
            "sum": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "hist": [],
        }
    values, counts = torch.unique(flat, sorted=True, return_counts=True)
    return {
        "numel": int(flat.numel()),
        "sum": int(flat.sum().item()),
        "min": int(flat.min().item()),
        "max": int(flat.max().item()),
        "mean": float(flat.float().mean().item()),
        "hist": [
            [int(v.item()), int(c.item())] for v, c in zip(values, counts)
        ],
    }


def _indptr_length_summary(indptr: torch.Tensor | None) -> dict[str, object] | None:
    if indptr is None:
        return None
    return _length_summary(indptr[1:] - indptr[:-1])


def _maybe_dump_sparse_decode_shape(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    dump_dir = os.getenv("DSV4_SPARSE_DECODE_SHAPE_DUMP_DIR", "")
    if not dump_dir:
        return
    if torch.cuda.is_current_stream_capturing():
        return

    global _DSV4_SPARSE_DECODE_SHAPE_CALLS
    call_idx = _DSV4_SPARSE_DECODE_SHAPE_CALLS
    _DSV4_SPARSE_DECODE_SHAPE_CALLS += 1

    limit = _shape_dump_limit()
    stride = _shape_dump_stride()
    if limit and call_idx >= limit:
        return
    if call_idx % stride != 0:
        return

    os.makedirs(dump_dir, exist_ok=True)
    main_ragged_lens = _indptr_length_summary(swa_ragged_indptr)
    extra_ragged_lens = _indptr_length_summary(topk_ragged_indptr)
    record = {
        "call_idx": call_idx,
        "pid": os.getpid(),
        "rank": os.getenv("RANK"),
        "local_rank": os.getenv("LOCAL_RANK"),
        "q_shape": _tensor_shape(q),
        "q_stride": _tensor_stride(q),
        "q_dtype": str(q.dtype),
        "output_shape": _tensor_shape(output),
        "output_stride": _tensor_stride(output),
        "output_dtype": str(output.dtype),
        "swa_only": bool(swa_only),
        "swa_cache_shape": _tensor_shape(swa_k_cache),
        "swa_cache_stride": _tensor_stride(swa_k_cache),
        "kv_cache_shape": _tensor_shape(kv_cache),
        "kv_cache_stride": _tensor_stride(kv_cache),
        "swa_indices_shape": _tensor_shape(swa_indices),
        "topk_indices_shape": _tensor_shape(topk_indices),
        "swa_ragged_indices_shape": _tensor_shape(swa_ragged_indices),
        "topk_ragged_indices_shape": _tensor_shape(topk_ragged_indices),
        "swa_lens": _length_summary(swa_lens),
        "topk_lens": _length_summary(topk_lens),
        "swa_ragged_lens": main_ragged_lens,
        "topk_ragged_lens": extra_ragged_lens,
        "effective_swa_lens": main_ragged_lens or _length_summary(swa_lens),
        "effective_topk_lens": extra_ragged_lens or _length_summary(topk_lens),
    }
    path = os.path.join(dump_dir, f"sparse_decode_shapes_{os.getpid()}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


@triton.jit
def _pack_dense_prefix_to_ragged_kernel(
    indices_ptr,
    lengths_ptr,
    indptr_ptr,
    out_ptr,
    indices_stride0,
    num_rows_limit,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_len = tl.load(lengths_ptr + row_idx)
    if block_idx * BLOCK_SIZE >= row_len:
        return

    mask = offsets < row_len
    safe_offsets = tl.where(offsets < row_width, offsets, 0)
    vals = tl.load(
        indices_ptr + row_idx * indices_stride0 + safe_offsets,
        mask=mask & (offsets < row_width),
        other=-1,
    ).to(tl.int32)
    if num_rows_limit >= 0:
        vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)

    out_start = tl.load(indptr_ptr + row_idx)
    tl.store(out_ptr + out_start + offsets, vals, mask=mask)


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    indptr = torch.empty(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    indptr[0] = 0
    torch.cumsum(lengths, dim=0, out=indptr[1:])

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        flat = torch.empty(
            int(indptr[-1].item()), dtype=torch.int32, device=indices.device
        )
        if flat.numel() > 0:
            block_size = 128
            _pack_dense_prefix_to_ragged_kernel[
                (indices.shape[0], triton.cdiv(max_width, block_size))
            ](
                indices,
                lengths,
                indptr,
                flat,
                indices.stride(0),
                int(num_rows),
                max_width,
                BLOCK_SIZE=block_size,
            )

    return flat, indptr


def build_ragged_indices_from_dense_out(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    indices_out: torch.Tensor,
    indptr_out: torch.Tensor,
    num_rows_limit: int = -1,
    max_entries_per_row: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack dense prefix rows directly into persistent ragged buffers.

    This is the CUDA-graph-friendly decode variant of
    build_ragged_indices_from_dense. It avoids allocating a dynamically sized
    flat tensor and, more importantly, avoids the host sync from
    ``int(indptr[-1].item())`` before copying into graph-stable buffers.

    ``indptr_out[0]`` must already be initialized to zero. Keeping that value
    persistent avoids a per-step host-to-device scalar copy in decode.
    """
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    num_rows = indices.shape[0]
    max_width = indices.shape[1] if indices.ndim == 2 else 0
    max_entries = max(num_rows * (max_entries_per_row or max_width), 1)
    assert indptr_out.numel() >= num_rows + 1
    assert indices_out.numel() >= max_entries

    lengths = lengths.clamp(min=0, max=max_width).contiguous()
    indptr = indptr_out[: num_rows + 1]
    torch.cumsum(lengths, dim=0, out=indptr[1:])

    flat = indices_out[:max_entries]
    if indices.numel() > 0 and max_width > 0:
        block_size = 128
        _pack_dense_prefix_to_ragged_kernel[
            (num_rows, triton.cdiv(max_width, block_size))
        ](
            indices,
            lengths,
            indptr,
            flat,
            indices.stride(0),
            int(num_rows_limit),
            max_width,
            BLOCK_SIZE=block_size,
        )

    return flat, indptr


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + safe_slot[:, None] * kv_stride_n
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        kv = tl.where(valid[:, None] & dim_mask[None, :], kv, 0.0)

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _sparse_attn_decode_ragged_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start

        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ:
                x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out_nope = tl.where(
            l_final[:, None] > 0.0,
            (acc_nope * alpha[:, None]) / denom[:, None],
            0.0,
        )
        out_rope = tl.where(
            l_final[:, None] > 0.0,
            (acc_rope * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out_nope = tl.where(l_i[:, None] > 0.0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l_i[:, None] > 0.0, acc_rope / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


def _rocm_sparse_attn_prefill_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_prefill_ragged_triton",
    )

    block_h = 16
    block_d = triton.next_power_of_2(head_dim)
    block_k = 16 if head_dim >= 256 else 32
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert out.shape == q.shape, f"expected out shape {q.shape}, got {out.shape}"
        assert out.dtype == torch.bfloat16, f"expected bf16 out, got {out.dtype}"
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def _rocm_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        num_rows=kv.shape[0],
    )
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )


def _rocm_sparse_attn_decode_ragged_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        not q.is_cpu
        and not main_cache.is_cpu
        and not main_indices.is_cpu
        and not main_indptr.is_cpu
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert main_indptr.numel() == num_queries + 1, (
        f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_decode_ragged_triton",
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    block_h, block_k, num_warps = _select_sparse_decode_config(
        num_queries,
        head_dim,
        extra_indices,
    )
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert out.shape == q.shape, f"expected out shape {q.shape}, got {out.shape}"
        assert out.dtype == torch.bfloat16, f"expected bf16 out, got {out.dtype}"
    _sparse_attn_decode_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        IS_FNUZ=current_platform.is_fp8_fnuz(),
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def _rocm_sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    main_lengths: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    main_ragged_indices: torch.Tensor | None = None,
    main_ragged_indptr: torch.Tensor | None = None,
    extra_ragged_indices: torch.Tensor | None = None,
    extra_ragged_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if main_ragged_indices is None or main_ragged_indptr is None:
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices,
            main_lengths
            if main_lengths is not None
            else (main_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=main_cache.shape[0] * main_cache.shape[1],
        )

    if (
        (extra_ragged_indices is None or extra_ragged_indptr is None)
        and extra_cache is not None
        and extra_indices is not None
    ):
        extra_ragged_indices, extra_ragged_indptr = build_ragged_indices_from_dense(
            extra_indices,
            extra_lengths
            if extra_lengths is not None
            else (extra_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=extra_cache.shape[0] * extra_cache.shape[1],
        )

    return _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
        out=out,
    )


def rocm_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> None:
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"ROCm Triton sparse prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_prefill",
    )

    if ragged_indices is not None and ragged_indptr is not None:
        direct_out = output if output.dtype == torch.bfloat16 else None
        output_chunk = _rocm_sparse_attn_prefill_ragged_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            out=direct_out,
        )
    else:
        indices_2d = indices.reshape(indices.shape[0], -1)
        output_chunk = _rocm_sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=indices_2d,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            topk_length=topk_length,
        )
    if output_chunk is not output:
        output.copy_(output_chunk.to(output.dtype))


def rocm_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    assert swa_k_cache.dtype == torch.uint8, (
        "ROCm Triton sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_decode",
    )

    main_indices = swa_indices.reshape(swa_indices.shape[0], -1)

    extra_cache = None
    extra_indices = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None or (
            topk_ragged_indices is not None and topk_ragged_indptr is not None
        )
        assert kv_cache.dtype == torch.uint8, (
            "ROCm Triton sparse decode expects uint8 fp8_ds_mla extra cache, "
            f"got {kv_cache.dtype}"
        )
        extra_cache = kv_cache
        if topk_indices is not None:
            extra_indices = topk_indices.reshape(topk_indices.shape[0], -1)

    _maybe_dump_sparse_decode_shape(
        q=q,
        kv_cache=kv_cache,
        swa_k_cache=swa_k_cache,
        swa_only=swa_only,
        topk_indices=topk_indices,
        topk_lens=topk_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_ragged_indices=swa_ragged_indices,
        swa_ragged_indptr=swa_ragged_indptr,
        topk_ragged_indices=topk_ragged_indices,
        topk_ragged_indptr=topk_ragged_indptr,
        output=output,
    )

    attn_out = _rocm_sparse_attn_decode_triton(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_indices,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        main_lengths=swa_lens,
        extra_lengths=topk_lens,
        main_ragged_indices=swa_ragged_indices,
        main_ragged_indptr=swa_ragged_indptr,
        extra_ragged_indices=topk_ragged_indices,
        extra_ragged_indptr=topk_ragged_indptr,
        out=(
            output
            if output.dtype == torch.bfloat16
            and _env_bool_default("DSV4_SPARSE_ATTN_DECODE_DIRECT_OUT_DEFAULT", False)
            else None
        ),
    )
    if attn_out is not output:
        output.copy_(attn_out.to(output.dtype))
