# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.deepseek_v4_ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="DeepSeek V4 cache layout kernels require a CUDA-like GPU",
)

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
QUANT_BLOCK = 64
SCALE_DIM = 8
FP8_MAX = torch.finfo(current_platform.fp8_dtype()).max
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2
HEAD_BYTES = TOKEN_DATA_BYTES + SCALE_DIM
SENTINEL = 0xA5


def _slot_to_offsets(slot: int, block_size: int, block_stride: int) -> tuple[int, int]:
    block_idx = slot // block_size
    pos_in_block = slot % block_size
    block_base = block_idx * block_stride
    token_base = block_base + pos_in_block * TOKEN_DATA_BYTES
    scale_base = block_base + block_size * TOKEN_DATA_BYTES + pos_in_block * SCALE_DIM
    return token_base, scale_base


def _ue8m0_quantize_nope(nope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized = torch.empty(
        NOPE_DIM, dtype=current_platform.fp8_dtype(), device="cuda"
    )
    encoded_scales = torch.zeros(SCALE_DIM, dtype=torch.uint8, device="cuda")

    for qblock_idx in range(NOPE_DIM // QUANT_BLOCK):
        start = qblock_idx * QUANT_BLOCK
        end = start + QUANT_BLOCK
        block = nope[start:end].float()
        amax = block.abs().max().clamp(min=1e-4)
        exponent = math.ceil(math.log2((amax / FP8_MAX).item()))
        scale = 2.0**exponent
        quantized[start:end] = (block / scale).clamp(-FP8_MAX, FP8_MAX).to(
            current_platform.fp8_dtype()
        )
        encoded_scales[qblock_idx] = exponent + 127

    return quantized.view(torch.uint8), encoded_scales


def _insert_reference_token(
    expected_cache: torch.Tensor,
    token: torch.Tensor,
    slot: int,
    block_size: int,
) -> None:
    cache_flat = expected_cache.view(torch.uint8).flatten()
    token_base, scale_base = _slot_to_offsets(
        slot, block_size, expected_cache.stride(0)
    )

    nope_bytes, scale_bytes = _ue8m0_quantize_nope(token[:NOPE_DIM])
    rope_bytes = token[NOPE_DIM:].contiguous().view(torch.uint8)

    cache_flat[token_base : token_base + NOPE_DIM].copy_(nope_bytes)
    cache_flat[token_base + NOPE_DIM : token_base + TOKEN_DATA_BYTES].copy_(rope_bytes)
    cache_flat[scale_base : scale_base + SCALE_DIM].copy_(scale_bytes)


def _build_reference_cache(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    expected_cache = torch.full(
        (num_blocks, block_size, HEAD_BYTES),
        SENTINEL,
        dtype=torch.uint8,
        device="cuda",
    )
    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot != -1:
            _insert_reference_token(expected_cache, k[token_idx], slot, block_size)
    return expected_cache


def _dequantize_reference_token(
    k_cache: torch.Tensor, slot: int, block_size: int
) -> torch.Tensor:
    cache_flat = k_cache.view(torch.uint8).flatten()
    token_base, scale_base = _slot_to_offsets(slot, block_size, k_cache.stride(0))

    nope_bytes = cache_flat[token_base : token_base + NOPE_DIM]
    nope_fp8 = nope_bytes.view(current_platform.fp8_dtype()).float()
    encoded_scales = cache_flat[scale_base : scale_base + SCALE_DIM]
    scales = torch.exp2(encoded_scales[:7].float() - 127.0)
    nope = nope_fp8 * scales.repeat_interleave(QUANT_BLOCK)

    rope_bytes = cache_flat[token_base + NOPE_DIM : token_base + TOKEN_DATA_BYTES]
    rope = rope_bytes.view(torch.bfloat16).float()
    return torch.cat([nope, rope]).to(torch.bfloat16)


def _dequantize_and_gather_reference(
    k_cache: torch.Tensor,
    seq_lens: list[int],
    gather_lens: list[int],
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> torch.Tensor:
    max_gather_len = max(gather_lens)
    out = torch.full(
        (len(seq_lens), offset + max_gather_len + 2, HEAD_DIM),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )

    block_table_host = block_table.cpu()
    for req_id, (seq_len, gather_len) in enumerate(zip(seq_lens, gather_lens)):
        start_pos = seq_len - gather_len
        for out_idx, pos in enumerate(range(start_pos, seq_len)):
            physical_block = int(block_table_host[req_id, pos // block_size].item())
            slot = physical_block * block_size + pos % block_size
            out[req_id, offset + out_idx] = _dequantize_reference_token(
                k_cache, slot, block_size
            )

    return out


def _make_synthetic_k(num_tokens: int) -> torch.Tensor:
    values = torch.arange(num_tokens * HEAD_DIM, dtype=torch.float32, device="cuda")
    values = ((values % 251) - 125) / 11.0
    k = values.reshape(num_tokens, HEAD_DIM)
    token_scale = torch.linspace(0.25, 2.0, num_tokens, device="cuda")[:, None]
    k = k * token_scale
    k[:, NOPE_DIM:] = torch.arange(
        num_tokens * ROPE_DIM, dtype=torch.float32, device="cuda"
    ).reshape(num_tokens, ROPE_DIM)
    return k.to(torch.bfloat16)


@pytest.mark.parametrize("block_size", [64, 256])
@torch.inference_mode()
def test_deepseek_v4_quantize_insert_matches_reference_byte_layout(
    block_size: int,
) -> None:
    num_blocks = 3
    slot_mapping_host = [
        -1,
        0,
        2 * block_size + 3,
        block_size - 1,
        -1,
        block_size + 7,
    ]
    slot_mapping = torch.tensor(slot_mapping_host, dtype=torch.int64, device="cuda")
    k = _make_synthetic_k(len(slot_mapping_host))
    actual_cache = torch.full(
        (num_blocks, block_size, HEAD_BYTES),
        SENTINEL,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_cache = _build_reference_cache(k, slot_mapping, num_blocks, block_size)

    quantize_and_insert_k_cache(
        k, actual_cache.view(num_blocks, -1), slot_mapping, block_size
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)

    cache_flat = actual_cache.flatten()
    expected_flat = expected_cache.flatten()
    for token_idx, slot in enumerate(slot_mapping_host):
        if slot == -1:
            assert torch.all(cache_flat == expected_flat)
            continue

        token_base, scale_base = _slot_to_offsets(
            slot, block_size, actual_cache.stride(0)
        )
        torch.testing.assert_close(
            cache_flat[token_base : token_base + NOPE_DIM],
            expected_flat[token_base : token_base + NOPE_DIM],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache_flat[token_base + NOPE_DIM : token_base + TOKEN_DATA_BYTES],
            k[token_idx, NOPE_DIM:].contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache_flat[scale_base : scale_base + SCALE_DIM],
            expected_flat[scale_base : scale_base + SCALE_DIM],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("block_size", [64, 256])
@torch.inference_mode()
def test_deepseek_v4_dequantize_gather_matches_reference_layout(
    block_size: int,
) -> None:
    seq_lens_host = [block_size + 5, block_size // 2 + 3]
    gather_lens_host = [7, 5]
    offset = 3
    num_reqs = len(seq_lens_host)
    num_blocks = 4

    block_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device="cuda")
    total_tokens = sum(seq_lens_host) + 3
    k = _make_synthetic_k(total_tokens)
    slot_mapping = torch.full((total_tokens,), -1, dtype=torch.int64, device="cuda")

    token_idx = 0
    for req_id, seq_len in enumerate(seq_lens_host):
        logical_pos = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        physical_blocks = block_table[req_id, logical_pos // block_size].to(torch.int64)
        slot_mapping[token_idx : token_idx + seq_len] = (
            physical_blocks * block_size + logical_pos % block_size
        )
        token_idx += seq_len

    k_cache = _build_reference_cache(k, slot_mapping, num_blocks, block_size)

    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor(gather_lens_host, dtype=torch.int32, device="cuda")
    actual = torch.empty(
        (num_reqs, offset + max(gather_lens_host) + 2, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = _dequantize_and_gather_reference(
        k_cache, seq_lens_host, gather_lens_host, block_table, block_size, offset
    )

    dequantize_and_gather_k_cache(
        actual, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )
    torch.cuda.synchronize()

    for req_id, gather_len in enumerate(gather_lens_host):
        actual_rows = actual[req_id, offset : offset + gather_len]
        expected_rows = expected[req_id, offset : offset + gather_len]
        torch.testing.assert_close(actual_rows, expected_rows, rtol=0, atol=0)
