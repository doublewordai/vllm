# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed-weight storage for transformer layers.

Stores per-transformer-layer parameter data compressed in VRAM (nvcomp
deflate, which auto-selects Blackwell's hardware decompression engine
on capable GPUs) and decompresses it on-demand into a double-buffered
staging area.

Enabled with ``VLLM_COMPRESS_WEIGHTS=1``.

Key implementation details
--------------------------

1. **Direct C-API bindings are not usable in nvcomp 5.x.** The flat
   ``nvcompBatchedDeflateCompressGetTempSizeSync``/``CompressAsync``
   symbols exist in the shared library but return ``NVCOMP_NOT_SUPPORTED``
   (status 11) at runtime — the supported path for nvcomp 5.x goes
   through the ``DeflateManager`` C++ class which the Python ``Codec``
   wraps. We therefore use the Python ``Codec`` instead.

2. **nvcomp's workspace lives in torch's caching allocator.** nvcomp
   exposes ``set_device_allocator()`` which accepts a user-provided
   allocator callable. We install one that returns a ``torch.empty``
   tensor. This puts every byte nvcomp internally allocates into
   torch's pool, so vLLM's memory profiler counts it as torch-owned
   weights (which we later report as the post-compression
   ``model_memory_usage``) rather than as ``non_torch_increase`` —
   the latter is subtracted directly from the KV cache budget, so
   without this step ~6 GB of nvcomp workspace directly reduces KV
   cache capacity.

3. **Decode is wrapped as a ``torch.library.custom_op``.** This makes
   the decode call dynamo-friendly (opaque black box, not traced
   through) and CUDA-graph-captureable (the underlying kernel launches
   are ordinary async stream operations). Forward pre-hooks on each
   transformer layer invoke the op with pre-computed per-layer
   metadata. Without this wrapper, dynamo fails with "cannot trace
   through pybind function" and CUDA graph capture aborts.

4. **Per-layer parameter storage is rebound to views into a double-
   buffered staging area.** Before layer N runs, decode N writes into
   ``staging[N % 2]``; layer N's params already point at that slice.

Caveats
-------
- Requires nvcomp >= 5.x.
- No decode/compute pipelining yet. Each layer's hook serially
  decompresses that layer's weights before the layer runs.
- Only per-transformer-layer parameters are compressed. Embeddings,
  ``lm_head``, and top-level norms are left uncompressed.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_CHUNK_BYTES = 65536


# ---------------------------------------------------------------------------
# Custom torch allocator for nvcomp's internal buffers
# ---------------------------------------------------------------------------

# We hold strong refs to the underlying torch tensors while nvcomp's buffers
# are in use. The _TorchBuffer objects are what nvcomp sees; the .tensor is
# what keeps the storage alive. nvcomp accesses .ptr (the raw device pointer).
class _TorchBuffer:
    """nvcomp-compatible buffer backed by a torch-owned uint8 tensor.

    nvcomp's ``set_device_allocator`` expects an object with a ``ptr``
    attribute (raw device pointer as an int). Storing the backing
    tensor keeps its memory alive until this object is GCed.
    """

    __slots__ = ("tensor", "ptr")

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.ptr = tensor.data_ptr()


def _torch_device_allocator(nbytes: int, stream):  # noqa: ANN001
    # `stream` is an nvcomp.CudaStream; nvcomp will free-on-GC, so we
    # just need a torch tensor of the right size. The torch caching
    # allocator handles everything.
    tensor = torch.empty(int(nbytes), dtype=torch.uint8, device="cuda")
    return _TorchBuffer(tensor)


_ALLOCATOR_INSTALLED = False


def _ensure_torch_allocator() -> None:
    global _ALLOCATOR_INSTALLED
    if _ALLOCATOR_INSTALLED:
        return
    import nvidia.nvcomp as nvcomp

    nvcomp.set_device_allocator(_torch_device_allocator)
    _ALLOCATOR_INSTALLED = True
    logger.info(
        "compressed_weights: nvcomp device allocator routed through "
        "torch.empty (workspace counted as weights_memory)"
    )


# ---------------------------------------------------------------------------
# Custom op wrapping nvcomp decompression
# ---------------------------------------------------------------------------

_DECODE_OP_REGISTERED = False


def _register_decode_op() -> None:
    global _DECODE_OP_REGISTERED
    if _DECODE_OP_REGISTERED:
        return

    # Lazy import to avoid ImportError in non-CUDA environments at module load
    import nvidia.nvcomp as nvcomp

    @torch.library.custom_op(
        "dw::decode_layer",
        mutates_args=("staging",),
        device_types="cuda",
    )
    def _impl(
        compressed: torch.Tensor,   # uint8 compressed bytes (torch-owned)
        staging: torch.Tensor,      # same dtype as parameters, output (mutated)
        layer_bytes: int,           # active byte count (== sum of param sizes)
    ) -> None:
        # Codec is resolved at call time from module-level storage set by
        # the manager; the custom op schema cannot carry arbitrary Python
        # objects as arguments.
        codec = _codec_handle
        if codec is None:
            raise RuntimeError(
                "dw::decode_layer called before compressed_weights was "
                "installed"
            )
        src_arr = nvcomp.from_dlpack(compressed)
        # nvcomp's Array computes capacity as shape[0] regardless of
        # dtype (it treats any DLPack-imported buffer as raw bytes
        # counted in elements). Feed it a uint8 view of the staging
        # tensor's bytes so capacity == actual byte count. The
        # .view(torch.uint8) is internal to this op's implementation,
        # so aot_autograd does not see a different-dtype view of the
        # same storage at the custom-op boundary — the op signature
        # still takes a ``staging`` arg of the parameter dtype.
        staging_u8 = staging.view(torch.uint8)
        out_view = staging_u8.narrow(0, 0, layer_bytes)
        out_arr = nvcomp.from_dlpack(out_view)
        codec.decode(src_arr, out=out_arr)

    @_impl.register_fake
    def _fake(compressed, staging, layer_bytes):
        return None

    _DECODE_OP_REGISTERED = True


# Module-level handle to the codec used by the custom op. Set by
# CompressedLayerWeights.__init__; None before installation.
_codec_handle = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


def _find_transformer_layers(model: nn.Module) -> tuple[str, nn.ModuleList] | None:
    known = (
        "language_model.model.layers",
        "model.language_model.model.layers",
        "model.layers",
        "layers",
    )
    for path in known:
        obj: nn.Module | None = model
        for part in path.split("."):
            if obj is None or not hasattr(obj, part):
                obj = None
                break
            obj = getattr(obj, part)
        if isinstance(obj, nn.ModuleList) and len(obj) >= 4:
            return path, obj
    best: tuple[str, nn.ModuleList] | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            if best is None or len(module) > len(best[1]):
                best = (name, module)
    return best


@dataclass
class _LayerEntry:
    idx: int
    compressed: torch.Tensor    # torch-owned uint8, compressed bytes
    total_bytes: int            # uncompressed size


class CompressedLayerWeights:
    """Compress per-layer transformer weights, rebind ``.data`` to
    views into a double-buffered staging area, install forward pre-hooks
    that invoke ``dw::decode_layer``.
    """

    def __init__(self, model: nn.Module, layers: nn.ModuleList):
        import nvidia.nvcomp as nvcomp
        from nvidia.nvcomp import Codec

        # Route nvcomp's internal allocations through torch, then build
        # the codec. Without the allocator set BEFORE codec construction,
        # nvcomp allocates workspace up-front from its own pool.
        _ensure_torch_allocator()
        _register_decode_op()

        self._codec = Codec(
            algorithm="deflate",
            uncomp_chunk_size=_CHUNK_BYTES,
            algorithm_type=0,  # entropy-only; smallest workspace at full ratio
        )
        global _codec_handle
        _codec_handle = self._codec

        self.model = model
        self.layers = layers
        self.num_layers = len(layers)
        self.decompress_calls = 0

        # Determine the dominant parameter dtype by total bytes, and
        # only compress parameters of that dtype. Others are left in
        # place. aot_autograd refuses "input mutations on views with
        # different dtypes" for the same storage, so every parameter we
        # rebind must view into a staging buffer of matching dtype — the
        # simplest way to satisfy that is to only compress one dtype at
        # a time. (On Qwen3.5-27B, bf16 dominates at ~45 GB vs a few GB
        # of float32 Mamba state.)
        dtype_bytes: dict[torch.dtype, int] = {}
        for layer in layers:
            for _, p in layer.named_parameters():
                b = p.nelement() * p.element_size()
                dtype_bytes[p.dtype] = dtype_bytes.get(p.dtype, 0) + b
        self._staging_dtype = max(dtype_bytes, key=lambda d: dtype_bytes[d])
        skipped_bytes = sum(b for d, b in dtype_bytes.items()
                            if d != self._staging_dtype)
        if skipped_bytes > 0:
            logger.info(
                "CompressedLayerWeights: compressing only %s params "
                "(%.2f GB); leaving %s other-dtype params (%.2f GB) "
                "uncompressed: %s",
                self._staging_dtype,
                dtype_bytes[self._staging_dtype] / 1024**3,
                len(dtype_bytes) - 1,
                skipped_bytes / 1024**3,
                {str(d): f"{b/1024**3:.2f} GB"
                 for d, b in dtype_bytes.items() if d != self._staging_dtype},
            )

        # Per-layer metadata: only parameters of the target dtype are
        # enrolled. Byte offsets are relative to each layer's packed
        # buffer of compressible params.
        self._per_layer: list[dict] = []
        for idx, layer in enumerate(layers):
            params = []
            offset = 0
            for pname, p in layer.named_parameters():
                if p.dtype != self._staging_dtype:
                    continue
                byte_size = p.nelement() * p.element_size()
                params.append({
                    "name": pname,
                    "param": p,
                    "dtype": p.dtype,
                    "shape": tuple(p.shape),
                    "byte_offset": offset,
                    "byte_size": byte_size,
                })
                offset += byte_size
            self._per_layer.append(
                {"idx": idx, "params": params, "total_bytes": offset}
            )

        if self.num_layers == 0 or not any(L["total_bytes"] > 0
                                           for L in self._per_layer):
            raise ValueError(
                "compressed_weights: no parameters of the dominant dtype "
                "in the supplied ModuleList; refusing to install."
            )

        self._max_layer_bytes = max(L["total_bytes"] for L in self._per_layer)
        total_layer_bytes = sum(L["total_bytes"] for L in self._per_layer)
        logger.info(
            "CompressedLayerWeights: %d layers, per-layer bytes "
            "min=%.1f MB max=%.1f MB, total %.2f GB (compressible)",
            self.num_layers,
            min(L["total_bytes"] for L in self._per_layer) / 1024**2,
            self._max_layer_bytes / 1024**2,
            total_layer_bytes / 1024**3,
        )
        staging_elem_size = torch.tensor([], dtype=self._staging_dtype).element_size()
        if self._max_layer_bytes % staging_elem_size != 0:
            raise AssertionError("staging size not aligned to element size")
        staging_elems = self._max_layer_bytes // staging_elem_size

        self._staging = [
            torch.empty(staging_elems, dtype=self._staging_dtype, device="cuda")
            for _ in range(2)
        ]
        logger.info(
            "CompressedLayerWeights: staging 2 × %.1f MB (%s) = %.1f MB",
            self._max_layer_bytes / 1024**2,
            self._staging_dtype,
            2 * self._max_layer_bytes / 1024**2,
        )

        # Compress each layer: pack bytes, encode, store torch-owned
        # uint8 tensor of actual compressed size, free originals.
        _empty: dict[torch.dtype, torch.Tensor] = {}

        def empty_of(dt: torch.dtype) -> torch.Tensor:
            if dt not in _empty:
                _empty[dt] = torch.empty(0, device="cuda", dtype=dt)
            return _empty[dt]

        self._entries: list[_LayerEntry] = []
        total_orig = 0
        total_comp = 0
        for idx, L in enumerate(self._per_layer):
            chunks = [
                info["param"].detach().contiguous().view(torch.uint8).flatten()
                for info in L["params"]
            ]
            packed = torch.cat(chunks).contiguous()
            del chunks
            for info in L["params"]:
                info["param"].data = empty_of(info["dtype"])

            arr = nvcomp.from_dlpack(packed)
            comp = self._codec.encode(arr)
            torch.cuda.synchronize()
            actual_size = comp.shape[0]
            full_view = torch.from_dlpack(comp)
            compressed = full_view[:actual_size].clone().contiguous()  # torch-owned

            self._entries.append(_LayerEntry(
                idx=idx,
                compressed=compressed,
                total_bytes=L["total_bytes"],
            ))
            total_orig += packed.nelement()
            total_comp += int(compressed.nelement())
            del packed, arr, comp, full_view

            if idx % 4 == 3 or idx == self.num_layers - 1:
                gc.collect()
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        ratio = total_orig / total_comp if total_comp else 1.0
        self.total_orig_bytes = total_orig
        self.total_comp_bytes = total_comp
        logger.info(
            "CompressedLayerWeights: %.2f GB → %.2f GB "
            "(ratio %.3fx, saved %.2f GB)",
            total_orig / 1024**3,
            total_comp / 1024**3,
            ratio,
            (total_orig - total_comp) / 1024**3,
        )

        # Rewire parameter .data into staging slot views. Staging is already
        # the target dtype (see dtype check above), so we just need to take
        # a narrow() at the element offset and reshape — no dtype reinterpret,
        # which would otherwise upset aot_autograd.
        for L in self._per_layer:
            buf = self._staging[L["idx"] % 2]
            for info in L["params"]:
                assert info["byte_offset"] % staging_elem_size == 0
                elem_off = info["byte_offset"] // staging_elem_size
                elem_len = info["byte_size"] // staging_elem_size
                info["param"].data = buf.narrow(0, elem_off, elem_len).view(*info["shape"])

        # Prime staging for layers 0 and 1
        self._decode_now(0)
        if self.num_layers > 1:
            self._decode_now(1)
        torch.cuda.synchronize()

        # Install forward pre-hooks
        for idx, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._make_hook(idx))
        logger.info(
            "CompressedLayerWeights: installed pre-hooks on %d layers",
            self.num_layers,
        )

    def _decode_now(self, idx: int) -> None:
        entry = self._entries[idx]
        staging = self._staging[idx % 2]
        torch.ops.dw.decode_layer(
            entry.compressed,
            staging,
            entry.total_bytes,
        )
        self.decompress_calls += 1

    def _make_hook(self, idx: int):
        def hook(module, inputs):  # noqa: ANN001 ARG001
            self._decode_now(idx)

        return hook


def install_compressed_weights(
    model: nn.Module,
) -> CompressedLayerWeights | None:
    found = _find_transformer_layers(model)
    if found is None:
        logger.warning(
            "VLLM_COMPRESS_WEIGHTS=1 requested but no transformer-layer "
            "ModuleList found; compression disabled."
        )
        return None
    path, layers = found
    logger.info(
        "CompressedLayerWeights: using layer stack at model.%s (%d layers)",
        path, len(layers),
    )
    return CompressedLayerWeights(model, layers)
