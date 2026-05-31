# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import PplxGardenAll2AllHandle
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)

logger = init_logger(__name__)


def _trace_enabled() -> bool:
    return os.environ.get("VLLM_PPLX_GARDEN_TRACE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


class PplxGardenPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using PPLX Garden's CXI/RDMA P2P all-to-all.

    This first integration intentionally targets the GH200/CXI path we are
    benchmarking: unquantized activations, TP=1, and async dispatch/combine.
    """

    def __init__(
        self,
        handle: PplxGardenAll2AllHandle,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        self.handle = handle
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.num_local_experts = num_local_experts
        self._dispatch_handles: dict[int, object] = {}

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
        )
        hook()
        return receiver()

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable[[], None], mk.ReceiverType]:
        del expert_map, num_experts
        if quant_config.quant_dtype is not None and not defer_input_quant:
            raise NotImplementedError(
                "pplx_garden currently dispatches unquantized activations only."
            )
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        ubatch_id = dbo_current_ubatch_id()
        assert ubatch_id not in self._dispatch_handles, (
            f"stale PPLX Garden dispatch handle for ubatch {ubatch_id}"
        )
        debug_sync = os.environ.get("VLLM_PPLX_GARDEN_DEBUG_SYNC", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        trace = _trace_enabled()
        if trace:
            logger.warning(
                "PPLX trace prepare enter ubatch=%s a1=%s/%s topk=%s/%s "
                "capturing=%s stream=%s",
                ubatch_id,
                tuple(a1.shape),
                a1.dtype,
                tuple(topk_ids.shape),
                topk_ids.dtype,
                torch.cuda.is_current_stream_capturing(),
                torch.cuda.current_stream(a1.device).cuda_stream,
            )
        if debug_sync:
            logger.warning(
                "PPLX debug prepare entry sync start ubatch=%s capturing=%s stream=%s",
                ubatch_id,
                torch.cuda.is_current_stream_capturing(),
                torch.cuda.current_stream(a1.device).cuda_stream,
            )
            torch.cuda.synchronize(a1.device)
            logger.warning("PPLX debug prepare entry sync returned ubatch=%s", ubatch_id)
        original_topk_ids = topk_ids.to(torch.uint32).contiguous()
        original_topk_weights = (
            torch.ones_like(topk_weights)
            if apply_router_weight_on_input
            else topk_weights
        ).to(torch.float32).contiguous()
        if os.environ.get("VLLM_PPLX_GARDEN_UNIFORM_WEIGHTS", "0").lower() in (
            "1",
            "true",
            "yes",
        ):
            original_topk_weights.fill_(1.0 / original_topk_weights.size(1))

        expert_num_tokens = torch.empty(
            (self.num_local_experts,), dtype=torch.int32, device=a1.device
        )
        expert_x = torch.empty(
            (
                self.num_local_experts,
                self.handle.max_tokens_per_expert,
                a1.shape[1],
            ),
            dtype=a1.dtype,
            device=a1.device,
        )
        dp_x = a1.contiguous()

        logger.debug(
            "PPLX prepare_async start ubatch=%s a1=%s topk_ids=%s expert_x=%s",
            ubatch_id,
            tuple(dp_x.shape),
            tuple(original_topk_ids.shape),
            tuple(expert_x.shape),
        )
        if debug_sync:
            ids_cpu = original_topk_ids.detach().cpu()
            flat_ids = ids_cpu.flatten().to(torch.int64)
            sample = ids_cpu[:4].tolist()
            logger.warning(
                "PPLX debug routes ubatch=%s min=%s max=%s unique=%s "
                "weight_min=%s weight_max=%s weight_nan=%s sample=%s",
                ubatch_id,
                int(flat_ids.min().item()),
                int(flat_ids.max().item()),
                int(torch.unique(flat_ids).numel()),
                float(original_topk_weights.min().item()),
                float(original_topk_weights.max().item()),
                bool(torch.isnan(original_topk_weights).any().item()),
                sample,
            )
        if trace:
            logger.warning("PPLX trace dispatch_async call ubatch=%s", ubatch_id)
        dispatch_handle = self.handle.dispatch_async(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=None,
            dp_x=dp_x,
            dp_x_scale=None,
            indices=original_topk_ids,
            weights=original_topk_weights,
        )
        if trace:
            logger.warning("PPLX trace dispatch_async returned ubatch=%s", ubatch_id)
        logger.debug("PPLX prepare_async dispatch_async returned ubatch=%s", ubatch_id)
        if debug_sync:
            torch.cuda.synchronize(a1.device)
            logger.warning("PPLX debug prepare dispatch sync returned ubatch=%s", ubatch_id)
        self._dispatch_handles[ubatch_id] = dispatch_handle

        def recv_dispatch() -> None:
            if trace:
                logger.warning("PPLX trace dispatch recv call ubatch=%s", ubatch_id)
            logger.debug("PPLX dispatch recv start ubatch=%s", ubatch_id)
            dispatch_handle.recv()
            if trace:
                logger.warning("PPLX trace dispatch recv returned ubatch=%s", ubatch_id)
            logger.debug("PPLX dispatch recv returned ubatch=%s", ubatch_id)

        def receiver() -> mk.PrepareResultType:
            if trace:
                logger.warning(
                    "PPLX trace dispatch wait_recv_done call ubatch=%s", ubatch_id
                )
            logger.debug("PPLX dispatch wait_recv_done start ubatch=%s", ubatch_id)
            dispatch_handle.wait_recv_done()
            if trace:
                logger.warning(
                    "PPLX trace dispatch wait_recv_done returned ubatch=%s", ubatch_id
                )
            logger.debug("PPLX dispatch wait_recv_done returned ubatch=%s", ubatch_id)
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
            )
            return (
                expert_x,
                None,
                expert_tokens_meta,
                None,
                None,
            )

        return recv_dispatch, receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        hook, receiver = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        hook()
        receiver()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable[[], None], Callable[[], None]]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the PPLX Garden "
            "combine kernel."
        )
        del apply_router_weight_on_input, topk_weights, topk_ids
        ubatch_id = dbo_current_ubatch_id()
        assert ubatch_id in self._dispatch_handles
        dispatch_handle = self._dispatch_handles[ubatch_id]

        if fused_expert_output.ndim == 3:
            assert fused_expert_output.shape[0] == self.num_local_experts
            assert fused_expert_output.shape[1] == self.handle.max_tokens_per_expert

        expert_y_send = fused_expert_output.to(dtype=output.dtype).contiguous()
        try:
            trace = _trace_enabled()
            if trace:
                logger.warning(
                    "PPLX trace finalize enter ubatch=%s fused=%s/%s output=%s/%s "
                    "capturing=%s stream=%s",
                    ubatch_id,
                    tuple(fused_expert_output.shape),
                    fused_expert_output.dtype,
                    tuple(output.shape),
                    output.dtype,
                    torch.cuda.is_current_stream_capturing(),
                    torch.cuda.current_stream(output.device).cuda_stream,
                )
            logger.debug(
                "PPLX finalize_async start ubatch=%s fused=%s/%s stride=%s "
                "send=%s/%s stride=%s output=%s/%s",
                ubatch_id,
                tuple(fused_expert_output.shape),
                fused_expert_output.dtype,
                fused_expert_output.stride(),
                tuple(expert_y_send.shape),
                expert_y_send.dtype,
                expert_y_send.stride(),
                tuple(output.shape),
                output.dtype,
            )
            if os.environ.get("VLLM_PPLX_GARDEN_DEBUG_SYNC", "0").lower() in (
                "1",
                "true",
                "yes",
            ):
                torch.cuda.synchronize(output.device)
                counts = dispatch_handle.out_expert_num_tokens
                logger.warning(
                    "PPLX debug before combine ubatch=%s token_sum=%s token_max=%s "
                    "capturing=%s stream=%s",
                    ubatch_id,
                    int(counts.sum().item()),
                    int(counts.max().item()),
                    torch.cuda.is_current_stream_capturing(),
                    torch.cuda.current_stream(output.device).cuda_stream,
                )
            if trace:
                logger.warning("PPLX trace maybe_run_recv_hook call ubatch=%s", ubatch_id)
            dbo_maybe_run_recv_hook()
            if trace:
                logger.warning(
                    "PPLX trace maybe_run_recv_hook returned ubatch=%s", ubatch_id
                )
                logger.warning("PPLX trace combine_async call ubatch=%s", ubatch_id)
            combine_handle = self.handle.combine_async(
                out_tokens=output,
                dispatch_handle=dispatch_handle,
                expert_y=expert_y_send,
            )
            if trace:
                logger.warning("PPLX trace combine_async returned ubatch=%s", ubatch_id)
            logger.debug("PPLX finalize_async combine_async returned ubatch=%s", ubatch_id)
            if trace:
                logger.warning("PPLX trace combine recv call ubatch=%s", ubatch_id)
            logger.debug("PPLX combine recv start ubatch=%s", ubatch_id)
            combine_handle.recv()
            if trace:
                logger.warning("PPLX trace combine recv returned ubatch=%s", ubatch_id)
            logger.debug("PPLX combine recv returned ubatch=%s", ubatch_id)
            if os.environ.get("VLLM_PPLX_GARDEN_SERIAL_COMBINE", "0").lower() in (
                "1",
                "true",
                "yes",
            ):
                logger.warning("PPLX serial combine wait start ubatch=%s", ubatch_id)
                combine_handle.wait_recv_done()
                torch.cuda.current_stream(output.device).synchronize()
                logger.warning("PPLX serial combine wait returned ubatch=%s", ubatch_id)
                self._dispatch_handles.pop(ubatch_id, None)
                return lambda: None, lambda: None
        except Exception:
            self._dispatch_handles.pop(ubatch_id, None)
            raise

        def recv_combine() -> None:
            # The PPLX handle needs its receive posted promptly after
            # combine_async. Keep a hook so DBO still gets an overlap yield
            # before waiting for completion.
            return None

        def receiver() -> None:
            if trace:
                logger.warning("PPLX trace combine wait_recv_done call ubatch=%s", ubatch_id)
            logger.debug("PPLX combine wait_recv_done start ubatch=%s", ubatch_id)
            combine_handle.wait_recv_done()
            if trace:
                logger.warning(
                    "PPLX trace combine wait_recv_done returned ubatch=%s", ubatch_id
                )
            logger.debug("PPLX combine wait_recv_done returned ubatch=%s", ubatch_id)
            if not dbo_enabled() and not torch.cuda.is_current_stream_capturing():
                # PPLX slots are reusable only after the combine work has really
                # completed. A CUDA event wait only orders later GPU work; during
                # profiling we can otherwise enqueue the next layer's dispatch
                # while the previous P2P slot is still live.
                torch.cuda.current_stream(output.device).synchronize()
                logger.debug("PPLX combine synchronized ubatch=%s", ubatch_id)
            self._dispatch_handles.pop(ubatch_id, None)

        return recv_combine, receiver
