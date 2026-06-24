# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Breakable CUDA graph runner for the prefill (extend) inner-model forward.

Owns the breakable graph(s) for one inner ``*Model.forward``. Installed on a
:class:`BaseCausalLM` as ``prefill_graph_runner``; the model's ``forward`` calls
:meth:`maybe_run` and falls back to the eager inner forward when it returns
``None``.

The captured region is purely token-shaped compute (attention runs as an eager
break, see :mod:`tokenspeed.runtime.execution.breakable_cuda_graph`), so one graph
per padded token bucket serves any batch size at that token count. This nests
*inside* the existing eager prefill path: the decode ``CudaGraphWrapper`` rejects
extend mode and runs ``_forward_step`` eagerly, having already set the extend
attention metadata -- so the eager attention break here reads live metadata, and
the eager logits/sampling tail runs unchanged after.

Capture happens up front (one breakable graph per token bucket at startup, like
the decode graph), driven by a dummy bs=1 extend batch per bucket; serving only
replays.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_prefill_token_buckets
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
    from tokenspeed.runtime.models.base.transformer_model import BaseTransformerModel


class PrefillGraphRunner:
    """Drives a breakable prefill graph for one inner transformer model.

    Args:
        inner_model: The token-shaped inner ``*Model.forward`` callable (returns
            ``(hidden_states, aux_hidden_states)``).
        input_buffers: The shared static input buffers the graph reads from.
        config: Model-executor config (token-bucket sizing + ``disable_prefill_graph``).
        model_is_mrope: Whether positions use the 3-row mrope buffer.
        pool: Optional CUDA mempool id to share with the decode graph.
        enabled: Master switch (e.g. disabled under ``enforce_eager``).
        supports_mixed: Whether mixed (prefill+decode) batches may use the graph.
            Only models whose attention break reads the live prefill/decode split
            from the singleton backend (MLA) are correct on mixed; others restrict
            the graph to pure-extend batches.
    """

    def __init__(
        self,
        inner_model: "BaseTransformerModel",
        input_buffers: "InputBuffers",
        config: "ModelExecutorConfig",
        *,
        model_is_mrope: bool,
        pool=None,
        enabled: bool = True,
        supports_mixed: bool = False,
        num_warmup: int = 3,
    ) -> None:
        self.inner_model = inner_model
        self.input_buffers = input_buffers
        self.model_is_mrope = model_is_mrope
        self.supports_mixed = supports_mixed
        self.num_warmup = num_warmup
        self.world_size = config.world_size
        self.dp_size = config.data_parallel_size
        self._buckets = get_prefill_token_buckets(config)
        self.enabled = (
            enabled and not config.disable_prefill_graph and bool(self._buckets)
        )
        self._ctx: "ForwardContext | None" = None
        self._pool = pool
        self._engaged_logged = False
        # One captured breakable graph per padded token bucket (captured up front
        # in capture()) plus its (bucket-sized) output, all sharing one mempool.
        self._captures: dict[int, BreakableCapture] = {}
        self._outputs: dict[int, tuple] = {}

    def capture(self, make_dummy_batch: "Callable[[int], ForwardContext]") -> None:
        """Capture one breakable graph per token bucket, up front (like decode).

        ``make_dummy_batch(bucket)`` populates the static input buffers for a bs=1,
        ``bucket``-token extend forward, sets the eager attention metadata, and
        returns the dummy ``ForwardContext``. Largest bucket first so every bucket
        shares the first-captured mempool.
        """
        if not self.enabled:
            return
        for bucket in sorted(self._buckets, reverse=True):
            self._ctx = make_dummy_batch(bucket)
            try:
                self._capture_bucket(bucket)
            finally:
                self._ctx = None
        sample = next(iter(self._captures.values()), None)
        logger.info(
            "prefill breakable graph: captured buckets %s (segments=%d, eager "
            "attention breaks)",
            sorted(self._captures),
            sample.num_segments if sample is not None else 0,
        )

    def maybe_run(self, ctx: "ForwardContext"):
        """Replay the prefill graph for ``ctx``, or return ``None`` to run eager.

        Returns ``(hidden_states, aux_hidden_states)`` sliced to the real token
        count, or ``None`` when this forward is ineligible / its bucket was never
        captured (caller runs the eager inner model).
        """
        if not self._eligible(ctx):
            return None
        n = ctx.input_num_tokens

        # Under data parallelism the MoE expert-parallel all-to-all is a collective
        # across ALL ranks, sized from a replicated per-rank token list. The captured
        # graph bakes a uniform ``[bucket]*world_size`` layout, so every rank must
        # replay the SAME bucket or the collective desyncs (NCCL deadlock). Decide
        # purely from replicated global state -- the all-extend flag and the global
        # max token count -- so all ranks reach the identical decision/bucket with no
        # extra sync (mirrors the decode graph). Idle ranks run a DECODE forward, so
        # ``all_extend`` is False whenever any rank is idle and the graph stays off
        # (e.g. warmup), correctly falling back to eager.
        dp = self.dp_size > 1 and ctx.global_num_tokens is not None
        if dp:
            if not ctx.all_extend:
                return None
            bucket = self._padded_bucket(max(ctx.global_num_tokens))
        else:
            bucket = self._padded_bucket(n)
        if bucket is None or bucket not in self._captures:
            return None

        if not self._engaged_logged:
            self._engaged_logged = True
            logger.info(
                "prefill breakable graph ENGAGED: bucket=%d dp=%s mode=%s "
                "(mixed prefill+decode batches supported)",
                bucket,
                dp,
                ctx.forward_mode,
            )

        # Replay over `bucket` (padded) tokens; attention metadata stays at the
        # real `n` (set upstream), so the eager attention break only touches real
        # tokens and the padded rows produce discarded garbage. Under DP also pin
        # global_num_tokens/global_bs to the captured uniform layout so any live
        # read during the break matches the baked EP shapes.
        # Publish the LIVE prefill/decode token split on the (singleton) attn backend
        # so the eager attention break can recover it at replay: the break closes
        # over the dummy ``ctx`` captured at startup, so its ``ctx.bs`` /
        # ``ctx.num_extends`` are stale -- but ``ctx.attn_backend`` is the live
        # singleton. Models that split prefill vs decode inside attention (MLA) read
        # this; others ignore it. ``n`` is the real total token count (pre-padding).
        spec = getattr(ctx.attn_backend, "spec_num_tokens", 1) or 1
        num_decode_tokens = max(ctx.bs - ctx.num_extends, 0) * spec
        num_prefill_tokens = n - num_decode_tokens
        self._ctx = ctx
        saved = ctx.input_num_tokens
        saved_global_num_tokens = ctx.global_num_tokens
        saved_global_bs = ctx.global_bs
        ctx.attn_backend.prefill_graph_token_split = (
            num_prefill_tokens,
            num_decode_tokens,
        )
        ctx.input_num_tokens = bucket
        if dp:
            ctx.global_num_tokens = [bucket] * self.world_size
            ctx.global_bs = [1] * self.world_size
        try:
            self._captures[bucket].replay()
            hidden_states, aux_hidden_states = self._outputs[bucket]
        finally:
            ctx.input_num_tokens = saved
            ctx.global_num_tokens = saved_global_num_tokens
            ctx.global_bs = saved_global_bs
            ctx.attn_backend.prefill_graph_token_split = None
            self._ctx = None

        hidden_states = hidden_states[:n]
        if aux_hidden_states is not None:
            aux_hidden_states = [a[:n] for a in aux_hidden_states]
        return hidden_states, aux_hidden_states

    # -- internals ---------------------------------------------------------

    def _eligible(self, ctx: "ForwardContext") -> bool:
        if not self.enabled or ctx.forward_mode is None:
            return False
        # Accept pure-extend AND mixed (extend+decode) batches, as long as there is
        # a prefill portion. Mixed batches work because the attention runs as a
        # single eager break that dispatches the prefill/decode split per-replay
        # (see the coarse break + live split below); the captured token-shaped
        # compute is uniform over all rows. Pure-decode (num_extends == 0) is served
        # by the decode graph, not here.
        if ctx.num_extends <= 0:
            return False
        is_mixed = ctx.forward_mode.is_mixed()
        if not (ctx.forward_mode.is_extend() or is_mixed):
            return False
        # Mixed batches are only correct under the graph for models whose attention
        # break recovers the LIVE prefill/decode split from the singleton backend
        # (MLA). Other families read the split/mode from the captured (stale) ctx,
        # so route their mixed batches to eager.
        if is_mixed and not self.supports_mixed:
            return False
        # No prefix caching in the captured path (v1): a non-zero prefix changes the
        # ragged attention layout. Checked on the pinned CPU mirror (no D2H sync).
        prefix_cpu = self.input_buffers.extend_prefix_lens_cpu[: ctx.num_extends]
        if ctx.num_extends > 0 and bool(prefix_cpu.any().item()):
            return False
        return True

    def _padded_bucket(self, num_tokens: int) -> int | None:
        """Smallest bucket >= ``num_tokens``, or ``None`` if over the largest."""
        idx = bisect.bisect_left(self._buckets, num_tokens)
        return self._buckets[idx] if idx < len(self._buckets) else None

    def _run_inner(self, num_tokens: int):
        """Run the inner model over the leading ``num_tokens`` of the static buffers.

        ``num_tokens`` is the padded bucket size; the padded tail [real:bucket] is
        already scrubbed to safe values (input_ids=1, positions=0,
        out_cache_loc=dummy_kv_slot) by ``InputBuffers.fill_input_buffers``.
        """
        ib = self.input_buffers
        if self.model_is_mrope:
            positions = ib.mrope_positions_buf[:, :num_tokens]
        else:
            positions = ib.positions_buf[:num_tokens]
        return self.inner_model(
            ib.input_ids_buf[:num_tokens],
            positions,
            self._ctx,
            ib.out_cache_loc_buf[:num_tokens],
        )

    def _capture_bucket(self, bucket: int) -> None:
        """Warm up and capture the breakable graph for ``bucket`` from the buffers."""
        for _ in range(self.num_warmup):
            self._run_inner(bucket)
        torch.cuda.synchronize()
        cap = BreakableCapture(pool=self._pool)
        with cap:
            self._outputs[bucket] = self._run_inner(bucket)
        if self._pool is None:
            self._pool = cap.pool  # share the pool across all subsequent buckets
        cap.replay()  # capture records kernels without executing; smoke-test replay
        self._captures[bucket] = cap
