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

"""Budget-bucketed CUDA graph capture/replay for vision encoders.

Vision-encoder analogue of the LM :class:`CudaGraphWrapper`. Capture-safety invariants
(violating any of these is silent numerical corruption):
  * each budget graph gets its own private pool -- a shared pool collides the
    vision-TP custom-AR IPC buffer registrations across budgets;
  * ``max_seqlen`` is baked at the per-budget worst case (single image filling
    the whole budget); it is frozen at capture, not refreshed on replay;
  * the captured region is attention-backend agnostic -- the tower supplies a
    capture-safe ``forward_blocks``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable

import torch

from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.utils import logger


@dataclass
class BudgetGraphMetadata:
    """One captured budget graph. Replay: zero + slice-copy into
    ``input_buffer`` / ``metadata_buffers``, ``graph.replay()``, read
    ``output_buffer``."""

    graph: torch.cuda.CUDAGraph
    input_buffer: torch.Tensor
    metadata_buffers: dict[str, torch.Tensor]
    output_buffer: torch.Tensor


@dataclass
class RaggedImageBatch:
    """Per-image patch rows concatenated on dim 0, indexed by ``grid``.
    Wrapper-internal. ``.tolist()`` syncs are confined here, in the eager
    region — never reached from inside a graph replay."""

    tokens: torch.Tensor
    grid: torch.Tensor
    out_div: int

    @cached_property
    def _grid_rows(self) -> list[list[int]]:
        return self.grid.tolist()

    def num_items(self) -> int:
        return self.grid.shape[0]

    @cached_property
    def output_tokens(self) -> list[int]:
        return [(t * h * w) // self.out_div for t, h, w in self._grid_rows]

    @cached_property
    def cu_input(self) -> list[int]:
        cu = [0]
        for t, h, w in self._grid_rows:
            cu.append(cu[-1] + t * h * w)
        return cu

    def select(self, indices: list[int]) -> "RaggedImageBatch":
        """Sub-batch at ``indices``, preserving order; gathers patch rows by
        cumulative offset."""
        cu = self.cu_input
        if indices:
            rows = torch.cat(
                [
                    torch.arange(cu[i], cu[i + 1], device=self.tokens.device)
                    for i in indices
                ]
            )
        else:
            rows = torch.zeros(0, dtype=torch.long, device=self.tokens.device)
        return RaggedImageBatch(self.tokens[rows], self.grid[indices], self.out_div)


class EncoderCudaGraphWrapper:
    """Wraps the model's image encode; dispatches between a captured CUDA graph
    (budget-bucketed, greedy-packed) and an eager fallback (single image
    exceeds the largest budget). Mirrors the LM-side :class:`CudaGraphWrapper`.

    Built by the model via ``model.make_encoder_cudagraph_wrapper(mapping)``
    and installed by the executor as ``model.image_encoder``. Init resolves
    device/dtype/budgets at construction; capture is lazy on first call.

    Args:
        mapping: distributed Mapping (vision tp_group for custom-AR capture).
        tower: capture-safe sub-module exposing
            ``prepare_metadata(grid) -> dict`` (eager; owns all GPU->CPU syncs)
            and ``forward_blocks(tokens, metadata) -> Tensor`` (graph body).
        pre_encode: ``(items) -> (tokens, grid)`` eager bookend (patch embed).
        post_encode: ``(outs, grid) -> features`` eager bookend (merge/proj).
        out_div: spatial-merge unit; output tokens = ``(t*h*w) // out_div``.
        merge: side length the synthetic capture grid's ``h, w`` is divisible
            by.
        budget_range: ``(min, max)`` output-token budgets, power-of-2 bucketed.
        input_feature_shape: per-token feature dims of the capture input
            buffer (excludes leading token axis), e.g. ``(1, h)`` or ``(h,)``.
        out_squeeze_dim: squeeze this dim of ``forward_blocks`` output before
            per-item slicing (towers that keep a leading batch dim 1).
    """

    def __init__(
        self,
        mapping: Mapping,
        tower: Any,
        pre_encode: Callable[[list[Any]], tuple[torch.Tensor, torch.Tensor]],
        post_encode: Callable[[list[torch.Tensor], torch.Tensor], torch.Tensor],
        out_div: int,
        merge: int,
        budget_range: tuple[int, int],
        input_feature_shape: tuple[int, ...],
        out_squeeze_dim: int | None = None,
    ):
        self.mapping = mapping
        self.tower = tower
        self._pre_encode = pre_encode
        self._post_encode = post_encode
        self._out_div = out_div
        self._merge = merge
        self._input_feature_shape = input_feature_shape
        self._out_squeeze_dim = out_squeeze_dim

        param = next(tower.parameters())
        self.device = param.device
        self.dtype = param.dtype

        min_budget, max_budget = budget_range
        self.token_budgets = self._generate_budgets(min_budget, max_budget)
        self.max_batch_size = max(1, max_budget // max(1, min_budget))

        self.budget_graphs: dict[int, BudgetGraphMetadata] = {}

        logger.info(
            "EncoderCudaGraphWrapper initialized: budgets=%s, max_batch_size=%d, "
            "vision_tp=%d",
            self.token_budgets,
            self.max_batch_size,
            mapping.vision.tp_size,
        )

    def __call__(self, items: list[Any]) -> torch.Tensor:
        tokens, grid = self._pre_encode(items)
        if not self.budget_graphs:
            self.capture()
        encoder_outs = self._dispatch(RaggedImageBatch(tokens, grid, self._out_div))
        return self._post_encode(encoder_outs, grid)

    @staticmethod
    def _generate_budgets(min_budget: int, max_budget: int) -> list[int]:
        """Power-of-2 budgets in ``[min_budget, max_budget]``."""
        budgets: list[int] = []
        b = max(1, min_budget)
        while b <= max_budget:
            budgets.append(b)
            b *= 2
        if not budgets or budgets[-1] < max_budget:
            budgets.append(max_budget)
        return budgets

    # ---- geometry ----------------------------------------------------------

    def _synthetic_grid(self, output_budget: int) -> list[list[int]]:
        """``[[1, h, w]]`` producing ``output_budget`` output tokens, with
        ``h, w`` divisible by ``merge``. Shapes-only — replay overwrites."""
        n_patches = output_budget * self._out_div
        m = self._merge
        units = max(n_patches // (m * m), 1)
        a = 1 << (units.bit_length() // 2)
        while a > 1 and units % a != 0:
            a >>= 1
        b = units // a
        return [[1, a * m, b * m]]

    # ---- metadata helpers --------------------------------------------------

    def _pad_cu_seqlens(self, metadata: dict[str, Any]) -> None:
        """Pad ``cu_seqlens`` to ``max_batch_size + 1`` with the last offset
        (trailing slots become zero-length sequences after slice-copy)."""
        cu = metadata["cu_seqlens"]
        pad = self.max_batch_size + 1 - cu.shape[0]
        if pad > 0:
            metadata["cu_seqlens"] = torch.cat([cu, cu[-1:].expand(pad)])

    def _tower_forward(
        self, tokens: torch.Tensor, metadata: dict[str, Any]
    ) -> torch.Tensor:
        out = self.tower.forward_blocks(tokens, metadata)
        if self._out_squeeze_dim is not None:
            out = out.squeeze(self._out_squeeze_dim)
        return out

    # ---- capture -----------------------------------------------------------

    def capture(self) -> None:
        for token_budget in self.token_budgets:
            self._capture_one(token_budget)
        logger.info(
            "Encoder CUDA graph capture complete: %d budget graphs.",
            len(self.budget_graphs),
        )

    def _capture_one(self, token_budget: int) -> None:
        n_input = token_budget * self._out_div

        # Synthetic grid is a device tensor so towers that index it
        # (e.g. grid[:, 0]) work.
        synthetic_grid = torch.tensor(
            self._synthetic_grid(token_budget),
            device=self.device,
            dtype=torch.int32,
        )
        metadata = dict(self.tower.prepare_metadata(synthetic_grid))
        self._pad_cu_seqlens(metadata)
        # max_seqlen is baked here; never refreshed on replay. Must be the
        # per-budget worst case (single image filling the whole budget).
        metadata["max_seqlen"] = n_input
        tokens = torch.zeros(
            (n_input, *self._input_feature_shape),
            device=self.device,
            dtype=self.dtype,
        )

        # Warmup also forces lazy JIT / autotune before capture.
        with torch.inference_mode():
            output = self._tower_forward(tokens, metadata)
            output_buffer = torch.empty_like(output)

        # Vision TP > 1: capture must record the per-layer all-reduce under
        # the custom-AR capture context.
        if self.mapping.vision.tp_size > 1:
            ar_ctx: Any = get_global_backend().custom_ar.capture(
                self.mapping.vision.tp_group
            )
        else:
            ar_ctx = contextlib.nullcontext()

        # No pool= argument: each budget graph gets its own private pool. A
        # shared pool collides custom-AR IPC registrations across budgets
        # (silent cross-rank corruption).
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), ar_ctx, torch.cuda.graph(graph):
            output = self._tower_forward(tokens, metadata)
            output_buffer.copy_(output)

        # Only tensor entries are captured; ints / None are baked at capture.
        metadata_buffers = {
            k: v for k, v in metadata.items() if isinstance(v, torch.Tensor)
        }
        self.budget_graphs[token_budget] = BudgetGraphMetadata(
            graph=graph,
            input_buffer=tokens,
            metadata_buffers=metadata_buffers,
            output_buffer=output_buffer,
        )
        logger.debug(
            "Captured encoder cudagraph: budget=%d, max_batch_size=%d, buffers=%s",
            token_budget,
            self.max_batch_size,
            {k: (v.dtype, tuple(v.shape)) for k, v in metadata_buffers.items()},
        )

    # ---- replay ------------------------------------------------------------

    def _smallest_fitting_budget(self, total_tokens: int) -> int | None:
        for budget in self.token_budgets:
            if budget >= total_tokens:
                return budget
        return None

    @staticmethod
    def _scatter_output_slices(
        output: torch.Tensor,
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor],
        clone: bool = False,
    ) -> None:
        """Slice ``output`` and scatter into ``dest`` by original item index."""
        offset = 0
        for idx in indices:
            n_tok = per_item_out_tokens[idx]
            sliced = output[offset : offset + n_tok]
            dest[idx] = sliced.clone() if clone else sliced
            offset += n_tok

    def _run_budget_graph(
        self,
        batch: RaggedImageBatch,
        token_budget: int,
    ) -> torch.Tensor:
        """Zero + slice-copy the batch into captured buffers, replay. Returns
        the shared output buffer."""
        graph_meta = self.budget_graphs[token_budget]

        # Buffers sized for full budget; real inputs smaller. Zero then
        # slice-copy so padded positions are invisible (cu_seqlens masks them
        # as zero-length).
        src = batch.tokens
        n = src.shape[0]
        graph_meta.input_buffer.zero_()
        graph_meta.input_buffer[:n].copy_(src)

        metadata = dict(self.tower.prepare_metadata(batch.grid))
        self._pad_cu_seqlens(metadata)
        replay_buffers = {
            k: v for k, v in metadata.items() if isinstance(v, torch.Tensor)
        }

        assert replay_buffers.keys() == graph_meta.metadata_buffers.keys()
        for key, buf in graph_meta.metadata_buffers.items():
            new = replay_buffers[key]
            if new.ndim == 0:
                buf.copy_(new)
            else:
                assert new.shape[1:] == buf.shape[1:]
                buf.zero_()
                buf[: new.shape[0]].copy_(new)

        graph_meta.graph.replay()
        return graph_meta.output_buffer

    def _dispatch(self, batch: RaggedImageBatch) -> list[torch.Tensor]:
        """Greedy smallest-first pack into budget graphs; eager fallback for a
        single item exceeding the largest budget."""
        num_items = batch.num_items()
        max_budget = self.token_budgets[-1]
        per_item_out_tokens = batch.output_tokens

        sorted_indices = sorted(range(num_items), key=lambda i: per_item_out_tokens[i])

        batches: list[tuple[list[int], int | None]] = []
        current_batch: list[int] = []
        current_batch_tokens = 0
        for orig_idx in sorted_indices:
            item_tokens = per_item_out_tokens[orig_idx]
            if (
                current_batch_tokens + item_tokens <= max_budget
                and len(current_batch) < self.max_batch_size
            ):
                current_batch.append(orig_idx)
                current_batch_tokens += item_tokens
            else:
                if current_batch:
                    batches.append(
                        (
                            current_batch,
                            self._smallest_fitting_budget(current_batch_tokens),
                        )
                    )
                current_batch = [orig_idx]
                current_batch_tokens = item_tokens
        if current_batch:
            batches.append(
                (
                    current_batch,
                    self._smallest_fitting_budget(current_batch_tokens),
                )
            )

        # Packing reorders; restore original order before return.
        outputs_by_orig_idx: dict[int, torch.Tensor] = {}
        for batch_orig_indices, token_budget in batches:
            sub_batch = batch.select(batch_orig_indices)
            if token_budget is None:
                # Oversized single item: no budget fits → eager fallback.
                with torch.inference_mode():
                    raw = self._tower_forward(
                        sub_batch.tokens,
                        self.tower.prepare_metadata(sub_batch.grid),
                    )
                self._scatter_output_slices(
                    raw, batch_orig_indices, per_item_out_tokens, outputs_by_orig_idx
                )
            else:
                output = self._run_budget_graph(sub_batch, token_budget)
                # clone: output is the shared, reused output_buffer.
                self._scatter_output_slices(
                    output,
                    batch_orig_indices,
                    per_item_out_tokens,
                    outputs_by_orig_idx,
                    clone=True,
                )

        return [outputs_by_orig_idx[i] for i in range(num_items)]
