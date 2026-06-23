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

"""Budget-bucketed CUDA graph capture/replay for multimodal encoders.

The :class:`EncoderCudaGraphWrapper` is the generic manager: it owns budget
selection, graph capture, replay buffer updates, greedy packing, and eager
fallback. Model/modality-specific input layout is supplied by an adapter object.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Protocol

import torch

from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend
from tokenspeed.runtime.utils import logger


@dataclass
class BudgetGraphMetadata:
    """One captured budget graph.

    Replay copies a real batch into the captured input/metadata buffers, calls
    ``graph.replay()``, then reads ``output_buffer``.
    """

    graph: torch.cuda.CUDAGraph
    input_buffers: dict[str, torch.Tensor]
    metadata_buffers: dict[str, torch.Tensor]
    output_buffer: torch.Tensor


class EncoderCudaGraphBatch(Protocol):
    """Minimal batch contract consumed by :class:`EncoderCudaGraphWrapper`."""

    @property
    def input_tensors(self) -> dict[str, torch.Tensor]: ...

    @property
    def encoder_output_tokens(self) -> list[int]: ...

    @property
    def metadata_sequences(self) -> list[int]: ...

    def num_items(self) -> int: ...

    def select(self, indices: list[int]) -> "EncoderCudaGraphBatch": ...


class EncoderCudaGraphAdapter(Protocol):
    """Model/modality-specific contract used by :class:`EncoderCudaGraphWrapper`."""

    @property
    def modality_name(self) -> str: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def capture_tp_size(self) -> int: ...

    @property
    def capture_tp_group(self) -> Any | None: ...

    def batch_from_items(self, items: list[Any]) -> EncoderCudaGraphBatch: ...

    def capture_batch_for_budget(
        self,
        encoder_output_token_budget: int,
        max_batch_size: int,
        metadata_sequence_budget: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> EncoderCudaGraphBatch: ...

    def prepare_metadata(
        self,
        batch: EncoderCudaGraphBatch,
        encoder_output_token_budget: int | None,
        metadata_sequence_budget: int,
    ) -> dict[str, Any]: ...

    def forward(
        self,
        input_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> torch.Tensor: ...

    def postprocess(
        self,
        encoder_outs: list[torch.Tensor],
        batch: EncoderCudaGraphBatch,
    ) -> torch.Tensor: ...


@dataclass
class VisionEncoderBatch:
    """Qwen/Kimi-style vision batch.

    Per-item patch rows are concatenated on dim 0 and indexed by ``grid_thw``.
    ``.tolist()`` syncs are confined here, in the eager region, never inside a
    graph replay.
    """

    tokens: torch.Tensor
    grid: torch.Tensor
    out_div: int

    @property
    def input_tensors(self) -> dict[str, torch.Tensor]:
        return {"tokens": self.tokens}

    @cached_property
    def _grid_rows(self) -> list[list[int]]:
        return self.grid.tolist()

    def num_items(self) -> int:
        return self.grid.shape[0]

    @cached_property
    def encoder_output_tokens(self) -> list[int]:
        return [(t * h * w) // self.out_div for t, h, w in self._grid_rows]

    @cached_property
    def cu_input(self) -> list[int]:
        cu = [0]
        for t, h, w in self._grid_rows:
            cu.append(cu[-1] + t * h * w)
        return cu

    @cached_property
    def metadata_sequences(self) -> list[int]:
        return [t for t, _, _ in self._grid_rows]

    def select(self, indices: list[int]) -> "VisionEncoderBatch":
        """Sub-batch at ``indices``, preserving order."""
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
        return VisionEncoderBatch(self.tokens[rows], self.grid[indices], self.out_div)


@dataclass
class VisionEncoderCudaGraphAdapter:
    """Adapter for Qwen/Kimi-style ``grid_thw`` vision encoders."""

    tower: Any
    pre_encode: Callable[[list[Any]], tuple[torch.Tensor, torch.Tensor]]
    post_encode: Callable[[list[torch.Tensor], torch.Tensor], torch.Tensor]
    out_div: int
    merge: int
    input_feature_shape: tuple[int, ...]
    modality_name: str = "vision"
    out_squeeze_dim: int | None = None
    capture_tp_size: int = 1
    capture_tp_group: Any | None = None

    @cached_property
    def _param(self) -> torch.nn.Parameter:
        return next(self.tower.parameters())

    @property
    def device(self) -> torch.device:
        return self._param.device

    @property
    def dtype(self) -> torch.dtype:
        return self._param.dtype

    def synthetic_grid(self, encoder_output_token_budget: int) -> list[list[int]]:
        n_patches = encoder_output_token_budget * self.out_div
        units = max(n_patches // (self.merge * self.merge), 1)
        a = 1 << (units.bit_length() // 2)
        while a > 1 and units % a != 0:
            a >>= 1
        b = units // a
        return [[1, a * self.merge, b * self.merge]]

    def pad_cu_seqlens(
        self, metadata: dict[str, Any], metadata_sequence_budget: int
    ) -> None:
        cu = metadata["cu_seqlens"]
        target = metadata_sequence_budget + 1
        if cu.shape[0] > target:
            raise RuntimeError(
                f"{self.modality_name} encoder cudagraph needs {cu.shape[0] - 1} "
                f"metadata sequences, but the configured limit is "
                f"{metadata_sequence_budget}"
            )
        pad = target - cu.shape[0]
        if pad > 0:
            metadata["cu_seqlens"] = torch.cat([cu, cu[-1:].expand(pad)])

    def batch_from_items(self, items: list[Any]) -> VisionEncoderBatch:
        tokens, grid = self.pre_encode(items)
        return VisionEncoderBatch(tokens, grid, self.out_div)

    def capture_batch_for_budget(
        self,
        encoder_output_token_budget: int,
        _max_batch_size: int,
        _metadata_sequence_budget: int,
        capture_device: torch.device,
        capture_dtype: torch.dtype,
    ) -> VisionEncoderBatch:
        grid = torch.tensor(
            self.synthetic_grid(encoder_output_token_budget),
            device=capture_device,
            dtype=torch.int32,
        )
        tokens = torch.zeros(
            (encoder_output_token_budget * self.out_div, *self.input_feature_shape),
            device=capture_device,
            dtype=capture_dtype,
        )
        return VisionEncoderBatch(tokens, grid, self.out_div)

    def prepare_metadata(
        self,
        batch: EncoderCudaGraphBatch,
        encoder_output_token_budget: int | None,
        metadata_sequence_budget: int,
    ) -> dict[str, Any]:
        if not isinstance(batch, VisionEncoderBatch):
            raise TypeError(
                f"{self.modality_name} encoder cudagraph expected "
                f"VisionEncoderBatch, got {type(batch).__name__}"
            )
        metadata = dict(self.tower.prepare_metadata(batch.grid))
        if encoder_output_token_budget is not None:
            self.pad_cu_seqlens(metadata, metadata_sequence_budget)
            # Non-tensor scalar gets baked at capture. Use the per-budget worst
            # case so replay never exceeds the captured attention max seqlen.
            metadata["max_seqlen"] = encoder_output_token_budget * self.out_div
        return metadata

    def forward(
        self, input_tensors: dict[str, torch.Tensor], metadata: dict[str, Any]
    ) -> torch.Tensor:
        out = self.tower.forward_blocks(input_tensors["tokens"], metadata)
        if self.out_squeeze_dim is not None:
            out = out.squeeze(self.out_squeeze_dim)
        return out

    def postprocess(
        self, encoder_outs: list[torch.Tensor], batch: EncoderCudaGraphBatch
    ) -> torch.Tensor:
        if not isinstance(batch, VisionEncoderBatch):
            raise TypeError(
                f"{self.modality_name} encoder cudagraph expected "
                f"VisionEncoderBatch, got {type(batch).__name__}"
            )
        return self.post_encode(encoder_outs, batch.grid)


class EncoderCudaGraphWrapper:
    """Generic budget-based CUDA graph manager for encoder callables.

    The wrapper does not know about image/video/audio internals. It only expects
    batches to expose input tensors, per-item output row counts, per-item
    metadata sequence counts, and a ``select`` operation. The adapter constructs
    batches, prepares metadata, runs the captured forward, and postprocesses
    output slices.
    """

    def __init__(
        self,
        *,
        adapter: EncoderCudaGraphAdapter,
        budget_range: tuple[int, int],
        max_batch_size: int | None = None,
        max_metadata_sequences_per_batch: int | None = None,
        metadata_sequence_budget_from_encoder_output_budget: bool = False,
    ):
        self.adapter = adapter
        self.device = adapter.device
        self.dtype = adapter.dtype
        self.modality_name = adapter.modality_name

        min_budget, max_budget = budget_range
        self.encoder_output_token_budgets = self._generate_budgets(
            min_budget, max_budget
        )
        self.max_batch_size = (
            max_batch_size
            if max_batch_size is not None
            else max(1, max_budget // max(1, min_budget))
        )
        self.max_metadata_sequences_per_batch = max_metadata_sequences_per_batch
        self.metadata_sequence_budget_from_encoder_output_budget = (
            metadata_sequence_budget_from_encoder_output_budget
        )

        self.capture_tp_size = adapter.capture_tp_size
        self.capture_tp_group = adapter.capture_tp_group

        self.budget_graphs: dict[int, BudgetGraphMetadata] = {}

        metadata_sequence_budget_log_value = (
            self.max_metadata_sequences_per_batch
            if self.max_metadata_sequences_per_batch is not None
            else (
                "encoder_output_token_budget"
                if self.metadata_sequence_budget_from_encoder_output_budget
                else "batch"
            )
        )
        logger.info(
            "EncoderCudaGraphWrapper initialized: modality=%s, budgets=%s, "
            "max_batch_size=%d, max_metadata_sequences_per_batch=%s, encoder_tp=%d",
            self.modality_name,
            self.encoder_output_token_budgets,
            self.max_batch_size,
            metadata_sequence_budget_log_value,
            self.capture_tp_size,
        )

    def __call__(self, items: list[Any]) -> torch.Tensor:
        batch = self.adapter.batch_from_items(items)
        if not self.budget_graphs:
            self.capture()
        encoder_outs = self._dispatch(batch)
        return self.adapter.postprocess(encoder_outs, batch)

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

    def _metadata_sequence_budget_for_encoder_output_budget(
        self, encoder_output_token_budget: int
    ) -> int:
        if self.max_metadata_sequences_per_batch is not None:
            return min(
                self.max_metadata_sequences_per_batch, encoder_output_token_budget
            )
        if self.metadata_sequence_budget_from_encoder_output_budget:
            return encoder_output_token_budget
        return self.max_batch_size

    def capture(self) -> None:
        for encoder_output_token_budget in self.encoder_output_token_budgets:
            self._capture_one(encoder_output_token_budget)
        logger.info(
            "Encoder CUDA graph capture complete: modality=%s, %d budget graphs.",
            self.modality_name,
            len(self.budget_graphs),
        )

    def _capture_one(self, encoder_output_token_budget: int) -> None:
        metadata_sequence_budget = (
            self._metadata_sequence_budget_for_encoder_output_budget(
                encoder_output_token_budget
            )
        )
        batch = self.adapter.capture_batch_for_budget(
            encoder_output_token_budget,
            self.max_batch_size,
            metadata_sequence_budget,
            self.device,
            self.dtype,
        )
        metadata = dict(
            self.adapter.prepare_metadata(
                batch, encoder_output_token_budget, metadata_sequence_budget
            )
        )
        input_buffers = batch.input_tensors

        # Warmup also forces lazy JIT / autotune before capture.
        with torch.inference_mode():
            output = self.adapter.forward(input_buffers, metadata)
            output_buffer = torch.empty_like(output)

        # Encoder TP > 1: capture must record per-layer all-reduce under the
        # custom-AR capture context.
        if self.capture_tp_size > 1 and self.capture_tp_group is not None:
            ar_ctx: Any = get_global_backend().custom_ar.capture(self.capture_tp_group)
        else:
            ar_ctx = contextlib.nullcontext()

        # No pool= argument: each budget graph gets its own private pool. A
        # shared pool collides custom-AR IPC registrations across budgets.
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), ar_ctx, torch.cuda.graph(graph):
            output = self.adapter.forward(input_buffers, metadata)
            output_buffer.copy_(output)

        # Only tensor entries are captured. Ints / None are baked at capture.
        metadata_buffers = {
            k: v for k, v in metadata.items() if isinstance(v, torch.Tensor)
        }
        self.budget_graphs[encoder_output_token_budget] = BudgetGraphMetadata(
            graph=graph,
            input_buffers=input_buffers,
            metadata_buffers=metadata_buffers,
            output_buffer=output_buffer,
        )
        logger.debug(
            "Captured encoder cudagraph: modality=%s, budget=%d, "
            "max_batch_size=%d, metadata_sequence_budget=%d, buffers=%s",
            self.modality_name,
            encoder_output_token_budget,
            self.max_batch_size,
            metadata_sequence_budget,
            {k: (v.dtype, tuple(v.shape)) for k, v in metadata_buffers.items()},
        )

    def _smallest_fitting_budget(
        self, total_encoder_output_tokens: int, total_metadata_sequences: int
    ) -> int | None:
        for budget in self.encoder_output_token_budgets:
            if (
                budget >= total_encoder_output_tokens
                and total_metadata_sequences
                <= self._metadata_sequence_budget_for_encoder_output_budget(budget)
            ):
                return budget
        return None

    @staticmethod
    def _scatter_output_slices(
        output: torch.Tensor,
        indices: list[int],
        per_item_encoder_output_tokens: list[int],
        dest: dict[int, torch.Tensor],
        clone: bool = False,
    ) -> None:
        """Slice ``output`` and scatter into ``dest`` by original item index."""
        offset = 0
        for idx in indices:
            n_tokens = per_item_encoder_output_tokens[idx]
            sliced = output[offset : offset + n_tokens]
            dest[idx] = sliced.clone() if clone else sliced
            offset += n_tokens

    def _run_budget_graph(
        self,
        batch: EncoderCudaGraphBatch,
        encoder_output_token_budget: int,
    ) -> torch.Tensor:
        """Copy the batch into captured buffers, replay, and return output."""
        graph_meta = self.budget_graphs[encoder_output_token_budget]
        metadata_sequence_budget = (
            self._metadata_sequence_budget_for_encoder_output_budget(
                encoder_output_token_budget
            )
        )

        src_buffers = batch.input_tensors
        if src_buffers.keys() != graph_meta.input_buffers.keys():
            raise RuntimeError(
                f"{self.modality_name} encoder cudagraph input keys changed: "
                f"capture={sorted(graph_meta.input_buffers.keys())}, "
                f"replay={sorted(src_buffers.keys())}"
            )

        for key, buf in graph_meta.input_buffers.items():
            src = src_buffers[key]
            n = src.shape[0]
            if n > buf.shape[0]:
                raise RuntimeError(
                    f"{self.modality_name} encoder cudagraph input {key} has "
                    f"{n} rows, but budget {encoder_output_token_budget} only "
                    f"captured {buf.shape[0]} rows"
                )
            if src.shape[1:] != buf.shape[1:]:
                raise RuntimeError(
                    f"{self.modality_name} encoder cudagraph input {key} "
                    f"shape changed after dim0: capture={tuple(buf.shape)}, "
                    f"replay={tuple(src.shape)}"
                )
            buf.zero_()
            buf[:n].copy_(src)

        metadata = dict(
            self.adapter.prepare_metadata(
                batch, encoder_output_token_budget, metadata_sequence_budget
            )
        )
        replay_buffers = {
            k: v for k, v in metadata.items() if isinstance(v, torch.Tensor)
        }

        if replay_buffers.keys() != graph_meta.metadata_buffers.keys():
            raise RuntimeError(
                f"{self.modality_name} encoder cudagraph metadata keys changed: "
                f"capture={sorted(graph_meta.metadata_buffers.keys())}, "
                f"replay={sorted(replay_buffers.keys())}"
            )

        for key, buf in graph_meta.metadata_buffers.items():
            new = replay_buffers[key]
            if new.ndim == 0:
                buf.copy_(new)
            else:
                if new.shape[1:] != buf.shape[1:]:
                    raise RuntimeError(
                        f"{self.modality_name} encoder cudagraph metadata {key} "
                        f"shape changed after dim0: capture={tuple(buf.shape)}, "
                        f"replay={tuple(new.shape)}"
                    )
                if new.shape[0] > buf.shape[0]:
                    raise RuntimeError(
                        f"{self.modality_name} encoder cudagraph metadata {key} "
                        f"has {new.shape[0]} rows, but the captured buffer only "
                        f"has {buf.shape[0]} rows"
                    )
                buf.zero_()
                buf[: new.shape[0]].copy_(new)

        graph_meta.graph.replay()
        return graph_meta.output_buffer

    def _run_eager(self, batch: EncoderCudaGraphBatch) -> torch.Tensor:
        metadata = dict(
            self.adapter.prepare_metadata(
                batch, None, max(1, sum(batch.metadata_sequences))
            )
        )
        return self.adapter.forward(batch.input_tensors, metadata)

    def _dispatch(self, batch: EncoderCudaGraphBatch) -> list[torch.Tensor]:
        """Greedy smallest-first pack into budget graphs with eager fallback."""
        num_items = batch.num_items()
        max_budget = self.encoder_output_token_budgets[-1]
        max_metadata_sequence_budget = (
            self._metadata_sequence_budget_for_encoder_output_budget(max_budget)
        )
        per_item_encoder_output_tokens = batch.encoder_output_tokens
        per_item_metadata_sequences = batch.metadata_sequences

        sorted_indices = sorted(
            range(num_items), key=lambda i: per_item_encoder_output_tokens[i]
        )

        batches: list[tuple[list[int], int | None]] = []
        current_batch: list[int] = []
        current_batch_encoder_output_tokens = 0
        current_batch_metadata_sequences = 0
        for orig_idx in sorted_indices:
            item_encoder_output_tokens = per_item_encoder_output_tokens[orig_idx]
            item_metadata_sequences = per_item_metadata_sequences[orig_idx]
            if (
                current_batch_encoder_output_tokens + item_encoder_output_tokens
                <= max_budget
                and len(current_batch) < self.max_batch_size
                and current_batch_metadata_sequences + item_metadata_sequences
                <= max_metadata_sequence_budget
            ):
                current_batch.append(orig_idx)
                current_batch_encoder_output_tokens += item_encoder_output_tokens
                current_batch_metadata_sequences += item_metadata_sequences
            else:
                if current_batch:
                    batches.append(
                        (
                            current_batch,
                            self._smallest_fitting_budget(
                                current_batch_encoder_output_tokens,
                                current_batch_metadata_sequences,
                            ),
                        )
                    )
                current_batch = [orig_idx]
                current_batch_encoder_output_tokens = item_encoder_output_tokens
                current_batch_metadata_sequences = item_metadata_sequences
        if current_batch:
            batches.append(
                (
                    current_batch,
                    self._smallest_fitting_budget(
                        current_batch_encoder_output_tokens,
                        current_batch_metadata_sequences,
                    ),
                )
            )

        # Packing reorders; restore original order before return.
        outputs_by_orig_idx: dict[int, torch.Tensor] = {}
        for batch_orig_indices, encoder_output_token_budget in batches:
            sub_batch = batch.select(batch_orig_indices)
            if encoder_output_token_budget is None:
                with torch.inference_mode():
                    raw = self._run_eager(sub_batch)
                self._scatter_output_slices(
                    raw,
                    batch_orig_indices,
                    per_item_encoder_output_tokens,
                    outputs_by_orig_idx,
                )
            else:
                output = self._run_budget_graph(sub_batch, encoder_output_token_budget)
                # clone: output is the shared, reused output_buffer.
                self._scatter_output_slices(
                    output,
                    batch_orig_indices,
                    per_item_encoder_output_tokens,
                    outputs_by_orig_idx,
                    clone=True,
                )

        return [outputs_by_orig_idx[i] for i in range(num_items)]
