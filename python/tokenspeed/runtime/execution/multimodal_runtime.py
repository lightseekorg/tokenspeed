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

"""Multimodal forward-path runtime, factored out of ModelExecutor.

Owns the M-RoPE position-override machinery (pinned staging, GPU delta
buffer, prefill/decode build paths), encoder CUDA-graph wrapper
installation, and the drafter's multimodal pad-token wiring. The
``mrope_positions_buf`` itself stays in ``InputBuffers``: captured graphs
record its address, and the prefill graph re-pads its tail rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.multimodal.inputs import resolve_mm_pad_substitute_ids
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers

logger = get_colorful_logger(__name__)


class MultimodalRuntime:
    """Per-executor multimodal state and logic.

    Constructed unconditionally; every method is a cheap no-op for pure
    text models (``model_is_mrope=False``, no multimodal context).
    """

    def __init__(
        self,
        *,
        model_is_mrope: bool,
        input_buffers: InputBuffers,
        device: torch.device | str,
    ):
        self.model_is_mrope = model_is_mrope
        self.input_buffers = input_buffers
        self.device = device

        if model_is_mrope:
            mrope_decode_capacity = input_buffers.max_num_tokens
            # Double-buffered pinned host staging for the decode delta copy.
            # Under overlap scheduling the next decode forward is dispatched
            # before the previous result is synchronized, so a single reused
            # pinned buffer could be refilled by the next step while the prior
            # step's ``non_blocking=True`` H2D copy is still reading it (a race
            # that corrupts M-RoPE deltas). Ping-pong two buffers so a buffer is
            # never overwritten while its copy is in flight (overlap depth 1).
            self._mrope_decode_deltas_cpu = [
                self._make_mrope_decode_deltas_cpu(mrope_decode_capacity),
                self._make_mrope_decode_deltas_cpu(mrope_decode_capacity),
            ]
            self._mrope_decode_deltas_cpu_idx = 0
            self._mrope_decode_deltas_buf = torch.zeros(
                mrope_decode_capacity, device=device, dtype=torch.int64
            )
        else:
            self._mrope_decode_deltas_cpu = None
            self._mrope_decode_deltas_cpu_idx = 0
            self._mrope_decode_deltas_buf = None

    @staticmethod
    def _make_mrope_decode_deltas_cpu(size: int) -> torch.Tensor:
        try:
            return torch.zeros(size, dtype=torch.int64, pin_memory=True)
        except RuntimeError:
            return torch.zeros(size, dtype=torch.int64)

    # ------------------------------------------------------------------
    # M-RoPE position overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_mrope_from_input(mm_input, seq_len: int) -> torch.Tensor:
        # Reached only by the chunked/retracted fallback below, where the
        # positions table has no row for this chunk. Computed fresh rather
        # than memoized on ``mm_input``: that struct belongs to the control
        # plane, and a forward must not write into what it was handed.
        return (mm_input.mrope_position_delta - 1).flatten().unsqueeze(0).repeat(
            3, 1
        ) + seq_len

    @staticmethod
    def _mrope_delta_scalar(mm_input) -> int:
        """Read the request's decode position delta.

        ``multimodal_context_for_forward`` resolves the scalar on the control
        plane, so this is a plain read for anything the engine dispatched;
        the tensor branch covers directly-constructed inputs.
        """
        delta = getattr(mm_input, "mrope_position_delta_scalar", None)
        if delta is not None:
            return int(delta)
        tensor = getattr(mm_input, "mrope_position_delta", None)
        if tensor is None:
            return 0
        return int(tensor.flatten()[0].item())

    def _build_decode_mrope_positions_override(
        self,
        forward_op,
        mm_inputs,
        total_tokens: int,
    ) -> torch.Tensor:
        if (
            self._mrope_decode_deltas_cpu is None
            or self._mrope_decode_deltas_buf is None
        ):
            raise RuntimeError(
                "M-RoPE decode buffers were not initialized for this model"
            )

        base_positions = self.input_buffers.positions_buf[:total_tokens]
        # Ping-pong the pinned host staging buffer (see __init__): the previous
        # step's non_blocking H2D copy may still be reading the other buffer.
        cpu_staging = self._mrope_decode_deltas_cpu[self._mrope_decode_deltas_cpu_idx]
        self._mrope_decode_deltas_cpu_idx ^= 1
        token_deltas_cpu = cpu_staging[:total_tokens]

        offset = 0
        has_nonzero_delta = False
        for batch_idx, input_len in enumerate(forward_op.input_lengths):
            input_len = int(input_len)
            if input_len <= 0:
                continue

            delta = 0
            mm_input = mm_inputs[batch_idx] if batch_idx < len(mm_inputs) else None
            # Honor scalar-only deltas: an upstream payload may set
            # mrope_position_delta_scalar while leaving the tensor field
            # mrope_position_delta as None (positions precomputed upstream).
            # _mrope_delta_scalar handles scalar, tensor, and the absent case
            # (returns 0), so call it whenever an mm_input is present.
            if mm_input is not None:
                delta = self._mrope_delta_scalar(mm_input)
                has_nonzero_delta = has_nonzero_delta or delta != 0

            token_deltas_cpu[offset : offset + input_len].fill_(delta)
            offset += input_len

        if offset != total_tokens:
            token_deltas_cpu[offset:total_tokens].zero_()

        if has_nonzero_delta:
            token_deltas = self._mrope_decode_deltas_buf[:total_tokens]
            token_deltas.copy_(token_deltas_cpu, non_blocking=True)
            mrope_base = base_positions + token_deltas
        else:
            mrope_base = base_positions

        self.input_buffers.mrope_positions_buf[:, :total_tokens].copy_(
            mrope_base.unsqueeze(0).expand(3, -1)
        )
        return self.input_buffers.mrope_positions_buf[:, :total_tokens]

    def build_positions_override(
        self,
        forward_op,
        multimodal_context,
        total_tokens: int,
    ) -> torch.Tensor | None:
        """Build the M-RoPE positions override for this forward, or ``None``.

        Writes into ``input_buffers.mrope_positions_buf`` (the graph-stable
        buffer captured forwards read positions from) and returns a view of it.
        """
        if not self.model_is_mrope or total_tokens == 0:
            return None

        is_prefill = forward_op.num_extends() > 0
        base_positions = self.input_buffers.positions_buf[:total_tokens]
        mm_inputs = (
            multimodal_context.mm_inputs
            if multimodal_context is not None and multimodal_context.has_inputs()
            else []
        )
        if not mm_inputs:
            mrope_positions = self.input_buffers.mrope_positions_buf[:, :total_tokens]
            mrope_positions.copy_(
                base_positions.unsqueeze(0).expand_as(mrope_positions)
            )
            return mrope_positions

        if not is_prefill:
            return self._build_decode_mrope_positions_override(
                forward_op=forward_op,
                mm_inputs=mm_inputs,
                total_tokens=total_tokens,
            )

        pos_chunks = torch.split(base_positions, list(forward_op.input_lengths), dim=0)
        mrope_chunks = []
        for batch_idx, base_chunk in enumerate(pos_chunks):
            mm_input = mm_inputs[batch_idx] if batch_idx < len(mm_inputs) else None
            # Fall back to linear only when there is neither a per-token mrope table
            # nor a transferred scalar delta. A decode-only mm_input may carry just
            # the delta (post-image decode positions = base+delta); it must skip the
            # fallback and take the base+delta branch below.
            if mm_input is None or (
                mm_input.mrope_positions is None
                and mm_input.mrope_position_delta is None
            ):
                mrope_chunks.append(base_chunk.unsqueeze(0).expand(3, -1))
                continue

            if (
                is_prefill
                and mm_input.mrope_positions is not None
                and batch_idx < len(forward_op.extend_prefix_lens)
            ):
                start = int(forward_op.extend_prefix_lens[batch_idx])
                end = start + int(forward_op.input_lengths[batch_idx])
                positions = mm_input.mrope_positions[:, start:end]
                if positions.numel() != 0:
                    mrope_chunks.append(
                        positions.to(device=self.device, dtype=torch.int64)
                    )
                    continue
                if base_chunk.numel() == 1:
                    seq_len = int(base_chunk[-1].item()) + 1
                    mrope_chunks.append(
                        self._expand_mrope_from_input(mm_input, seq_len).to(
                            device=self.device, dtype=torch.int64
                        )
                    )
                    continue

            delta = mm_input.mrope_position_delta
            if delta is None:
                delta = torch.zeros(1, dtype=torch.int64)
            delta = delta.flatten()[0].to(device=self.device, dtype=torch.int64)
            # Decode positions need (mrope_delta - 1) + seq_len. positions_buf
            # already stores the per-token zero-based position (seq_len - 1 for
            # decode), so this is the same value without a GPU-to-CPU sync.
            mrope_chunks.append((base_chunk + delta).unsqueeze(0).expand(3, -1))

        mrope_positions = torch.cat(mrope_chunks, dim=1).contiguous()
        self.input_buffers.mrope_positions_buf[:, :total_tokens].copy_(mrope_positions)
        return self.input_buffers.mrope_positions_buf[:, :total_tokens]

    # ------------------------------------------------------------------
    # One-time wiring
    # ------------------------------------------------------------------

    @staticmethod
    def install_encoder_graphs(model, server_args) -> dict:
        """Install encoder CUDA-graph wrappers onto the model.

        Overrides modality encoder callables (e.g. ``image_encoder``,
        ``video_encoder``) with model-built graph wrappers — the
        multimodal-encoder analogue of ``forward_step``'s
        ``ForwardStepRunner``. Returns the installed wrappers by attribute
        name (empty when the model has no encoder-graph support or it is
        disabled).
        """
        if not (
            hasattr(model, "make_encoder_cudagraph_wrappers")
            and getattr(model, "is_multimodal_active", True)
            and envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH.get()
            and server_args.mm_attention_backend != "flashinfer_cudnn"
        ):
            return {}

        wrappers = model.make_encoder_cudagraph_wrappers(model.mapping)
        active_wrappers = {}
        for encoder_attr, wrapper in wrappers.items():
            if not hasattr(model, encoder_attr):
                logger.warning(
                    "Skipping encoder CUDA graph wrapper for missing attribute %s",
                    encoder_attr,
                )
                continue
            setattr(model, encoder_attr, wrapper)
            active_wrappers[encoder_attr] = wrapper
        return active_wrappers

    @staticmethod
    def wire_drafter(input_buffers, target_model_config) -> None:
        """Wire mm pad substitution for the draft path.

        Substitution runs once at fill time on the drafter-only shift-1
        buffer, so drafters only ever see in-vocab input ids. A multimodal
        target with no resolvable substitutes must fail here, loudly —
        otherwise the fill clamp would silently rewrite media positions.
        """
        substitute_ids = resolve_mm_pad_substitute_ids(target_model_config.hf_config)
        if substitute_ids:
            input_buffers.set_mm_pad_substitute_ids(
                substitute_ids, target_model_config.vocab_size
            )
        elif target_model_config.is_multimodal_active:
            raise ValueError(
                "Speculative decoding on a multimodal target requires draft "
                "substitute tokens, but none could be resolved from the HF config."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def timing_counts(multimodal_context) -> tuple[bool, int, int]:
        """(has_mm, mm_count, mm_delta_count) for the timing log line."""
        has_mm = multimodal_context is not None and multimodal_context.has_inputs()
        mm_count = 0
        mm_delta_count = 0
        if has_mm:
            for mm_input in multimodal_context.mm_inputs:
                if mm_input is None:
                    continue
                mm_count += 1
                if mm_input.mrope_position_delta is not None:
                    mm_delta_count += 1
        return has_mm, mm_count, mm_delta_count
