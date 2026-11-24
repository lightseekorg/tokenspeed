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

"""Cross-pass messages of a looped Inkling2 core.

In every pass after the first, each core layer reads the residual blocks the
previous pass wrote (the message bank) through an attention over depth and
gates the result into four of its inputs: the attention Q, K and V inputs and
the MoE input. Each input has its own pseudo-query and low-rank controller.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.layernorm.precision import rmsnorm_precision
from tokenspeed_kernel.ops.residual.attention import (
    attention_residual_mix,
    message_controller_inputs,
    message_gate,
)
from torch import nn

from tokenspeed.runtime.layers.attention_residual import PrecisionRMSNorm

# The inputs a core layer gates a message into, in score-weight row order.
LOOP_INPUTS = ("attn", "k", "v", "mlp")


class Inkling2MessageController(nn.Module):
    """Gate a message into one input of a core layer.

    ``h = [rms_norm(x), rms_norm(message)] @ down.T`` (BF16), then
    ``x' = (1 - tanh(h @ forget.T)) * x + tanh(h @ gain.T) * message`` in
    FP32, where ``x`` is the FP32 residual mixture the layer would otherwise
    normalize.
    """

    eps = 1e-6

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.down_weight = nn.Parameter(
            torch.empty((rank, 2 * hidden_size), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.gain_weight = nn.Parameter(
            torch.empty((hidden_size, rank), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.forget_weight = nn.Parameter(
            torch.empty((hidden_size, rank), dtype=torch.bfloat16),
            requires_grad=False,
        )
        # The gain rows, then the forget rows: one GEMM computes both logits.
        self.register_buffer(
            "gate_weight",
            torch.zeros((2 * hidden_size, rank), dtype=torch.bfloat16),
            persistent=False,
        )

    def derive_gate_weight(self) -> None:
        """Write ``gate_weight`` from the loaded projections; safe to repeat."""
        with torch.no_grad():
            self.gate_weight[: self.hidden_size].copy_(self.gain_weight)
            self.gate_weight[self.hidden_size :].copy_(self.forget_weight)

    def forward(self, destination: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """The gated FP32 ``[tokens, hidden]`` input.

        Args:
            destination: Contiguous FP32 ``[tokens, hidden]`` mixture.
            message: Contiguous FP32 ``[tokens, hidden]`` message.
        """
        tokens = destination.shape[0]
        normalized = rmsnorm_precision(
            torch.stack((destination, message), dim=1),
            None,
            self.eps,
            torch.bfloat16,
            mean_by_reciprocal=False,
        ).view(tokens, 2 * self.hidden_size)
        hidden = kernel_mm(normalized, self.down_weight, out_dtype=torch.bfloat16)
        gates = kernel_mm(hidden, self.gate_weight, out_dtype=torch.float32)
        return message_gate(destination, message, gates)


class Inkling2LoopMessages:
    """The message bank of one pass: the residual blocks the previous pass
    wrote, read by every core layer of this pass.

    Args:
        stored: Contiguous ``[blocks, tokens, hidden]`` stored blocks of the
            previous pass (copied out of the residual state).
        partial: The previous pass's rounded ``[tokens, hidden]`` partial.
        score_weights: FP32 ``[core layers, 4, hidden]`` pseudo-queries scaled
            by ``hidden**-0.5``, rows in ``LOOP_INPUTS`` order.
        eps: Epsilon of the RMS-normalized keys and values.
    """

    def __init__(
        self,
        stored: torch.Tensor,
        partial: torch.Tensor,
        score_weights: torch.Tensor,
        eps: float,
    ) -> None:
        self.stored = stored
        self.partial = partial
        self.score_weights = score_weights
        self.eps = eps

    def read(self, core_layer: int, first: int, count: int) -> tuple[torch.Tensor, ...]:
        """The FP32 messages of inputs ``LOOP_INPUTS[first:first + count]``
        of core layer ``core_layer``: a softmax over the bank's sources of
        their RMS-normalized scores, mixing RMS-normalized sources."""
        return tuple(self._mixed(core_layer, first, count).unbind(0))

    def _mixed(self, core_layer: int, first: int, count: int) -> torch.Tensor:
        """:meth:`read`'s messages as one FP32 ``[count, tokens, hidden]``."""
        return attention_residual_mix(
            self.stored,
            (self.partial,),
            self.score_weights[core_layer, first : first + count],
            candidate_out=None,
            normalize_values=True,
            eps=self.eps,
            norm_weights=None,
            norm_eps=self.eps,
            output_dtype=torch.float32,
            tokens=None,
            out=None,
        )

    def gate(
        self,
        core_layer: int,
        first: int,
        destinations: tuple[torch.Tensor, ...],
        controllers: nn.ModuleDict,
    ) -> tuple[torch.Tensor, ...]:
        """Gate the messages of inputs ``LOOP_INPUTS[first:]`` into
        ``destinations``, one FP32 mixture per input."""
        messages = self.read(core_layer, first, len(destinations))
        names = LOOP_INPUTS[first : first + len(destinations)]
        return tuple(
            controllers[name](destination, message)
            for name, destination, message in zip(
                names, destinations, messages, strict=True
            )
        )

    def gate_normalized(
        self,
        core_layer: int,
        first: int,
        destinations: tuple[torch.Tensor, ...],
        controllers: nn.ModuleDict,
        norms: tuple[PrecisionRMSNorm, ...],
    ) -> tuple[torch.Tensor, ...]:
        """:meth:`gate`, then each input's ``norms`` RMSNorm, in one
        controller op (:func:`message_controller_inputs`) for all inputs."""
        count = len(destinations)
        messages = self._mixed(core_layer, first, count)
        names = LOOP_INPUTS[first : first + count]
        gates = tuple(controllers[name] for name in names)
        if (
            any(
                norm.weight is None
                or norm.mean_by_reciprocal
                or norm.output_dtype != torch.bfloat16
                for norm in norms
            )
            or len({norm.eps for norm in norms}) != 1
        ):
            raise ValueError("loop input norms must be BF16 RMSNorms with gains")
        return tuple(
            message_controller_inputs(
                destinations,
                messages,
                tuple(gate.down_weight for gate in gates),
                tuple(gate.gate_weight for gate in gates),
                tuple(norm.weight for norm in norms),
                Inkling2MessageController.eps,
                norms[0].eps,
            ).unbind(0)
        )
