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

"""Optional FlashInfer Kimi-K3 front projections with their source payload."""

from __future__ import annotations

import functools
from pathlib import Path

import torch


@functools.cache
def latent_front_available() -> bool:
    try:
        from flashinfer.experimental.kimi_k3_latent_moe import cake_jit
        from flashinfer.experimental.kimi_k3_latent_moe.cake_backend import (
            QUALIFIED_SM_COUNTS,
        )
    except ImportError:
        return False
    root = Path(cake_jit.__file__).parent / "csrc"
    return (
        152 in QUALIFIED_SM_COUNTS
        and bool(cake_jit.MODULES)
        and all(
            (root / source).is_file()
            for record in cake_jit.MODULES.values()
            for source in record["sources"]
        )
    )


def latent_front(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer.kimi_k3_latent_moe import kimi_k3_latent_moe_front

    rows = hidden_states.shape[0]
    logits = torch.empty(
        (rows, router_weight.shape[0]), device=hidden_states.device, dtype=torch.float32
    )
    routed = hidden_states.new_empty((rows, routed_weight.shape[0]))
    shared = hidden_states.new_empty((rows, shared_gate_up_weight.shape[0] // 2))
    return kimi_k3_latent_moe_front(
        hidden_states,
        router_weight,
        routed_weight,
        shared_gate_up_weight,
        logits,
        routed,
        shared,
        backend="cake",
    )
