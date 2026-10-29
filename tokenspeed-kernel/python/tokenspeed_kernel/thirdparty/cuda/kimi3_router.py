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

"""Vendored FlashInfer/Cake Kimi-K3 sigmoid-bias top-16 routing."""

from functools import cache
from pathlib import Path

import torch


@cache
def _module():
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

    source = Path(__file__).parent / "csrc" / "kimi3_router.cu"
    return gen_jit_spec(
        "tokenspeed_kimi3_router",
        [source],
        extra_cuda_cflags=[*sm100a_nvcc_flags, *sm103a_nvcc_flags],
        extra_include_paths=[source.parent],
        use_fast_math=False,
    ).build_and_load()


def sigmoid_bias_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select and normalize top-16 routes over 896 experts.

    Args:
        logits: Contiguous CUDA FP32 tensor [tokens, 896].
        correction_bias: Contiguous CUDA FP32 tensor [896], used for selection.
        out: Optional caller-owned FP32 weights and INT32 IDs [tokens, 16].

    Returns:
        Normalized sigmoid weights and expert IDs, ordered by ascending ID.
        Compile once in eager mode before capturing calls into a CUDA graph.
    """
    if out is None:
        out = (
            torch.empty(
                (logits.shape[0], 16), device=logits.device, dtype=torch.float32
            ),
            torch.empty((logits.shape[0], 16), device=logits.device, dtype=torch.int32),
        )
    max_blocks = (
        4 * torch.cuda.get_device_properties(logits.device).multi_processor_count
    )
    _module().run(logits, correction_bias, out[0], out[1], max_blocks)
    return out
