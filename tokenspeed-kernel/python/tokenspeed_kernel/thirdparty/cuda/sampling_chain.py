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

"""Speculative decoding chain sampling ops: verify_chain_greedy, chain_speculative_sampling_target_only."""

import functools
from pathlib import Path

import torch


@functools.cache
def _load_sampling_chain_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "sampling_chain"
    so_path = objs_dir / "sampling_chain.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel sampling_chain library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def verify_chain_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    batch_size: int,
    num_draft_tokens: int,
    enable_pdl: bool = False,
) -> None:
    _load_sampling_chain_module().verify_chain_greedy(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        target_predict,
        int(batch_size),
        int(num_draft_tokens),
        bool(enable_pdl),
    )


def chain_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor | None = None,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
    enable_pdl: bool = False,
) -> None:
    """Target-only chain speculative sampling.

    When ``draft_probs`` is ``None``, the kernel treats draft probabilities as
    all zeros and avoids the corresponding GMEM traffic.
    """
    _load_sampling_chain_module().chain_speculative_sampling_target_only(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        float(threshold_single),
        float(threshold_acc),
        bool(deterministic),
        bool(enable_pdl),
    )
