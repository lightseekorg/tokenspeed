# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CUDA sampling kernels."""

import os as _os

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

chain_speculative_sampling_target_only = error_fn
# Prepare is a side-stream registration helper. On non-NVIDIA the fused
# kernel doesn't exist, so callers should never reach the renorm path —
# but prepare gets called unconditionally from FlashInfer backends'
# __init__, so we keep it as a silent no-op rather than error_fn.
fused_topk_topp_prepare = lambda *_args, **_kwargs: None  # noqa: E731
fused_topk_topp_renorm = error_fn
fused_topk_topp_workspace_size = error_fn
verify_chain_greedy = error_fn

# One availability decision for every sampler backend. The 2026-07-14
# misaligned-address quarantine was lifted after
# upstream #683 fixed the kernel's graph-capture addressing without losing
# float4 throughput.
fused_topk_topp_available = False

if current_platform().is_nvidia:
    try:
        from tokenspeed_kernel.thirdparty.cuda.sampling_chain import (
            chain_speculative_sampling_target_only,
            verify_chain_greedy,
        )
    except ImportError:
        pass

    try:
        from tokenspeed_kernel.thirdparty.cuda.fused_topk_topp import (
            fused_topk_topp_renorm,
            fused_topk_topp_workspace_size,
        )
        from tokenspeed_kernel.thirdparty.cuda.fused_topk_topp import (
            prepare_for_device as fused_topk_topp_prepare,
        )

        fused_topk_topp_available = (
            _os.environ.get("TS_DISABLE_FUSED_TOPK_TOPP", "0") != "1"
        )
    except ImportError:
        pass

    # Its tvm-ffi library loads on the first launch, not on import.
    from tokenspeed_kernel.thirdparty.cuda.eagle4_candidate import (
        eagle4_candidate_tail as _eagle4_candidate_tail_impl,
    )
    from tokenspeed_kernel.thirdparty.cuda.eagle4_candidate import (
        eagle4_shard_topk as _eagle4_shard_topk_impl,
    )

    # Thread-block clusters: SM90 onward.
    _EAGLE4_CAPABILITY = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0), vendors=frozenset({"nvidia"})
    )

    @register_kernel(
        "sampling",
        "eagle4_shard_topk",
        name="cuda_eagle4_shard_topk",
        solution="cuda",
        capability=_EAGLE4_CAPABILITY,
        signatures=format_signatures("scores", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
    )
    def cuda_eagle4_shard_topk(
        scores: torch.Tensor, k: int, vocab_start: int, global_ids: torch.Tensor
    ) -> None:
        _eagle4_shard_topk_impl(scores, k, vocab_start, global_ids)

    @register_kernel(
        "sampling",
        "eagle4_candidate_tail",
        name="cuda_eagle4_candidate_tail",
        solution="cuda",
        capability=_EAGLE4_CAPABILITY,
        signatures=format_signatures("cand_logits", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
    )
    def cuda_eagle4_candidate_tail(
        cand_logits: torch.Tensor,
        cand_tokens: torch.Tensor,
        top_k: int,
        logits_out: torch.Tensor,
        tokens_out: torch.Tensor,
        topk_probs: torch.Tensor | None,
        topk_tokens: torch.Tensor | None,
    ) -> None:
        _eagle4_candidate_tail_impl(
            cand_logits,
            cand_tokens,
            top_k,
            logits_out,
            tokens_out,
            topk_probs,
            topk_tokens,
        )


__all__ = [
    "chain_speculative_sampling_target_only",
    "fused_topk_topp_available",
    "fused_topk_topp_prepare",
    "fused_topk_topp_renorm",
    "fused_topk_topp_workspace_size",
    "verify_chain_greedy",
]
