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

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

chain_speculative_sampling_target_only = error_fn
# Prepare is a side-stream registration helper. On non-NVIDIA the fused
# kernel doesn't exist, so callers should never reach the renorm path —
# but prepare gets called unconditionally from FlashInfer backends'
# __init__, so we keep it as a silent no-op rather than error_fn.
fused_topk_topp_prepare = lambda *_args, **_kwargs: None  # noqa: E731
fused_topk_topp_renorm = error_fn
fused_topk_topp_workspace_size = error_fn
verify_chain_greedy = error_fn

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
    except ImportError:
        pass

__all__ = [
    "chain_speculative_sampling_target_only",
    "fused_topk_topp_prepare",
    "fused_topk_topp_renorm",
    "fused_topk_topp_workspace_size",
    "verify_chain_greedy",
]
