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

"""Stable TokenSpeed entry points for Ascend sampling kernels."""

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import ErrorClass, error_fn

SamplingWorkspace = ErrorClass
chain_speculative_sampling_target_only = error_fn
fused_topk_topp_renorm = error_fn
min_p_renorm_prob = error_fn
sample_from_probs = error_fn
verify_chain_greedy = error_fn

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.sampling import (
        SamplingWorkspace,
        chain_speculative_sampling_target_only,
        fused_topk_topp_renorm,
        min_p_renorm_prob,
        sample_from_probs,
        verify_chain_greedy,
    )

__all__ = [
    "SamplingWorkspace",
    "chain_speculative_sampling_target_only",
    "fused_topk_topp_renorm",
    "min_p_renorm_prob",
    "sample_from_probs",
    "verify_chain_greedy",
]
