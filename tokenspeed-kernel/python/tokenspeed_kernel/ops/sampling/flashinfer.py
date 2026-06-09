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

"""FlashInfer sampling kernels."""

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

min_p_sampling_from_probs = error_fn
softmax = error_fn
top_k_renorm_prob = error_fn
top_k_top_p_sampling_from_probs = error_fn
top_k_top_p_sampling_from_logits = error_fn
top_p_renorm_prob = error_fn
top_p_renorm_probs = error_fn

if current_platform().is_nvidia:
    try:
        from flashinfer.sampling import (
            min_p_sampling_from_probs,
            top_k_renorm_prob,
            top_k_top_p_sampling_from_logits,
            top_k_top_p_sampling_from_probs,
            top_p_renorm_prob,
            top_p_renorm_probs,
        )
    except ImportError:
        pass

    try:
        from tokenspeed_kernel.thirdparty.cuda.flashinfer_softmax import softmax
    except ImportError:
        pass

__all__ = [
    "min_p_sampling_from_probs",
    "softmax",
    "top_k_renorm_prob",
    "top_k_top_p_sampling_from_logits",
    "top_k_top_p_sampling_from_probs",
    "top_p_renorm_prob",
    "top_p_renorm_probs",
]
