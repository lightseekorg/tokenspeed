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

"""Triton sampling kernel entry points."""

from .common import gather_and_expand_scalars
from .generic import gumbel_sample_from_pools_generic
from .gumbel import (
    gumbel_sample_from_pools,
    gumbel_sample_from_pools_compact,
)
from .logprobs import selected_token_logprobs
from .min_p import (
    gumbel_sample_min_p_from_pools,
    gumbel_sample_min_p_from_pools_parallel,
)
from .penalties import accumulate_counts_inplace, apply_penalties_logit_bias_inplace
from .probability import min_p_renorm_prob
from .topk_topp import (
    _QRITA_PERCENTILE_TO_STD_TABLE,
    gumbel_sample_top_k_top_p_from_pools,
    gumbel_sample_top_k_top_p_qrita_from_pools,
)
from .topp import gumbel_sample_top_p_parallel_from_pools
from .verify import verify_chain_target_sampled

__all__ = [
    "_QRITA_PERCENTILE_TO_STD_TABLE",
    "gather_and_expand_scalars",
    "gumbel_sample_from_pools",
    "gumbel_sample_from_pools_compact",
    "gumbel_sample_min_p_from_pools",
    "gumbel_sample_min_p_from_pools_parallel",
    "gumbel_sample_from_pools_generic",
    "gumbel_sample_top_p_parallel_from_pools",
    "gumbel_sample_top_k_top_p_from_pools",
    "gumbel_sample_top_k_top_p_qrita_from_pools",
    "min_p_renorm_prob",
    "apply_penalties_logit_bias_inplace",
    "accumulate_counts_inplace",
    "selected_token_logprobs",
    "verify_chain_target_sampled",
]
