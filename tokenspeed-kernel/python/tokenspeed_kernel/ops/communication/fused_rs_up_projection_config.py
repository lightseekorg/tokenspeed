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

"""Explicit configurations for the qualified shared-RS/up-projection/AG path.

This module is pure configuration. It does not enable a runtime route or
promise serving performance. All returned dictionaries are independent.
"""


def fused_rs_up_projection_config(num_tokens: int) -> dict:
    """Return the fixed large-M GEMM configuration and requested cluster cap.

    Args:
        num_tokens: Exactly 4096 or 8192; unmeasured endpoints are rejected.
    Returns:
        A new mapping with explicit tuning and launch-policy fields. A null
        cap requests hardware capacity; it does not hardcode an observed grid.
    """
    if type(num_tokens) is not int or num_tokens not in (4096, 8192):
        raise ValueError("qualified fused RS/up/AG requires M4096 or M8192")
    return {
        "num_tokens": num_tokens,
        "tuning": {
            "tile_m": 256,
            "tile_n": 128,
            "two_cta": True,
            "cluster_m": 2,
            "cluster_n": 1,
            "enable_pdl": False,
            "store_kind": "tma",
            "c_stages": 0,
            "release_acc_early": True,
            "acquire_before_store": True,
            "prefetch_acc_tile": False,
            "epilogue_m": 128,
            "epilogue_n": 64,
            "addend_cache_policy": "no_allocate",
            "paired_addend_loads": False,
            "addend_stages": 1,
            "reduce_vectors": 4,
        },
        "cluster_cap": 38 if num_tokens == 4096 else None,
        "ht_stages": 10,
        "raw_publication_warps": 4,
        "output_exit_warps": 4,
        "staging_order": "before_ht",
        "residual_issue_order": "after_reduce",
    }


def fused_rs_tail_ht_config(stages: int) -> dict:
    """Return explicit HT settings for a matched complete-tail comparison.

    Args:
        stages: Seven for the materialized-RS baseline, ten for the fused tail.
    Returns:
        An independent mapping accepted by MNNVLCuteDSLHTFinalizeTuning.
    """
    if type(stages) is not int or stages not in (7, 10):
        raise ValueError("the qualified tail comparison uses HT stages 7 or 10")
    return {
        "max_tokens": 8192,
        "persistent_ctas": None,
        "consumer_threads": 448,
        "vectors_per_thread": 1,
        "stages": stages,
        "reduction_warps": 2,
        "reduction_cta_groups": None,
        "rms_token_groups": 2,
        "rms_pipeline_stages": 3,
        "rms_shard_major": False,
        "enable_pdl": True,
    }
