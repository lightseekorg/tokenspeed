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

"""Continuous medium/large geometry for the integrated K3 fused tail."""

from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_config import (
    MediumFusedRsUpProjectionTuning,
)


def integrated_fused_rs_serving_config(
    num_tokens: int,
) -> MediumFusedRsUpProjectionTuning:
    """Select the unqualified continuous-range integrated acceptance profile.

    Args:
        num_tokens: Actual kernel M, or padded graph M, in (32,8192].

    Returns:
        Explicit medium/large geometry. Interpolated intervals require fresh
        numerical and serving acceptance; historical endpoints do not qualify them.
    """
    if type(num_tokens) is not int or not 32 < num_tokens <= 8192:
        raise ValueError("integrated fused profile requires M in (32,8192]")
    # The M256 two-CTA screen was within one percent of one CTA. Keep a
    # single one-CTA policy for the medium range instead of a narrow exception.
    if num_tokens <= 512:
        tile_m, tile_n, two_cta, cluster_m = 64, 64, False, 1
    elif num_tokens <= 1024:
        tile_m, tile_n, two_cta, cluster_m = 64, 128, False, 1
    else:
        tile_m, tile_n, two_cta, cluster_m = 256, 128, True, 2
    large = num_tokens > 1024
    tuning = MediumFusedRsUpProjectionTuning(
        tile_m=tile_m,
        tile_n=tile_n,
        two_cta=two_cta,
        cluster_m=cluster_m,
        cluster_n=1,
        c_stages=3 if large else 0,
        ab_stages=6 if large else 0,
        addend_stages=1,
        reduce_vectors=4,
        cluster_cap=38 if 1024 < num_tokens <= 4096 else None,
        scheduler_type="static_persistent",
    )
    tuning.validate()
    return tuning
