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

"""CPU-only measurement contract for the medium-M fused-tail experiment."""

import math

TOKENS = (256, 384, 512, 768, 1024)
BT_TUNING = {
    "max_tokens": 1024,
    "elements_per_thread": 2,
    "threads": 256,
    "prefetch_group": 1,
    "reduction_threads": 224,
    "rms_threads": 448,
    "enable_pdl": True,
}
REFERENCE_ATOL = 0.046875
REFERENCE_RTOL = 0.01
REFERENCE_L2 = 0.006


def reference_passes(max_absolute, relative_l2, reference_max):
    """Keep the existing complete-tail numerical limits, allowing BF16 order."""
    return (
        all(math.isfinite(x) for x in (max_absolute, relative_l2, reference_max))
        and max_absolute >= 0
        and relative_l2 >= 0
        and reference_max >= 0
        and max_absolute <= REFERENCE_ATOL + REFERENCE_RTOL * reference_max
        and relative_l2 <= REFERENCE_L2
    )


def measurement_config(tokens, rounds, warmup, replays, slots, generations):
    """Validate explicit sampling controls and identify the actual denominator."""
    if type(tokens) is not int or not 256 <= tokens <= 1024:
        raise ValueError("medium experiments require 256 <= M <= 1024")
    for name, value, minimum in (
        ("rounds", rounds, 3),
        ("warmup", warmup, 1),
        ("replays", replays, 1),
        ("slots", slots, 4),
        ("generations", generations, 3),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an explicit integer >= {minimum}")
    if slots != 4:
        raise ValueError("this matched complete-tail boundary uses four layer slots")
    return {
        "M": tokens,
        "rounds": rounds,
        "warmup": warmup,
        "replays": replays,
        "graph_layers": slots,
        "generations": generations,
        "latency_rank_reduction": "max",
        "instrumented": False,
        "baseline": "identical_bt_then_shared_residual_addmm_full_width_multimem_ar2_clone",
        "candidate": "identical_bt_then_published_shared_rs_up_residual_ag",
        "shared_staging_in_both": True,
        "producer_included": False,
        "serving_measurement": False,
        "formal_sampling": rounds >= 31 and warmup >= 8 and replays >= 20,
        "two_batch_acceptance": False,
    }
