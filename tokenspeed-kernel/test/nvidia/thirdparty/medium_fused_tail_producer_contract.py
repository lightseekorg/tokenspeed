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

"""Explicit producer-inclusive boundary; never relabel the pre-generated tail."""

from medium_fused_tail_contract import measurement_config

ARMS = ("baseline", "copied", "direct")
COMPARISONS = (("baseline", "copied"), ("copied", "direct"), ("baseline", "direct"))


def producer_measurement_config(tokens, rounds, warmup, replays, generations):
    """Return the same sampling policy with a distinct three-arm timed boundary."""
    base = measurement_config(tokens, rounds, warmup, replays, 4, generations)
    for name in ("baseline", "candidate", "shared_staging_in_both"):
        base.pop(name)
    return {
        **base,
        "producer_included": True,
        "boundary": "per_layer_shared_down_projection_start_through_complete_tail",
        "producer_solution": "torch",
        "producer_dtype": "bfloat16",
        "producer_activation_shape": [tokens, 768],
        "producer_weight_shape": [7168, 768],
        "producer_output_shape": [tokens, 7168],
        "arms": {
            "baseline": "producer_to_ordinary_then_copy_bt_owner_addmm_full_ar2_clone",
            "copied": "producer_to_ordinary_then_copy_bt_fused_rs_up_residual_ag",
            "direct": "producer_to_symmetric_then_bt_fused_rs_up_residual_ag",
        },
        "same_fused_plan_for_copied_and_direct": True,
        "same_bt_for_all_arms": True,
        "persistent_results_alias_between_copied_direct": True,
        "correctness_snapshots_before_next_arm": True,
        "trace_verified_no_staging": False,
    }
