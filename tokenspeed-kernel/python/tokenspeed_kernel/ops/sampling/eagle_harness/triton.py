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

"""Triton build of the EAGLE-Harness draft-tree and device-policy ops
(greedy trees and chains), registered in family ``sampling`` with the op
name as mode. This module only registers them: the host wrappers and the
kernels live in ``_triton/tree.py`` and ``_triton/policy.py``.

Every op launches a fixed number of programs and reads its per-round counts
(live rows, widths, pool sizes) from device int32 ``[1]`` tensors, so a
launch captured in a graph once is replayed for every round. Kernels run
without FMA contraction in a fixed reduction order: results are
deterministic bit for bit.

Not built here: the sampled-tree ops (``sample_children``,
``tree_reject_sample``, ``residual_sample``, ``publish_draft_q``) and the
device-chain ops, which need device-side graph launch.
``policy_select_verify`` registers for AMD only; its fixed summation order
(column means, then a blocked prefix sum) is spelled out on
``_triton.policy.triton_eagle_harness_policy_select_verify``.
"""

from __future__ import annotations

from tokenspeed_kernel.ops.sampling.eagle_harness._triton import policy as _policy
from tokenspeed_kernel.ops.sampling.eagle_harness._triton import tree as _tree
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature

__all__ = [
    "triton_eagle_harness_update_node_data",
    "triton_eagle_harness_work_deltas",
    "triton_eagle_harness_draft_depth_update",
    "triton_eagle_harness_stable_topk_desc",
    "triton_eagle_harness_write_verify_tree",
    "triton_eagle_harness_policy_load",
    "triton_eagle_harness_policy_begin_round",
    "triton_eagle_harness_policy_decide_tree",
    "triton_eagle_harness_policy_decide_chain",
    "triton_eagle_harness_policy_select_verify",
]

_PORTABLE = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))
_AMD = CapabilityRequirement(vendors=frozenset({"amd"}))


triton_eagle_harness_update_node_data = register_kernel(
    "sampling",
    "eagle_harness_update_node_data",
    name="triton_eagle_harness_update_node_data",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_tree.triton_eagle_harness_update_node_data)

triton_eagle_harness_work_deltas = register_kernel(
    "sampling",
    "eagle_harness_work_deltas",
    name="triton_eagle_harness_work_deltas",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_tree.triton_eagle_harness_work_deltas)

triton_eagle_harness_draft_depth_update = register_kernel(
    "sampling",
    "eagle_harness_draft_depth_update",
    name="triton_eagle_harness_draft_depth_update",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_tree.triton_eagle_harness_draft_depth_update)

triton_eagle_harness_stable_topk_desc = register_kernel(
    "sampling",
    "eagle_harness_stable_topk_desc",
    name="triton_eagle_harness_stable_topk_desc",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_tree.triton_eagle_harness_stable_topk_desc)

triton_eagle_harness_write_verify_tree = register_kernel(
    "sampling",
    "eagle_harness_write_verify_tree",
    name="triton_eagle_harness_write_verify_tree",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_tree.triton_eagle_harness_write_verify_tree)

triton_eagle_harness_policy_load = register_kernel(
    "sampling",
    "eagle_harness_policy_load",
    name="triton_eagle_harness_policy_load",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_policy.triton_eagle_harness_policy_load)

triton_eagle_harness_policy_begin_round = register_kernel(
    "sampling",
    "eagle_harness_policy_begin_round",
    name="triton_eagle_harness_policy_begin_round",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_policy.triton_eagle_harness_policy_begin_round)

triton_eagle_harness_policy_decide_tree = register_kernel(
    "sampling",
    "eagle_harness_policy_decide_tree",
    name="triton_eagle_harness_policy_decide_tree",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_policy.triton_eagle_harness_policy_decide_tree)

triton_eagle_harness_policy_decide_chain = register_kernel(
    "sampling",
    "eagle_harness_policy_decide_chain",
    name="triton_eagle_harness_policy_decide_chain",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_policy.triton_eagle_harness_policy_decide_chain)

triton_eagle_harness_policy_select_verify = register_kernel(
    "sampling",
    "eagle_harness_policy_select_verify",
    name="triton_eagle_harness_policy_select_verify",
    solution="triton",
    capability=_AMD,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)(_policy.triton_eagle_harness_policy_select_verify)
