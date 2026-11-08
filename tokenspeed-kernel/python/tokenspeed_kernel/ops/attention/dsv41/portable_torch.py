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

"""Portable, graph-capturable DSpark verification budget allocation."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "attention",
    "dsv41_verify_layout",
    name="torch_dsv41_verify_layout",
    solution="torch",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=[format_signature(x=dense_tensor_format(torch.float32))],
    traits={},
    priority=Priority.PORTABLE,
)
def torch_dsv41_verify_layout(
    scores, query_capacity, rows_out, indptr_out, counts_out, valid_out
):
    batch_size, draft_width = scores.shape
    width = draft_width + 1
    finite = torch.isfinite(scores).all(dim=1)
    survival = torch.sigmoid(scores.clamp(-60, 60)).cumprod(dim=1)
    gains = torch.where(
        finite[:, None], survival, torch.full_like(survival, float("inf"))
    ).flatten()
    # Stable ties preserve a prefix within each request, including underflowed
    # probabilities and unknown-score rows requiring their complete draft.
    order = torch.argsort(gains, descending=True, stable=True)
    extra = torch.zeros_like(gains, dtype=torch.int64)
    extra.scatter_(0, order[: query_capacity - batch_size], 1)
    counts = 1 + extra.reshape(batch_size, draft_width).sum(dim=1)
    valid = (~finite).sum() * draft_width <= query_capacity - batch_size
    counts = torch.where(valid, counts, torch.ones_like(counts))
    indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    positions = torch.arange(query_capacity, device=scores.device, dtype=torch.int64)
    owners = torch.searchsorted(indptr[1:], positions, right=True).clamp_max(
        batch_size - 1
    )
    rows = owners * width + positions - indptr[owners]
    rows_out.copy_(torch.where(positions < indptr[-1], rows, torch.zeros_like(rows)))
    indptr_out.copy_(indptr)
    counts_out.copy_(counts)
    valid_out.copy_(valid)
