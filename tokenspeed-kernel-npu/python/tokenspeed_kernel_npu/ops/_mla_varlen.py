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

"""Empty variable-length attention states at the Ascend adapter boundary."""

import torch


def set_empty_attention_rows(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> None:
    """Write zero output and log(0) only for requests with no key rows.

    FIA reports +inf LSE for an empty KV segment. That value is its empty
    sentinel, not a partition function, and cannot enter a partial-state merge.
    Nonempty request rows retain the original output and LSE bytes.
    """
    if cu_seqlens_q.numel() != cu_seqlens_kv.numel():
        raise ValueError("MLA query and KV endpoints must describe the same requests")
    query_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    empty_keys = cu_seqlens_kv[1:].eq(cu_seqlens_kv[:-1])
    empty_rows = torch.repeat_interleave(
        empty_keys.to(device=output.device),
        query_lengths.to(device=output.device),
        output_size=output.shape[0],
    )
    output.masked_fill_(empty_rows[:, None, None], 0)
    if lse is not None:
        lse.masked_fill_(empty_rows[:, None], -torch.inf)
