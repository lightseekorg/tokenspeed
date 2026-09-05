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

"""The taped padding scrub must leave exactly what the torch fills left.

The captured graph reads the padded rows, so a tail this misses feeds stale
token ids / positions / seq lens into the recorded kernels; the comparison is
bitwise on every buffer. (KV write locations are backend-owned now — no
location buffer lives in InputBuffers.)
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.metadata import Reg  # noqa: E402

from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402

MAX_BS, MAX_TOKENS, PAD_POOL = 8, 64, 5
BUFS = (
    "input_ids_buf",
    "positions_buf",
    "req_pool_indices_buf",
    "state_write_req_pool_indices_buf",
    "seq_lens_buf",
)


def _make(device="cuda"):
    return InputBuffers(
        max_bs=MAX_BS,
        max_num_tokens=MAX_TOKENS,
        state_write_padding_pool_index=PAD_POOL,
        device=device,
    )


def _torch_scrub(ib, total_tokens, batch_size):
    if total_tokens < ib.max_num_tokens:
        ib.input_ids_buf[total_tokens:].fill_(1)
        ib.positions_buf[total_tokens:].fill_(0)
    if batch_size < ib.max_bs:
        ib.req_pool_indices_buf[batch_size:].fill_(0)
        ib.state_write_req_pool_indices_buf[batch_size:].fill_(
            ib.state_write_padding_pool_index
        )
        ib.seq_lens_buf[batch_size:].fill_(1)


def _dirty(ib):
    for name in BUFS:
        b = getattr(ib, name)
        b.copy_(torch.full_like(b, -99))


@pytest.mark.parametrize(
    ("total_tokens", "batch_size"),
    [(0, 0), (1, 1), (17, 3), (63, 7), (MAX_TOKENS, MAX_BS), (MAX_TOKENS - 1, 1)],
)
def test_tape_matches_the_torch_fills(total_tokens, batch_size):
    taped, plain = _make(), _make()
    assert taped._pad_tape is not None
    _dirty(taped)
    _dirty(plain)
    taped._pad_tape.run({Reg.TOKENS: total_tokens, Reg.BS: batch_size})
    _torch_scrub(plain, total_tokens, batch_size)
    torch.cuda.synchronize()
    for name in BUFS:
        assert torch.equal(getattr(taped, name), getattr(plain, name)), name


def _count_device_work(fn):
    """Kernels and memcpies this issues, counted apart: the tape trades the
    kernel launches for one register upload, so both sides have to be seen."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = copies = 0
    for e in prof.events():
        if getattr(e, "device_time", 0) <= 0:
            continue
        if e.name.startswith(("Memcpy", "Memset")):
            copies += 1
        else:
            kernels += 1
    return kernels, copies


def test_the_scrub_costs_one_launch():
    """Five fills on the step's critical path were five launches; collapsing
    them is the whole point, so the count is the assertion."""
    ib = _make()
    taped_k, taped_c = _count_device_work(
        lambda: ib._pad_tape.run({Reg.TOKENS: 4, Reg.BS: 2})
    )
    plain_k, _ = _count_device_work(lambda: _torch_scrub(_make(), 4, 2))
    assert plain_k >= 5, plain_k
    assert taped_k == 1, taped_k
    assert taped_c <= 1, taped_c


def test_a_cpu_input_buffer_keeps_the_torch_path():
    """The runtime builds these on CPU in metadata tests; a tape would need a
    GPU there."""
    ib = _make(device="cpu")
    assert ib._pad_tape is None
