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

"""The multicast GEMM's column fold, exercised on one GPU through a local pointer.

The launcher folds this rank's column base into the pointer it is handed, so
every peer receives this rank's shard at the right columns and the Lamport
sentinels around it survive for the others. Nothing else in the suite runs that
arithmetic: the runtime tests mock the launcher, and the only end-to-end
exerciser is a manual eight-rank program.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe.latent_down import (
    _DOWN_SENTINEL,
    _LAMPORT_CTAS,
    _LAMPORT_THREADS,
    arm_mailbox,
)

HIDDEN = 7168
LATENT = 3584
TP = 8
SHARD = LATENT // TP
MAX_M = 8
MARK = 1.5
# The two empty words as int32 holds them: this mailbox's (-0, -0) and, still
# spelled out rather than derived, the (+0, -0) every other Lamport buffer keeps.
DOWN_EMPTY = -0x7FFF8000
TAIL_EMPTY = -0x80000000


def _is_blackwell() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0)[0] == 10
    )


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="requires NVIDIA sm100+")


def _launch(rank: int, m: int, tp: int = TP) -> tuple[torch.Tensor, ...]:
    """Publish through a pointer that is NOT the validated mailbox's own address.

    In production that pointer is the multicast VA. Passing the mailbox's own
    address would verify the column fold only modulo the two coinciding.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.fused_multicast_latent_down_gemm import (  # noqa: E501
        FusedMulticastLatentDownGemmKernel,
    )

    device = torch.device("cuda", 0)
    torch.manual_seed(20260902 + rank)
    shard = LATENT // tp
    hidden = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = torch.randn(shard, HIDDEN, device=device, dtype=torch.bfloat16)
    mailbox = torch.full((1, MAX_M, LATENT), MARK, device=device, dtype=torch.bfloat16)
    published = torch.full(
        (1, MAX_M, LATENT), MARK, device=device, dtype=torch.bfloat16
    )
    kernel = FusedMulticastLatentDownGemmKernel(
        rank=rank, tp_size=tp, in_dim=HIDDEN, latent_dim=LATENT, num_rows=m
    )
    kernel(hidden, weight, mailbox, published.data_ptr())
    torch.cuda.synchronize(device)
    return published, mailbox, (hidden.float() @ weight.float().T)


# TP8 covers the two- and four-output stores; the eight-output one is only ever
# selected at a 224-wide block, which is what TP16 gives.
@pytest.mark.parametrize("tp, rank", [(8, 0), (8, 3), (8, 7), (16, 0), (16, 15)])
@pytest.mark.parametrize("m", [1, 4, 5, 8])
def test_the_block_lands_at_this_rank_s_columns(tp: int, rank: int, m: int) -> None:
    """Only this rank's columns of the first m rows may be written."""
    published, mailbox, reference = _launch(rank, m, tp)
    shard = LATENT // tp
    start = rank * shard
    written = published[0, :m, start : start + shard]
    torch.testing.assert_close(written.float(), reference, rtol=2e-2, atol=2e-2)

    untouched = published.clone()
    untouched[0, :m, start : start + shard] = MARK
    assert torch.equal(untouched, torch.full_like(untouched, MARK))
    # The validated tensor is not the destination; only the pointer is.
    assert torch.equal(mailbox, torch.full_like(mailbox, MARK))


def test_a_short_mailbox_is_refused_rather_than_overrun() -> None:
    """The kernel writes by row stride through a raw pointer."""
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.fused_multicast_latent_down_gemm import (  # noqa: E501
        FusedMulticastLatentDownGemmKernel,
    )

    device = torch.device("cuda", 0)
    kernel = FusedMulticastLatentDownGemmKernel(
        rank=0, tp_size=TP, in_dim=HIDDEN, latent_dim=LATENT, num_rows=4
    )
    short = torch.zeros((1, 2, LATENT), device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="smaller than the compiled M"):
        kernel(
            torch.zeros(4, HIDDEN, device=device, dtype=torch.bfloat16),
            torch.zeros(SHARD, HIDDEN, device=device, dtype=torch.bfloat16),
            short,
            short.data_ptr(),
        )


@pytest.mark.parametrize("m", [1, 8])
def test_the_gather_re_arms_the_rows_it_consumed(m: int) -> None:
    """A slot handed back un-armed reads as arrived on its next round."""
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.fused_multicast_latent_down_gemm import (  # noqa: E501
        FusedMulticastLatentDownGemmKernel,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.lamport_copy import (
        LamportCopyKernel,
    )

    device = torch.device("cuda", 0)
    torch.manual_seed(902)
    hidden = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = torch.randn(SHARD, HIDDEN, device=device, dtype=torch.bfloat16)
    mailbox = torch.empty(1, MAX_M, LATENT, device=device, dtype=torch.bfloat16)
    gemm = FusedMulticastLatentDownGemmKernel(
        rank=0, tp_size=TP, in_dim=HIDDEN, latent_dim=LATENT, num_rows=m
    )
    gather = LamportCopyKernel(
        hidden_dim=LATENT,
        max_m=MAX_M,
        ctas=_LAMPORT_CTAS,
        threads=_LAMPORT_THREADS,
        sentinel=_DOWN_SENTINEL,
    )

    arm_mailbox(mailbox)
    gemm(hidden, weight, mailbox, mailbox.data_ptr())
    # One GPU publishes one rank's columns; the gather waits on the whole row, so
    # the peers' columns have to be stood in for or it spins forever.
    mailbox[:, :m, SHARD:] = 1.0
    out = gather(mailbox, m=m)[0]
    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        out[:, :SHARD].float(), hidden.float() @ weight.float().T, rtol=2e-2, atol=2e-2
    )
    consumed = mailbox[0, :m].reshape(-1).view(torch.int32)
    assert torch.equal(consumed, torch.full_like(consumed, DOWN_EMPTY))
    # The re-arm has to write what the host arming writes, or the next round hangs.
    rearmed = torch.empty_like(mailbox)
    arm_mailbox(rearmed)
    assert torch.equal(consumed, rearmed[0, :m].reshape(-1).view(torch.int32))


# Each store width has its own sanitize call, and which one runs is chosen by the
# block's shard width and static M: (8, 1) takes the two-output store, (8, 4) the
# four-output one, (16, 1) the eight-output one.
@pytest.mark.parametrize(("tp", "m"), [(8, 1), (8, 4), (8, 8), (16, 1), (16, 4)])
def test_a_result_of_negative_zero_is_not_left_looking_like_an_empty_slot(
    tp: int, m: int
) -> None:
    """A written word equal to the sentinel reads as "not arrived" and hangs."""
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.fused_multicast_latent_down_gemm import (  # noqa: E501
        FusedMulticastLatentDownGemmKernel,
    )

    device = torch.device("cuda", 0)
    tiny = 2.0**-70
    shard = LATENT // tp
    hidden = torch.zeros(m, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = torch.zeros(shard, HIDDEN, device=device, dtype=torch.bfloat16)
    # Every marked column of the block sums to -2**-140, a negative zero in bf16.
    # Every row, not just the first: the epilogue's row loop is unrolled, so each
    # row's sanitize is its own compiled instance and needs its own witness.
    hidden[:, 0] = tiny
    mailbox = torch.full((1, MAX_M, LATENT), MARK, device=device, dtype=torch.bfloat16)
    published = torch.full(
        (1, MAX_M, LATENT), MARK, device=device, dtype=torch.bfloat16
    )
    kernel = FusedMulticastLatentDownGemmKernel(
        rank=0, tp_size=tp, in_dim=HIDDEN, latent_dim=LATENT, num_rows=m
    )
    # Odd columns alone pair each -0 with its even neighbour's +0 to spell the
    # tail's sentinel; every column spells this mailbox's, both lanes -0. Either
    # way all four words of the widest store carry one, and the same compiled
    # kernel writes both, so the second pattern costs nothing to witness.
    for columns in (range(1, 8, 2), range(8)):
        weight.zero_()
        for column in columns:
            weight[column, 0] = -tiny
        published.fill_(MARK)
        kernel(hidden, weight, mailbox, published.data_ptr())
        torch.cuda.synchronize(device)

        written = published[0, :m, :shard].contiguous().view(torch.int32)
        assert not bool((written == TAIL_EMPTY).any())
        assert not bool((written == DOWN_EMPTY).any())
        # Not vacuous: those words were written, and sanitizing -0 leaves +0.
        assert torch.equal(written[:, :4], torch.zeros_like(written[:, :4]))


def test_the_widths_this_file_parametrizes_still_select_three_stores() -> None:
    """The sanitizer's coverage claim rests on which store each pair selects.

    Not a tuned value being pinned: it is the mapping the parametrization above
    depends on, so a retune that changes it should fail here and be told to add
    the width back rather than quietly retiring the only sentinel witness these
    primitives have anywhere in the suite.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.fused_multicast_latent_down_gemm import (  # noqa: E501
        config_for_m,
    )

    widths = {
        config_for_m(m, LATENT // tp).outputs_per_block
        for tp, m in ((8, 1), (8, 4), (8, 8), (16, 1), (16, 4))
    }
    assert widths == {2, 4, 8}


# One 16-byte fragment is four words, assembled from two to four independent
# multicast stores arriving over the fabric, so a partly-filled fragment is the
# steady state the arrival check exists for. The check can only be witnessed by
# a gather that does NOT finish, which needs its own process to bound.
_WAITER = """
import torch
from tokenspeed_kernel.ops.moe.latent_down import (
    _DOWN_SENTINEL,
    _LAMPORT_CTAS,
    _LAMPORT_THREADS,
    arm_mailbox,
)
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.lamport_copy import (
    LamportCopyKernel,
)

device = torch.device("cuda", 0)
mailbox = torch.empty(1, {max_m}, {latent}, device=device, dtype=torch.bfloat16)
arm_mailbox(mailbox)
row = mailbox[0, 0].view(torch.int32)
row.fill_(0x3F803F80)
row[{word}] = {empty}
gather = LamportCopyKernel(
    hidden_dim={latent},
    max_m={max_m},
    ctas=_LAMPORT_CTAS,
    threads=_LAMPORT_THREADS,
    sentinel=_DOWN_SENTINEL,
)
# The kernel compiles on its first call, so warm it on a mailbox that is fully
# arrived; only then is the marker true and the window below purely the wait.
row.fill_(0x3F803F80)
gather(mailbox, m=1)
torch.cuda.synchronize(device)
row.fill_(0x3F803F80)
row[{word}] = {empty}
open({marker!r}, "w").close()
gather(mailbox, m=1)
torch.cuda.synchronize(device)
print("COMPLETED", flush=True)
"""


@pytest.mark.parametrize("word", [0, 1, 2, 3])
def test_the_gather_waits_for_every_word_of_a_fragment(word: int) -> None:
    """A fragment missing any one word must not be treated as arrived."""
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        marker = Path(scratch) / "reached-the-gather"
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WAITER.format(
                    max_m=MAX_M,
                    latent=LATENT,
                    word=word,
                    empty=DOWN_EMPTY,
                    marker=str(marker),
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # The marker separates the child's startup from the wait under test, so
        # the window the gather is given need not cover import and compile time.
        for _ in range(600):
            if marker.exists() or child.poll() is not None:
                break
            time.sleep(0.1)
        assert (
            marker.exists() or child.poll() is not None
        ), "the child never reached the gather"
        try:
            out, err = child.communicate(timeout=6)
        except subprocess.TimeoutExpired:
            return  # still waiting on the missing word, which is the contract
        finally:
            child.kill()
        assert (
            "COMPLETED" not in out
        ), f"the gather consumed a fragment whose word {word} had not arrived"
        assert (
            child.returncode == 0
        ), f"the waiter failed for another reason: {err[-2000:]}"


# Two mailboxes, two sentinels, one primitive: the words each spins on are the
# other's ordinary payload, so a gather that ignored the sentinel it was built
# with would either hang here or hand back a word it had rewritten. A hang is
# not a failure a pytest process can report, so this runs in a bounded child.
_CROSS = """
import torch
from tokenspeed_kernel.ops.moe.latent_down import (
    _DOWN_SENTINEL,
    _LAMPORT_CTAS,
    _LAMPORT_THREADS,
    arm_mailbox,
)
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.lamport_copy import (
    LamportCopyKernel,
)

device = torch.device("cuda", 0)
down = LamportCopyKernel(
    hidden_dim={latent},
    max_m={max_m},
    ctas=_LAMPORT_CTAS,
    threads=_LAMPORT_THREADS,
    sentinel=_DOWN_SENTINEL,
)
tail = LamportCopyKernel(
    hidden_dim={latent}, max_m={max_m}, ctas=_LAMPORT_CTAS, threads=_LAMPORT_THREADS
)

mailbox = torch.empty(1, {max_m}, {latent}, device=device, dtype=torch.bfloat16)
mailbox.view(torch.int32).fill_({tail_empty})
out = down(mailbox, m={max_m})[0].contiguous().view(torch.int32)
torch.cuda.synchronize(device)
assert torch.equal(out, torch.full_like(out, {tail_empty})), "down misread a payload"
armed = torch.empty_like(mailbox)
arm_mailbox(armed)
assert torch.equal(mailbox.view(torch.int32), armed.view(torch.int32)), "down re-arm"

mailbox.view(torch.int32).fill_({down_empty})
out = tail(mailbox, m={max_m})[0].contiguous().view(torch.int32)
torch.cuda.synchronize(device)
assert torch.equal(out, torch.full_like(out, {down_empty})), "tail misread a payload"
words = mailbox.view(torch.int32)
assert torch.equal(words, torch.full_like(words, {tail_empty})), "tail re-arm"
print("COMPLETED", flush=True)
"""


def test_the_two_mailboxes_hold_their_own_sentinels_at_once() -> None:
    """Neither gather may read, or leave behind, the other's empty word."""
    import subprocess
    import sys

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CROSS.format(
                max_m=MAX_M,
                latent=LATENT,
                down_empty=DOWN_EMPTY,
                tail_empty=TAIL_EMPTY,
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Four kernels compile here; the bound is against a gather that spins on
        # the wrong word forever, not against a slow build.
        out, err = child.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        child.kill()
        raise AssertionError("a gather never stopped waiting on the other sentinel")
    assert child.returncode == 0, f"{out[-2000:]}\n{err[-2000:]}"
    assert "COMPLETED" in out


def test_constructing_the_gather_compiles_what_its_first_launch_needs() -> None:
    """The pre-warm exists so a launch inside a graph capture cannot compile.

    ``functools.cache`` keys on the shape of the call, not on the values in it,
    so a pre-warm passing the same values in another shape builds an entry no
    launch ever looks up: two compiles a boot, and the capture the pre-warm was
    added for still compiles inside itself.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.lamport_copy import (
        LamportCopyKernel,
        compile_kernel,
    )

    device = torch.device("cuda", 0)
    compile_kernel.cache_clear()
    gather = LamportCopyKernel(
        hidden_dim=LATENT,
        max_m=MAX_M,
        ctas=_LAMPORT_CTAS,
        threads=_LAMPORT_THREADS,
        sentinel=_DOWN_SENTINEL,
    )
    built = compile_kernel.cache_info().misses
    assert built == 2, "the constructor pre-warms the plain and the residual gather"

    residual = torch.zeros(1, LATENT, device=device, dtype=torch.bfloat16)
    mailbox = torch.empty(1, MAX_M, LATENT, device=device, dtype=torch.bfloat16)
    for addend in (None, residual):
        # A payload, not the sentinel: neither launch may wait on anything.
        mailbox.view(torch.int32).fill_(0x3F803F80)
        gather(mailbox, m=1, residual=addend)
        torch.cuda.synchronize(device)
        assert compile_kernel.cache_info().misses == built
    assert compile_kernel.cache_info().hits == 2
