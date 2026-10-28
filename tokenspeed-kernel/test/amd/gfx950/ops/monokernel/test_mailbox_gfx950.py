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

"""Tagged-pair mailboxes on gfx950."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch
from utils import is_cdna4

# The CPU-side helpers come from the AMD kernel package, which other vendors
# do not install.
pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    LAYER_SLOTS,
    Plan,
    Stage,
    launch_tag,
)

if is_cdna4():
    from tokenspeed_kernel_amd._triton import gl, gluon
    from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule

    @gluon.jit
    def _exchange_kernel(
        scratch,
        step_ptr,
        bad,
        PRODUCERS: gl.constexpr,
        PER: gl.constexpr,
        CONSUMERS: gl.constexpr,
    ):
        # Producers sit on the highest CTAs, consumers on the lowest: a
        # consumer waits on CTAs that start after it.
        pid = gl.program_id(0)
        tag = mailbox.launch_tag(step_ptr, 3)
        first_producer = gl.num_programs(0) - PRODUCERS
        if pid >= first_producer:
            layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
            i = gl.arange(0, PER, layout)
            p = pid - first_producer
            mailbox.put(scratch, p * PER + i, p * 65536 + i + tag, tag)
        elif pid < CONSUMERS:
            lanes = gl.arange(
                0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps())
            )
            # One hint per producer: its last pair.
            hint = gl.minimum(lanes, PRODUCERS - 1) * PER + PER - 1
            mailbox.pre_poll(scratch, hint, tag)
            total: gl.constexpr = PRODUCERS * PER
            per_thread: gl.constexpr = total // (64 * gl.num_warps())
            layout2: gl.constexpr = gl.BlockedLayout(
                [per_thread], [64], [gl.num_warps()], [0]
            )
            j = gl.arange(0, total, layout2)
            got = mailbox.poll(scratch, j, tag)
            expect = (j // PER) * 65536 + j % PER + tag
            gl.store(bad + pid, gl.sum((got != expect).to(gl.int32), axis=0))

    _ROUNDS_PLAN = Plan(
        grid=8, stages=(Stage("s", 21, 5),), regions=(), rows=1, chain=1
    ).key

    @gluon.jit
    def _rounds_kernel(ran, PLAN: gl.constexpr):
        task, rounds = schedule.stage_rounds(PLAN, "s")
        for i in range(rounds):
            gl.atomic_add(ran + task + i * 8, gl.program_id(0) + 1)

    def test_stage_rounds_run_every_task_once_on_its_cta():
        ran = torch.zeros(21, dtype=torch.int32, device="cuda")
        _rounds_kernel[(8,)](ran, _ROUNDS_PLAN, num_warps=1)
        # Task t runs once, on CTA (5 + t) % 8. The kernel records CTA + 1.
        assert ran.tolist() == [(5 + t) % 8 + 1 for t in range(21)]

    def test_every_consumer_reads_the_current_launch_pairs():
        producers, per, consumers, grid = 64, 64, 128, 256
        scratch = torch.zeros(producers * per, dtype=torch.int64, device="cuda")
        bad = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        step = torch.zeros(1, dtype=torch.int32, device="cuda")
        for value in (7, 8):
            step.fill_(value)
            bad.fill_(-1)
            _exchange_kernel[(grid,)](
                scratch, step, bad, producers, per, consumers, num_warps=8
            )
            torch.cuda.synchronize()
            assert bad[:consumers].tolist() == [0] * consumers
        # The pairs carry the last launch's tag.
        assert (int(scratch[0].item()) >> 32) & 0xFFFFFFFF == launch_tag(8, 3)

    def test_a_tag_past_bit_31_still_matches():
        # The int32 tag wraps negative after about 2**31 / LAYER_SLOTS steps.
        # The pair holds its 32 bits, so the poll must still match it.
        producers, per, consumers, grid = 64, 64, 128, 256
        scratch = torch.zeros(producers * per, dtype=torch.int64, device="cuda")
        bad = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        value = (1 << 31) // LAYER_SLOTS
        assert launch_tag(value, 3) >= 1 << 31
        step = torch.full((1,), value, dtype=torch.int32, device="cuda")
        _exchange_kernel[(grid,)](
            scratch, step, bad, producers, per, consumers, num_warps=8
        )
        torch.cuda.synchronize()
        assert bad[:consumers].tolist() == [0] * consumers
        assert (int(scratch[0].item()) >> 32) & 0xFFFFFFFF == launch_tag(value, 3)

    # A poll whose pair never arrives, with a small retry bound so that the
    # test is quick. It runs in its own process: the trap ends the process's
    # GPU context.
    _STUCK_POLL = textwrap.dedent("""
        import torch
        from tokenspeed_kernel_amd._triton import gl, gluon
        from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox

        mailbox.MAX_RETRIES = 1 << 12

        @gluon.jit
        def _stuck_kernel(scratch, step_ptr, out):
            tag = mailbox.launch_tag(step_ptr, 0)
            lanes = gl.arange(
                0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps())
            )
            gl.store(out + lanes, mailbox.poll(scratch, lanes, tag))

        scratch = torch.zeros(256, dtype=torch.int64, device="cuda")
        step = torch.zeros(1, dtype=torch.int32, device="cuda")
        out = torch.zeros(256, dtype=torch.int32, device="cuda")
        _stuck_kernel[(1,)](scratch, step, out, num_warps=4)
        torch.cuda.synchronize()
        print("the poll returned")
        """)

    def test_a_poll_that_never_completes_stops_the_process(tmp_path):
        """Fail-stop: a missing producer makes the poll trap, and the process
        gets a GPU error instead of hanging."""
        script = tmp_path / "stuck_poll.py"  # a JIT function needs a file
        script.write_text(_STUCK_POLL)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "the poll returned" not in result.stdout
        # The trap reaches the host as a failed launch. Where the runtime
        # handles GPU exceptions itself (with GPU core dumps on), it aborts
        # the queue and the process instead.
        assert (
            "launch failure" in result.stderr
            or "HSA_STATUS_ERROR_EXCEPTION" in result.stderr
        ), result.stderr
