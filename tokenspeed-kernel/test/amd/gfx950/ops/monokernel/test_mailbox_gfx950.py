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

from tokenspeed_kernel_amd.ops.gfx950.monokernel import stuck  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    LAYER_SLOTS,
    POLL_TIMEOUT_MANTISSA,
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
        PLAN: gl.constexpr,
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
            mailbox.pre_poll(scratch, hint, tag, PLAN)
            total: gl.constexpr = PRODUCERS * PER
            per_thread: gl.constexpr = total // (64 * gl.num_warps())
            layout2: gl.constexpr = gl.BlockedLayout(
                [per_thread], [64], [gl.num_warps()], [0]
            )
            j = gl.arange(0, total, layout2)
            got = mailbox.poll(scratch, j, tag, PLAN)
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
                scratch,
                step,
                bad,
                producers,
                per,
                consumers,
                Plan(grid=grid, stages=(), regions=(), rows=1, chain=1).key,
                num_warps=8,
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
            scratch,
            step,
            bad,
            producers,
            per,
            consumers,
            Plan(grid=grid, stages=(), regions=(), rows=1, chain=1).key,
            num_warps=8,
        )
        torch.cuda.synchronize()
        assert bad[:consumers].tolist() == [0] * consumers
        assert (int(scratch[0].item()) >> 32) & 0xFFFFFFFF == launch_tag(value, 3)

    @gluon.jit
    def _waves_kernel(scratch, step_ptr, out, SPIN: gl.constexpr, PLAN: gl.constexpr):
        # CTA 1 publishes pairs 0 .. 1023. CTA 0's waves below SPIN poll two
        # tensors of them; the other waves point at pairs that no one writes.
        pid = gl.program_id(0)
        tag = mailbox.launch_tag(step_ptr, 3)
        lanes = gl.arange(0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps()))
        if pid == 1:
            mailbox.put(scratch, lanes, lanes * 3 + tag, tag)
            mailbox.put(scratch, 512 + lanes, lanes * 5 + tag, tag)
        else:
            spin = lanes // 64 < SPIN
            first, second = mailbox.poll_many_waves(
                scratch,
                (
                    gl.where(spin, lanes, 2048 + lanes),
                    512 + gl.where(spin, lanes, 2048),
                ),
                tag,
                spin.to(gl.int32),
                PLAN,
            )
            gl.store(out + lanes, first, mask=spin)
            gl.store(out + 512 + lanes, second, mask=spin)

    def test_waves_without_spin_skip_the_poll():
        scratch = torch.zeros(4096, dtype=torch.int64, device="cuda")
        out = torch.full((1024,), -1, dtype=torch.int32, device="cuda")
        step = torch.full((1,), 5, dtype=torch.int32, device="cuda")
        # The waves past 3 would wait for ever on pairs that no CTA writes.
        _waves_kernel[(2,)](
            scratch,
            step,
            out,
            3,
            Plan(grid=2, stages=(), regions=(), rows=1, chain=1).key,
            num_warps=8,
        )
        torch.cuda.synchronize()
        tag = launch_tag(5, 3)
        lanes = torch.arange(192)
        assert out[:192].tolist() == (lanes * 3 + tag).tolist()
        assert out[512:704].tolist() == (lanes * 5 + tag).tolist()
        assert (out[192:512] == -1).all() and (out[704:] == -1).all()

    # A poll whose pair never arrives, with a short timeout (11 << 10 ticks
    # of the 100 MHz clock: about 0.1 ms, checked every 64 retries) so that
    # the test is quick. It runs in its own process: the trap ends the
    # process's GPU context.
    _STUCK_POLL = textwrap.dedent("""
        import torch
        from tokenspeed_kernel_amd._triton import gl, gluon
        from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox
        from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan

        @gluon.jit
        def _stuck_kernel(scratch, step_ptr, out, PLAN: gl.constexpr):
            tag = mailbox.launch_tag(step_ptr, 0)
            lanes = gl.arange(
                0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps())
            )
            gl.store(out + lanes, mailbox.poll(scratch, lanes, tag, PLAN))

        scratch = torch.zeros(256, dtype=torch.int64, device="cuda")
        step = torch.zeros(1, dtype=torch.int32, device="cuda")
        out = torch.zeros(256, dtype=torch.int32, device="cuda")
        plan = Plan(grid=1, stages=(), regions=(), rows=1, chain=1,
                    poll_check=64, poll_timeout_shift=10)
        _stuck_kernel[(1,)](scratch, step, out, plan.key, num_warps=4)
        torch.cuda.synchronize()
        print("the poll returned")
        """)

    def _trapped(stderr: str) -> bool:
        """Whether a process's stderr reports a GPU trap. The trap reaches the
        host as a failed launch; some runtimes report it as an illegal memory
        access (hipErrorIllegalAddress). Where the runtime handles GPU
        exceptions itself (with GPU core dumps on), it aborts the queue and
        the process instead."""
        return any(
            m in stderr
            for m in (
                "launch failure",
                "illegal memory access",
                "HSA_STATUS_ERROR_EXCEPTION",
            )
        )

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
        assert _trapped(result.stderr), result.stderr

    # A stuck poll with a record buffer in a file (it outlives the process):
    # region "want" of row 1 carries this launch's tag except pair 37, which
    # holds the step before's. CTA 1 polls row 1 and traps; CTA 0 polls row
    # 0, which is complete, and writes no record.
    _STUCK_RECORD = textwrap.dedent("""
        import sys
        import torch
        from tokenspeed_kernel_amd._triton import gl, gluon
        from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, stuck
        from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (
            Plan,
            Stage,
            launch_tag,
        )

        @gluon.jit
        def _stuck_kernel(scratch, step_ptr, rec, out, PLAN: gl.constexpr):
            mb = mailbox.Mailbox(
                scratch, mailbox.launch_tag(step_ptr, 3), PLAN, None, None, rec
            )
            lanes = gl.arange(
                0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps())
            )
            row = gl.program_id(0)
            gl.store(out + row * 256 + lanes, mb.row(row).poll("want", lanes))

        plan = Plan(
            grid=2,
            stages=(Stage("s", 2, 0, reads=("want",)),),
            regions=(("pad", 5), ("want", 256)),
            rows=2,
            chain=1,
            external=("want",),
            poll_check=64,
            poll_timeout_shift=10,
        )
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        buf = stuck.stuck_buffer(plan.grid, 4, sys.argv[1], scratch.data_ptr(), 3)
        tag, old = launch_tag(5, 3), launch_tag(4, 3)
        for row in range(2):
            at = row * plan.row_pairs + plan.offset("want")
            scratch[at : at + 256] = (tag << 32) | torch.arange(256, device="cuda")
        at = plan.row_pairs + plan.offset("want") + 37
        scratch[at] = (old << 32) | 0x1234
        step = torch.full((1,), 5, dtype=torch.int32, device="cuda")
        out = torch.zeros(512, dtype=torch.int32, device="cuda")
        _stuck_kernel[(2,)](scratch, step, buf.tensor, out, plan.key, num_warps=4)
        torch.cuda.synchronize()
        print("the poll returned")
        """)

    def test_a_stuck_poll_records_what_it_waited_for(tmp_path):
        """The record names the region, row and pair of the stale pair, the
        launch's tag and the stale one, and only the stuck CTA writes one."""
        script = tmp_path / "stuck_record.py"  # a JIT function needs a file
        script.write_text(_STUCK_RECORD)
        path = tmp_path / "stuck.bin"
        result = subprocess.run(
            [sys.executable, str(script), str(path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "the poll returned" not in result.stdout
        assert _trapped(result.stderr), result.stderr
        plan = Plan(
            grid=2,
            stages=(Stage("s", 2, 0, reads=("want",)),),
            regions=(("pad", 5), ("want", 256)),
            rows=2,
            chain=1,
            external=("want",),
            poll_check=64,
            poll_timeout_shift=10,
        )
        t = stuck.read_file(str(path))
        head = stuck.header(t)
        assert head["rank"] == 3
        recs = stuck.records(t)
        assert [r["cta"] for r in recs] == [1]
        (rec,) = recs
        assert rec["expected"] == launch_tag(5, 3)
        assert rec["observed"] == launch_tag(4, 3)
        assert rec["payload"] == 0x1234
        pair = (rec["address"] - head["scratch_ptr"]) // 8
        assert stuck.locate(plan, pair) == ("want", 1, 37, 0)
        assert stuck.stages_at(plan, 1, "want") == ["s"]
        # The wait ran past the 0.11 ms timeout, by the realtime clock.
        assert rec["trap"] - rec["start"] >= POLL_TIMEOUT_MANTISSA << 10
        (line,) = stuck.decode(t, lambda layer: [plan] if layer == 3 else [])
        assert (
            "rank 3 CTA 1 wave 0: stage s waited for region want row 1 pair 37" in line
        )
        assert "step 4 layer 3" in line and "step 5 layer 3" in line

    # A trap may abort other waves before they record, even with a shared
    # deadline. Exercise the real flag timeout recorder, then join the waves
    # before the trap, so both completed records are guaranteed to survive.
    _TWO_WAVE_RECORD = textwrap.dedent("""
        import sys
        import torch
        from tokenspeed_kernel_amd._triton import gl, gluon
        from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, stuck, trace
        from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (
            POLL_TIMEOUT_MANTISSA,
        )

        # The tagged polls' bound at poll_timeout_shift 10, in ns.
        TIMEOUT_NS = gl.constexpr((POLL_TIMEOUT_MANTISSA << 10) * trace.TICK_NS)

        @gluon.jit
        def _two_wave_kernel(scratch, rec):
            lanes = gl.arange(0, 128, mailbox.lane_layout(2))
            ptrs = scratch + lanes // 64
            start = trace.now()
            ready = gl.atomic_poll(
                ptrs, gl.full((128,), (7 << 32) | 1, gl.int64, lanes.type.layout),
                sem="relaxed", scope="gpu",
                timeout_ns=gl.full((), TIMEOUT_NS, gl.int64),
            )
            now = trace.now()
            mailbox._record_flag_timeout(
                ptrs, ~ready, gl.full((), 7, gl.int32), start, now, rec
            )
            # The recorder drains system-scope stores; all waves must finish
            # their records before any wave stops the process.
            gl.barrier()
            mailbox._trap()

        scratch = torch.zeros(2, dtype=torch.int64, device="cuda")
        buf = stuck.stuck_buffer(1, 2, sys.argv[1], scratch.data_ptr(), 0)
        _two_wave_kernel[(1,)](scratch, buf.tensor, num_warps=2)
        torch.cuda.synchronize()
        print("the poll returned")
        """)

    def test_two_stuck_waves_keep_distinct_records_before_trapping(tmp_path):
        script = tmp_path / "two_wave_record.py"
        script.write_text(_TWO_WAVE_RECORD)
        path = tmp_path / "two_wave.bin"
        result = subprocess.run(
            [sys.executable, str(script), str(path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "the poll returned" not in result.stdout
        assert _trapped(result.stderr), result.stderr
        t = stuck.read_file(str(path))
        head = stuck.header(t)
        recs = stuck.records(t)
        assert [(r["cta"], r["wave"]) for r in recs] == [(0, 0), (0, 1)]
        assert [(r["address"] - head["scratch_ptr"]) // 8 for r in recs] == [0, 1]
        assert all(r["expected"] == 7 and r["observed"] == 0 for r in recs)
        assert all(r["trap"] > r["start"] for r in recs)

    # Flags have a known payload as well as a tag. One never-raised flag
    # retains either zero or the current tag with a zero payload.
    _STUCK_FLAG = textwrap.dedent("""
        import sys
        import torch
        from tokenspeed_kernel_amd._triton import gl, gluon
        from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, stuck
        from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (
            Plan,
            Stage,
            launch_tag,
        )

        @gluon.jit
        def _stuck_flag_kernel(scratch, step_ptr, rec, PLAN: gl.constexpr):
            mb = mailbox.Mailbox(
                scratch, mailbox.launch_tag(step_ptr, 3), PLAN, None, None, rec
            )
            layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [2, 2], [1, 0])
            r = gl.arange(0, 16, gl.SliceLayout(1, layout))[:, None]
            c = gl.arange(0, 32, gl.SliceLayout(0, layout))[None, :]
            mb.row(gl.program_id(0)).poll_flags("flag", r * 32 + c)

        plan = Plan(
            grid=2,
            stages=(Stage("s", 2, 0, reads=("flag",)),),
            regions=(("pad", 5), ("flag", 512)),
            rows=2,
            chain=1,
            external=("flag",),
            poll_timeout_shift=10,
        )
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        buf = stuck.stuck_buffer(plan.grid, 4, sys.argv[1], scratch.data_ptr(), 3)
        tag = launch_tag(5, 3)
        for row in range(2):
            at = row * plan.row_pairs + plan.offset("flag")
            scratch[at : at + 512] = (tag << 32) | 1
        at = plan.row_pairs + plan.offset("flag") + 37
        scratch[at] = int(sys.argv[2])
        step = torch.full((1,), 5, dtype=torch.int32, device="cuda")
        _stuck_flag_kernel[(2,)](scratch, step, buf.tensor, plan.key, num_warps=4)
        torch.cuda.synchronize()
        print("the poll returned")
        """)

    @pytest.mark.parametrize("matching_tag", [False, True])
    def test_a_flag_that_is_never_raised_stops_and_records(tmp_path, matching_tag):
        """A missing flag traps and records its pair, even if its tag matches."""
        script = tmp_path / "stuck_flag.py"
        script.write_text(_STUCK_FLAG)
        path = tmp_path / "stuck_flag.bin"
        observed = launch_tag(5, 3) << 32 if matching_tag else 0
        result = subprocess.run(
            [sys.executable, str(script), str(path), str(observed)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "the poll returned" not in result.stdout
        assert _trapped(result.stderr), result.stderr
        plan = Plan(
            grid=2,
            stages=(Stage("s", 2, 0, reads=("flag",)),),
            regions=(("pad", 5), ("flag", 512)),
            rows=2,
            chain=1,
            external=("flag",),
            poll_timeout_shift=10,
        )
        t = stuck.read_file(str(path))
        head = stuck.header(t)
        assert head["rank"] == 3
        (rec,) = stuck.records(t)
        assert rec["cta"] == 1
        assert rec["expected"] == launch_tag(5, 3)
        assert rec["observed"] == observed >> 32
        assert rec["payload"] == 0
        pair = (rec["address"] - head["scratch_ptr"]) // 8
        assert stuck.locate(plan, pair) == ("flag", 1, 37, 0)
        assert rec["trap"] - rec["start"] >= POLL_TIMEOUT_MANTISSA << 10

    # Mailbox.check_parity: a launch fails stop when its peer-region parity
    # repeats the last launch's (the copies must alternate, see Peers).
    def _parity_plan():
        return Plan(
            grid=1,
            stages=(Stage("s", 1, 0),),
            regions=(("pad", 3), ("parity", 1)),
            rows=1,
            chain=1,
        )

    @gluon.jit
    def _parity_kernel(scratch, step_ptr, parity, PLAN: gl.constexpr):
        mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step_ptr, 0), PLAN, None)
        mb.check_parity(parity)

    def test_alternating_launches_pass_the_parity_check():
        plan = _parity_plan()
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        step = torch.zeros(1, dtype=torch.int32, device="cuda")
        for parity in (0, 1, 0, 1):
            _parity_kernel[(1,)](scratch, step, parity, plan.key, num_warps=4)
            torch.cuda.synchronize()
            # The region keeps the last parity plus one.
            assert int(scratch[plan.offset("parity")]) == parity + 1

    _REPEATED_PARITY = textwrap.dedent("""
        import torch
        from tokenspeed_kernel_amd._triton import gl, gluon
        from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox
        from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage

        @gluon.jit
        def _parity_kernel(scratch, step_ptr, parity, PLAN: gl.constexpr):
            mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step_ptr, 0), PLAN, None)
            mb.check_parity(parity)

        plan = Plan(
            grid=1,
            stages=(Stage("s", 1, 0),),
            regions=(("pad", 3), ("parity", 1)),
            rows=1,
            chain=1,
        )
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        step = torch.zeros(1, dtype=torch.int32, device="cuda")
        for parity in (1, 1):
            _parity_kernel[(1,)](scratch, step, parity, plan.key, num_warps=4)
            torch.cuda.synchronize()
        print("the repeated parity passed")
        """)

    def test_a_repeated_parity_stops_the_process(tmp_path):
        """Two launches of the same parity in a row trap, as a stuck poll does."""
        script = tmp_path / "repeated_parity.py"  # a JIT function needs a file
        script.write_text(_REPEATED_PARITY)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "the repeated parity passed" not in result.stdout
        assert _trapped(result.stderr), result.stderr
