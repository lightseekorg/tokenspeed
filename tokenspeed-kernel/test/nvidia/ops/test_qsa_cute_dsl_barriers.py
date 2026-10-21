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

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform


def _run_delayed_pv(bf16_smem_slots: int) -> None:
    """Delay PV arrivals while replaying live and fully padded query CTAs."""
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse as sparse
    from cutlass.cutlass_dsl import dsl_user_op
    from test_qsa_sparse_attention import _reference

    original_wait = pipeline.NamedBarrier.arrive_and_wait

    @cute.jit
    def delay_pv(barrier_id):
        # Barrier IDs can be staged values, including the alternating PV phase.
        if barrier_id == 2 or barrier_id == 6:
            start = cute.arch.clock64()
            now = start
            while now - start < 200000:
                now = cute.arch.clock64()

    @dsl_user_op
    def delayed_wait(self, *, loc=None, ip=None):
        # Only PV waits on QK barriers 2/6. Let QK reach its next notification
        # first, including when the same CTA advances to another query.
        delay_pv(self.barrier_id)
        original_wait(self, loc=loc, ip=ip)

    def select_config(
        num_rows: int,
        head_tiles_per_row: int,
        bf16_kv: bool,
        sm_count: int,
        wide_cluster_capacity: int,
    ) -> tuple[int, int, int, bool]:
        del num_rows, head_tiles_per_row, bf16_kv, sm_count, wide_cluster_capacity
        return 4, 1, bf16_smem_slots, True

    torch.manual_seed(239)
    q = torch.randn(8, 24, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4096, 2, 256, device="cuda", dtype=torch.bfloat16) * 0.25
    v = torch.randn_like(k) * 0.25
    k[:2].zero_()
    v[:2].zero_()
    selected = torch.randint(2, 4096, (8, 2051), device="cuda", dtype=torch.int32)
    selected[:, 7::11] = -1
    selected[4:].fill_(-1)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline.NamedBarrier, "arrive_and_wait", delayed_wait)
        patch.setattr(sparse, "_select_launch_config", select_config)

        def forward():
            return sparse.kernel(
                q,
                k,
                v,
                selected,
                scale=1 / 16,
                max_seqlen_q=4,
                k_scale=None,
                v_scale=None,
                enable_pdl=True,
            )

        eager = forward()
        torch.cuda.synchronize()
        expected = _reference(q, k, v, selected, 1 / 16)
        torch.testing.assert_close(eager, expected, rtol=0.02, atol=0.002)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()
        for _ in range(64):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        assert torch.count_nonzero(output[4:]).item() == 0

        # Reuse the same graph with changed data and the opposite padding CTA.
        k.neg_()
        v.add_(0.125)
        k[:2].zero_()
        v[:2].zero_()
        selected.copy_(selected.flip(0))
        for _ in range(64):
            graph.replay()
        torch.cuda.synchronize()
        expected = _reference(q, k, v, selected, 1 / 16)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)
        assert torch.count_nonzero(output[:4]).item() == 0


@pytest.mark.parametrize("bf16_smem_slots", [2, 3])
def test_qsa_decode_barriers_allow_delayed_pv(bf16_smem_slots: int) -> None:
    if not torch.cuda.is_available() or current_platform().arch_version not in (
        ArchVersion(10, 0),
        ArchVersion(10, 3),
    ):
        pytest.skip("CuTe DSL QSA requires NVIDIA SM100 or SM103")
    # Isolate a synchronization regression so a stalled kernel cannot leave
    # the pytest process blocked indefinitely at CUDA synchronization.
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--bf16-smem-slots",
            str(bf16_smem_slots),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, (
        f"Delayed PV failed with {bf16_smem_slots} shared slots:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _run_delayed_stream_producer(enable_pdl: bool) -> None:
    """Delay one producer warp after publishing each candidate-count update."""
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    from cutlass.cutlass_dsl import dsl_user_op
    from test_qsa_stream_topk import _assert_matches_reference
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_stream_topk import kernel

    original_wait = pipeline.NamedBarrier.arrive_and_wait

    @cute.jit
    def delay_producer(barrier_id):
        if barrier_id == 11 or barrier_id == 12:
            tidx, _, _ = cute.arch.thread_idx()
            if (tidx // 32) % 4 == 3:
                start = cute.arch.clock64()
                now = start
                while now - start < 20000:
                    now = cute.arch.clock64()

    @dsl_user_op
    def delayed_wait(self, *, loc=None, ip=None):
        original_wait(self, loc=loc, ip=ip)
        delay_producer(self.barrier_id)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(83)
    rows, pages = 136, 128
    query = torch.randn(rows, 5, 128, device="cuda", dtype=torch.bfloat16)[:, :4]
    keys = torch.randn(pages * 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    complete = torch.full((rows,), pages * 64, device="cuda", dtype=torch.int32)

    with pytest.MonkeyPatch.context() as patch:
        # Perturb real warp scheduling without changing scores or metadata.
        patch.setattr(pipeline.NamedBarrier, "arrive_and_wait", delayed_wait)

        def forward():
            return kernel(
                query,
                keys,
                table,
                requests,
                complete,
                page_size=64,
                block_topk=512,
                queries_per_request=rows,
                max_partial_bytes=0,
                enable_pdl=enable_pdl,
            )

        eager = forward()
        torch.cuda.synchronize()
        _assert_matches_reference(
            query, keys, table, requests, complete, eager, 64, 512
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selected = forward()
        for frontiers in (
            [8192] * 8,
            [0, 511, 512, 513, 1023, 1024, 6143, 8192],
            [512] * 4 + [8192] * 4,
            [8192] * 4 + [512] * 4,
            [0] * 8,
        ):
            complete.copy_(
                torch.tensor(frontiers * (rows // len(frontiers)), device="cuda")
            )
            for _ in range(3):
                graph.replay()
            _assert_matches_reference(
                query, keys, table, requests, complete, selected, 64, 512
            )


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_qsa_stream_producer_count_snapshot_allows_delayed_warp(enable_pdl: bool):
    if not torch.cuda.is_available() or current_platform().arch_version not in (
        ArchVersion(10, 0),
        ArchVersion(10, 3),
    ):
        pytest.skip("CuTe DSL QSA requires NVIDIA SM100 or SM103")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stream-producer-pdl",
            "on" if enable_pdl else "off",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"Delayed stream producer failed with PDL={enable_pdl}:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bf16-smem-slots", type=int, choices=(2, 3))
    group.add_argument("--stream-producer-pdl", choices=("on", "off"))
    args = parser.parse_args()
    if args.bf16_smem_slots is not None:
        _run_delayed_pv(args.bf16_smem_slots)
    else:
        _run_delayed_stream_producer(args.stream_producer_pdl == "on")
