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
    def delay_pv():
        start = cute.arch.clock64()
        now = start
        while now - start < 200000:
            now = cute.arch.clock64()

    @dsl_user_op
    def delayed_wait(self, *, loc=None, ip=None):
        # Only PV waits on QK barrier 2. Let QK reach its next notification
        # first, including when the same CTA advances to another query.
        if self.barrier_id == 2:
            delay_pv()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-smem-slots", type=int, choices=(2, 3), required=True)
    _run_delayed_pv(parser.parse_args().bf16_smem_slots)
