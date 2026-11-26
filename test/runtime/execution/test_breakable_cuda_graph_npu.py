"""Unit tests for the breakable graph core on NPU (piecewise capture).

The same Linear -> eager break -> Linear forward :class:`BreakableCapture`
captures on NPU as a sequence of ``torch.npu.NPUGraph`` segments with eager
breaks, sharing one pool. This is the NPU analogue of the CUDA
``test_breakable_cuda_graph.py`` invariants and the mechanism that lets
breakable prefill graphs (vLLM-Ascend PIECEWISE-style) run on NPU.
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.npu  # noqa: E402

from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
    active_forward,
    break_here,
    current_forward_ctx,
    current_valid_rows,
    is_breakable_capture_active,
    scrub_padding_tail,
    slice_to_real_tokens,
)

DEV = "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
class TestBreakableGraphNpu:
    def setup_method(self):
        torch.manual_seed(0)
        self.dev = DEV
        self.dtype = torch.bfloat16
        self.n, self.d = 8, 16
        self.w1 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        self.w2 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        self.x_static = torch.zeros(self.n, self.d, device=self.dev, dtype=self.dtype)

    def _eager(self, x):
        h = x @ self.w1
        h = torch.relu(h)
        return h @ self.w2

    def _build_capture(self):
        cap = BreakableCapture(device=self.dev)

        def forward():
            h = self.x_static @ self.w1
            dst = torch.empty_like(h)
            h = break_here(torch.relu, dst, h)
            return h @ self.w2

        for _ in range(3):
            forward()
        torch.npu.synchronize()

        with cap:
            captured_out = forward()
        return cap, captured_out

    def test_npu_breakable_capture_replays_eager(self):
        cap, captured_out = self._build_capture()
        assert cap.num_segments == 3  # 2 NPUGraph segments + 1 eager break
        for trial in range(5):
            new_x = torch.randn(self.n, self.d, device=self.dev, dtype=self.dtype)
            self.x_static.copy_(new_x)
            cap.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                captured_out, self._eager(new_x), msg=f"trial {trial}", rtol=0, atol=0
            )

    def test_npu_multiple_breaks_chain(self):
        depth = 6
        ws = [
            torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
            for _ in range(depth + 1)
        ]

        def eager(x):
            h = x @ ws[0]
            for i in range(depth):
                h = torch.relu(h)
                h = h @ ws[i + 1]
            return h

        cap = BreakableCapture(device=self.dev)

        def forward():
            h = self.x_static @ ws[0]
            for i in range(depth):
                dst = torch.empty_like(h)
                h = break_here(torch.relu, dst, h)
                h = h @ ws[i + 1]
            return h

        for _ in range(3):
            forward()
        torch.npu.synchronize()
        with cap:
            captured_out = forward()

        assert cap.num_segments == (2 * depth + 1)
        for trial in range(4):
            new_x = torch.randn(self.n, self.d, device=self.dev, dtype=self.dtype)
            self.x_static.copy_(new_x)
            cap.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured_out, eager(new_x), msg=f"trial {trial}")

    def test_npu_bucketed_capture_reuses_pool(self):
        """Prefill-style bucketed capture on NPU: buckets share one pool and
        replayed outputs match eager exactly (piecewise prefill graph)."""
        buckets = [4, 8, 16]
        self.ws = [
            torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
            for _ in range(4)
        ]
        self.x_static = torch.zeros(
            max(buckets), self.d, device=self.dev, dtype=self.dtype
        )

        def _inner(n):
            h = self.x_static[:n] @ self.ws[0]
            for i in range(3):
                dst = torch.empty_like(h)
                h = break_here(torch.relu, dst, h)
                h = h @ self.ws[i + 1]
            return h

        def _eager(x):
            h = x @ self.ws[0]
            for i in range(3):
                h = torch.relu(h)
                h = h @ self.ws[i + 1]
            return h

        captures, outputs, pool = {}, {}, None
        for bucket in buckets:
            for _ in range(2):
                _inner(bucket)
            torch.npu.synchronize()
            cap = BreakableCapture(pool=pool, device=self.dev)
            with cap:
                out = _inner(bucket)
            pool = pool or cap.pool
            cap.replay()
            torch.npu.synchronize()
            captures[bucket], outputs[bucket] = cap, out

        for n in [3, 4, 5, 8, 11, 16, 1]:
            bucket = next(b for b in buckets if b >= n)
            new_x = torch.randn(n, self.d, device=self.dev, dtype=self.dtype)
            self.x_static.zero_()
            self.x_static[:n].copy_(new_x)
            captures[bucket].replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                outputs[bucket][:n], _eager(new_x), msg=f"n={n}", rtol=0, atol=0
            )


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
class TestBreakableHelpersNpu:
    def test_passthrough_when_inactive(self):
        assert not is_breakable_capture_active()
        h = torch.randn(4, 8, device=DEV, dtype=torch.bfloat16)
        dst = torch.empty_like(h)
        out = break_here(torch.relu, dst, h)
        assert out is dst
        torch.testing.assert_close(out, torch.relu(h))

    def test_scrub_and_slice(self):
        t1 = torch.ones(6, 3, device=DEV, dtype=torch.bfloat16)
        scrub_padding_tail(4, t1)
        assert bool((t1[:4] == 1).all()) and bool((t1[4:] == 0).all())
        a = torch.arange(6, device=DEV)
        (ra,) = slice_to_real_tokens(4, a)
        assert ra.shape[0] == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
