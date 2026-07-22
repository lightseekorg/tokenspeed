"""Unit tests for the breakable CUDA graph core (Phase 1).

Captures a tiny ``Linear -> eager break -> Linear`` forward with
:class:`BreakableCapture`, mutates the static input buffer, replays, and asserts
the replayed output matches an eager recompute. This exercises the load-bearing
invariants in isolation -- segment splitting, the eager break handoff, shared
mempool address stability -- without touching the model or the hot path.
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
    active_forward,
    break_here,
    break_point,
    current_forward_ctx,
    is_breakable_capture_active,
    scrub_padding_tail,
    slice_to_real_tokens,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBreakableCudaGraph(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dev = "cuda"
        self.dtype = torch.float32
        self.n, self.d = 8, 16
        self.w1 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        self.w2 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        # Static input buffer: graphs read this address; live inputs are copied in.
        self.x_static = torch.zeros(self.n, self.d, device=self.dev, dtype=self.dtype)

    def _eager(self, x):
        """Reference forward: Linear -> relu (the "break" op) -> Linear."""
        h = x @ self.w1
        h = torch.relu(h)
        return h @ self.w2

    def _build_capture(self):
        """Capture the same forward with relu as an eager break."""
        cap = BreakableCapture()

        def forward():
            h = self.x_static @ self.w1
            # Break-output buffer, allocated in-segment so it is pool-pinned.
            dst = torch.empty_like(h)
            h = break_here(torch.relu, dst, h)
            return h @ self.w2

        # Warm up (cublas workspace / lazy init) before capture.
        for _ in range(3):
            forward()
        torch.cuda.synchronize()

        with cap:
            captured_out = forward()
        return cap, captured_out

    def test_break_here_passthrough_when_inactive(self):
        self.assertFalse(is_breakable_capture_active())
        h = torch.randn(self.n, self.d, device=self.dev, dtype=self.dtype)
        dst = torch.empty_like(h)
        out = break_here(torch.relu, dst, h)
        self.assertIs(out, dst)
        torch.testing.assert_close(out, torch.relu(h))

    def test_segments_split_at_break(self):
        cap, _ = self._build_capture()
        # Two graph segments (before/after relu) + one eager break = 3.
        self.assertEqual(cap.num_segments, 3)

    def test_replay_matches_eager(self):
        cap, captured_out = self._build_capture()
        for trial in range(5):
            new_x = torch.randn(self.n, self.d, device=self.dev, dtype=self.dtype)
            self.x_static.copy_(new_x)
            cap.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                captured_out, self._eager(new_x), msg=f"trial {trial}"
            )

    def test_replay_clears_unwritten_padding_at_break_handoff(self):
        """A padded replay must not pass an eager break's undefined tail onward."""
        state = {"valid_rows": self.n}

        class VarlenOp:
            @break_point
            def forward(self, x):
                out = torch.full_like(x, torch.nan)
                rows = state["valid_rows"]
                out[:rows].copy_(x[:rows])
                return out

        op = VarlenOp()

        def forward():
            h = self.x_static @ self.w1
            return op.forward(h) @ self.w2

        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        cap = BreakableCapture()
        with cap:
            captured_out = forward()

        state["valid_rows"] = 1
        new_x = torch.randn(self.n, self.d, device=self.dev, dtype=self.dtype)
        self.x_static.copy_(new_x)
        cap.replay(valid_rows=1)
        torch.cuda.synchronize()

        expected = torch.zeros_like(captured_out)
        expected[:1] = (new_x @ self.w1)[:1] @ self.w2
        torch.testing.assert_close(captured_out, expected, rtol=1e-4, atol=1e-4)

    def test_multiple_breaks_chain(self):
        """Many breaks (like a deep transformer) must chain correctly."""
        depth = 6
        ws = [
            torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
            for _ in range(depth + 1)
        ]

        def eager(x):
            h = x @ ws[0]
            for i in range(depth):
                h = torch.relu(h)  # the "break" op
                h = h @ ws[i + 1]
            return h

        cap = BreakableCapture()

        def forward():
            h = self.x_static @ ws[0]
            for i in range(depth):
                dst = torch.empty_like(h)
                h = break_here(torch.relu, dst, h)
                h = h @ ws[i + 1]
            return h

        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        with cap:
            captured_out = forward()

        # depth breaks => depth+1 graph segments + depth eager breaks.
        self.assertEqual(cap.num_segments, 2 * depth + 1)
        for trial in range(4):
            new_x = torch.randn(self.n, self.d, device=self.dev, dtype=self.dtype)
            self.x_static.copy_(new_x)
            cap.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured_out, eager(new_x), msg=f"trial {trial}")

    def test_nested_capture_rejected(self):
        with BreakableCapture():
            with self.assertRaises(RuntimeError):
                with BreakableCapture():
                    pass

    def test_scrub_padding_tail(self):
        # Zeros [num_real:] in place across tensors; skips None; no-op when unpadded.
        t1 = torch.ones(6, 3, device=self.dev, dtype=self.dtype)
        t2 = torch.ones(6, device=self.dev, dtype=self.dtype)
        scrub_padding_tail(4, t1, None, t2)
        self.assertTrue(bool((t1[:4] == 1).all()) and bool((t1[4:] == 0).all()))
        self.assertTrue(bool((t2[:4] == 1).all()) and bool((t2[4:] == 0).all()))
        # Unpadded (count == rows): untouched.
        t3 = torch.ones(4, 3, device=self.dev, dtype=self.dtype)
        scrub_padding_tail(4, t3)
        self.assertTrue(bool((t3 == 1).all()))

    def test_slice_to_real_tokens(self):
        """Leading [:num_real] per tensor in order; None and unpadded pass through."""
        a = torch.arange(6, device=self.dev)
        b = torch.arange(6, device=self.dev).view(6, 1)
        c = torch.arange(4, device=self.dev)  # already real length
        ra, rn, rb, rc = slice_to_real_tokens(4, a, None, b, c)
        self.assertEqual(ra.shape[0], 4)
        self.assertEqual(rb.shape[0], 4)
        self.assertIsNone(rn)
        self.assertIs(rc, c)  # no-op: not padded
        torch.testing.assert_close(ra, torch.arange(4, device=self.dev))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBucketedCapture(unittest.TestCase):
    """Per-token-bucket lazy capture with input padding (what PrefillGraph does).

    A token-shaped inner forward whose per-layer "attention" runs as an eager
    break is captured once per padded token bucket and replayed for any real token
    count <= bucket. Mirrors the runner's mechanics directly on BreakableCapture so
    the bucketing + padding + replay-parity invariant is unit-tested in isolation.
    """

    def setUp(self):
        torch.manual_seed(1)
        self.dev, self.dtype = "cuda", torch.float32
        self.d, self.depth = 16, 3
        self.buckets = [4, 8, 16]
        self.ws = [
            torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
            for _ in range(self.depth + 1)
        ]
        # Persistent static input buffer (max bucket); graph reads this address.
        self.x_static = torch.zeros(
            max(self.buckets), self.d, device=self.dev, dtype=self.dtype
        )
        self._captures: dict = {}
        self._outputs: dict = {}
        self._pool = None

    def _inner(self, n):
        """Inner forward over the leading ``n`` rows of the static buffer."""
        h = self.x_static[:n] @ self.ws[0]
        for i in range(self.depth):
            dst = torch.empty_like(h)
            h = break_here(torch.relu, dst, h)  # per-token "attention" break
            h = h @ self.ws[i + 1]
        return h

    def _eager(self, x):
        h = x @ self.ws[0]
        for i in range(self.depth):
            h = torch.relu(h)
            h = h @ self.ws[i + 1]
        return h

    def _run_bucketed(self, n):
        """Pad ``n`` up to a bucket, lazily capture/replay, return output[:n]."""
        idx = next(i for i, b in enumerate(self.buckets) if b >= n)
        bucket = self.buckets[idx]
        cap = self._captures.get(bucket)
        if cap is None:
            for _ in range(2):  # warmup
                self._inner(bucket)
            torch.cuda.synchronize()
            cap = BreakableCapture(pool=self._pool)
            with cap:
                out = self._inner(bucket)
            self._pool = self._pool or cap.pool
            cap.replay()  # capture doesn't execute; populate `out`
            self._captures[bucket], self._outputs[bucket] = cap, out
        else:
            cap.replay()
        return self._outputs[bucket][:n], bucket

    def test_replay_matches_eager_across_buckets(self):
        captured = set()
        for n in [3, 4, 5, 8, 11, 16, 1]:
            new_x = torch.randn(n, self.d, device=self.dev, dtype=self.dtype)
            self.x_static.zero_()  # scrub the padded tail
            self.x_static[:n].copy_(new_x)
            out, bucket = self._run_bucketed(n)
            captured.add(bucket)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out, self._eager(new_x), msg=f"n={n}", rtol=1e-4, atol=1e-4
            )
        # First sighting of each distinct bucket captured once; later n replay.
        self.assertEqual(set(self._captures), {4, 8, 16})


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBreakPointAndAmbientCtx(unittest.TestCase):
    """The ``@break_point`` decorator + ambient live-context rebind.

    Exercises the two properties the model refactor relies on: (1) a decorated
    method runs as an eager break under capture / a direct call otherwise, with
    its output buffer sized from the named ``out`` arg; (2) the ForwardContext a
    break reads is rebound to the LIVE ambient context at replay, so a graph
    captured with a dummy ctx replays correctly against a different live ctx.
    """

    def setUp(self):
        torch.manual_seed(2)
        self.dev, self.dtype = "cuda", torch.float32
        self.d = 16
        self.w0 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        self.w1 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        self.x_static = torch.zeros(8, self.d, device=self.dev, dtype=self.dtype)

    def test_passthrough_runs_method_when_not_capturing(self):
        """Off the capture path @break_point always runs the method.

        Including side effects and 0-row inputs -- the decorator never silently
        skips a method; 0-row / idle handling is each model's own explicit
        ``if hidden_states.shape[0] == 0`` guard.
        """
        calls = []

        class M:
            @break_point
            def forward(self, x, ctx):
                calls.append(x.shape[0])
                return x * ctx.scale

        m = M()
        self.assertFalse(is_breakable_capture_active())
        x = torch.randn(4, self.d, device=self.dev, dtype=self.dtype)
        out = m.forward(x, SimpleNamespace(scale=3.0))  # direct call
        torch.testing.assert_close(out, x * 3.0)
        # 0 rows: the method still runs and its output (not the input) is returned.
        empty = torch.zeros(0, self.d, device=self.dev, dtype=self.dtype)
        out0 = m.forward(empty, SimpleNamespace(scale=9.0))
        torch.testing.assert_close(out0, empty * 9.0)
        self.assertEqual(calls, [4, 0])  # method ran for both, including 0 rows

    def test_ambient_ctx_rebinds_at_replay(self):
        class M:
            def __init__(self, w):
                self.w = w

            @break_point
            def forward(self, x, ctx):
                # Reads the (live) ctx; output [tokens, d] matches arg ``x``.
                return torch.relu(x @ self.w) * ctx.scale

        m = M(self.w1)
        dummy = SimpleNamespace(scale=2.0)  # capture-time ctx
        live = SimpleNamespace(scale=5.0)  # replay-time ctx

        def outer():
            h = self.x_static @ self.w0  # captured graph segment
            return m.forward(h, dummy)  # eager break reading the ambient ctx

        for _ in range(3):  # warmup
            with active_forward(dummy):
                outer()
        torch.cuda.synchronize()
        cap = BreakableCapture()
        with active_forward(dummy):
            with cap:
                captured = outer()
        # 1 break => 2 graph segments + 1 eager break.
        self.assertEqual(cap.num_segments, 3)

        new_x = torch.randn(8, self.d, device=self.dev, dtype=self.dtype)
        self.x_static.copy_(new_x)
        with active_forward(live):  # replay against a DIFFERENT live ctx
            cap.replay()
        torch.cuda.synchronize()
        # The break must have used live.scale (5.0), not the captured dummy (2.0).
        expected = torch.relu((new_x @ self.w0) @ self.w1) * 5.0
        torch.testing.assert_close(captured, expected, rtol=1e-4, atol=1e-4)

    def test_break_reads_live_scalar_off_ambient_not_frozen_arg(self):
        """Scalar args freeze at capture; live values must come off the ambient ctx.

        Mirrors the hybrid attention-backend pattern: only the ambient ctx is
        rebound at replay, so a break needing a live scalar (forward_mode, bs)
        reads current_forward_ctx(), never its own frozen arg.
        """
        seen = {}

        class M:
            @break_point
            def forward(self, x, frozen_mode):
                amb = current_forward_ctx()
                seen["frozen"] = frozen_mode
                seen["live"] = amb.mode
                return x * amb.mult

        m = M()
        dummy = SimpleNamespace(mode="EXTEND", mult=1.0)  # capture-time ctx
        live = SimpleNamespace(mode="MIXED", mult=4.0)  # replay-time ctx

        def outer():
            h = self.x_static @ self.w0
            return m.forward(h, frozen_mode="EXTEND")  # frozen scalar arg

        for _ in range(3):
            with active_forward(dummy):
                outer()
        torch.cuda.synchronize()
        cap = BreakableCapture()
        with active_forward(dummy):
            with cap:
                captured = outer()

        new_x = torch.randn(8, self.d, device=self.dev, dtype=self.dtype)
        self.x_static.copy_(new_x)
        with active_forward(live):  # replay against a DIFFERENT live ctx
            cap.replay()
        torch.cuda.synchronize()
        # The positional/kw scalar arg stayed frozen at capture time...
        self.assertEqual(seen["frozen"], "EXTEND")
        # ...but the ambient read tracked the live ctx (mode + multiplier).
        self.assertEqual(seen["live"], "MIXED")
        torch.testing.assert_close(
            captured, new_x @ self.w0 * 4.0, rtol=1e-4, atol=1e-4
        )

    def test_break_point_computed_out_shape(self):
        """A NARROW break whose output shape matches no input (deepseek_v3 MLA-like)."""
        d2 = self.d // 2
        wv = torch.randn(self.d, d2, device=self.dev, dtype=self.dtype)

        class M:
            @break_point  # out-shape (d2 != input d) inferred from the actual output
            def attn(self, x):  # output last-dim d2 != input last-dim d
                return torch.relu(x) @ wv

        m = M()

        def outer():
            h = self.x_static @ self.w0  # captured segment
            return m.attn(h)  # narrow break, computed output shape

        for _ in range(3):
            outer()
        torch.cuda.synchronize()
        cap = BreakableCapture()
        with cap:
            captured = outer()
        self.assertEqual(tuple(captured.shape), (self.x_static.size(0), d2))

        new_x = torch.randn(8, self.d, device=self.dev, dtype=self.dtype)
        self.x_static.copy_(new_x)
        cap.replay()
        torch.cuda.synchronize()
        expected = torch.relu(new_x @ self.w0) @ wv
        torch.testing.assert_close(captured, expected, rtol=1e-4, atol=1e-4)

    def test_nested_break_inner_passes_through(self):
        """A broader @break_point overrides a nested one.

        Capture is inactive while the outer break runs eagerly, so an inner
        break called inside it passes straight through -- exactly one break.
        """
        seen = {"inner_active": None}

        class M:
            @break_point
            def inner(self, x):  # would-be default backend break
                seen["inner_active"] = is_breakable_capture_active()
                return torch.relu(x)

            @break_point
            def outer(self, x):  # broader override break
                return self.inner(x) @ self_w1  # noqa: F821

        self_w1 = self.w1
        m = M()

        def fwd():
            h = self.x_static @ self.w0  # captured segment
            return m.outer(h)  # broad break; inner passes through

        for _ in range(3):
            fwd()
        torch.cuda.synchronize()
        cap = BreakableCapture()
        with cap:
            captured = fwd()
        # Exactly one break (outer) => 2 graph segments + 1 break = 3.
        self.assertEqual(cap.num_segments, 3)
        self.assertIs(seen["inner_active"], False)  # inner saw inactive capture

        new_x = torch.randn(8, self.d, device=self.dev, dtype=self.dtype)
        self.x_static.copy_(new_x)
        cap.replay()
        torch.cuda.synchronize()
        expected = torch.relu(new_x @ self.w0) @ self.w1
        torch.testing.assert_close(captured, expected, rtol=1e-4, atol=1e-4)


class TestPrefillTokenBuckets(unittest.TestCase):
    """Pure-function tests for the prefill-graph token-bucket schedule (no GPU)."""

    @staticmethod
    def _cfg(**overrides):
        base = dict(
            prefill_graph_max_tokens=2048,
            disable_prefill_graph=False,
            chunked_prefill_size=2048,
            prefill_graph_capture_sizes=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_disabled(self):
        from tokenspeed.runtime.execution.prefill_graph import (
            get_prefill_token_buckets,
        )

        self.assertEqual(
            get_prefill_token_buckets(self._cfg(prefill_graph_max_tokens=0)), []
        )
        self.assertEqual(
            get_prefill_token_buckets(self._cfg(disable_prefill_graph=True)), []
        )

    def test_clamped_to_chunk(self):
        from tokenspeed.runtime.execution.prefill_graph import (
            get_prefill_token_buckets,
        )

        # No bucket above the chunk (see get_prefill_token_buckets for why).
        buckets = get_prefill_token_buckets(
            self._cfg(prefill_graph_max_tokens=8192, chunked_prefill_size=2048)
        )
        self.assertEqual(buckets[-1], 2048)

    def test_relative_ladder_properties(self):
        from tokenspeed.runtime.execution.prefill_graph import (
            get_prefill_token_buckets,
        )

        # Gaps bounded relatively (size/8) and absolutely (512); exact top; increasing.
        for max_tokens in (8192, 2048, 1500):
            buckets = get_prefill_token_buckets(
                self._cfg(
                    prefill_graph_max_tokens=max_tokens,
                    chunked_prefill_size=max_tokens,
                )
            )
            self.assertEqual(buckets[-1], max_tokens)
            self.assertEqual(buckets, sorted(set(buckets)))
            gaps = [b2 - b1 for b1, b2 in zip(buckets, buckets[1:])]
            for b1, g in zip(buckets, gaps):
                self.assertLessEqual(g, 512, f"cap violated at {b1}")
                if b1 >= 256:
                    self.assertLessEqual(g, max(b1 // 8, 16), f"relative bound at {b1}")

    def test_explicit_capture_sizes(self):
        from tokenspeed.runtime.execution.prefill_graph import (
            get_prefill_token_buckets,
        )

        # Explicit list overrides the ladder; clamped to max_tokens (always included).
        buckets = get_prefill_token_buckets(
            self._cfg(prefill_graph_capture_sizes=[256, 1024, 4096])
        )
        self.assertEqual(buckets, [256, 1024, 2048])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestWeakRefTensor(unittest.TestCase):
    """The non-owning-view op behind break-closure weak refs."""

    def test_alias_without_ownership(self):
        from tokenspeed.runtime.execution.breakable_cuda_graph import weak_ref_tensor

        x = torch.arange(12, device="cuda", dtype=torch.float32).view(3, 4)[:, 1:]
        w = weak_ref_tensor(x)
        if w is x:  # identity fallback (no C++ toolchain) -- still correct
            self.skipTest("weak_ref extension unavailable; identity fallback")
        # Aliases the same memory (incl. strides), sees writes, owns nothing.
        self.assertEqual(w.data_ptr(), x.data_ptr())
        self.assertEqual(w.stride(), x.stride())
        x.fill_(3.0)
        torch.cuda.synchronize()
        self.assertTrue(bool((w == 3.0).all()))
        # Non-tensor / CPU passthrough.
        self.assertIsNone(weak_ref_tensor(None))
        cpu = torch.ones(2)
        self.assertIs(weak_ref_tensor(cpu), cpu)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPoolReuseAcrossCaptures(unittest.TestCase):
    """Graph-pool memory must NOT grow per capture (the dense-ladder enabler).

    Allocator pool blocks are stream-keyed, so all BreakableCaptures must share
    one capture stream (class-level default) -- a fresh stream per capture makes
    pool memory grow with the SUM of bucket sizes instead of the max.
    """

    @staticmethod
    def _pool_mb():
        return (
            sum(
                s["total_size"]
                for s in torch.cuda.memory_snapshot()
                if s.get("segment_pool_id", (0, 0)) != (0, 0)
            )
            / 2**20
        )

    def test_same_and_smaller_captures_reuse_pool(self):
        d, d2, depth = 512, 2048, 4
        w1 = torch.randn(d, d2, device="cuda")
        w2 = torch.randn(d2, d, device="cuda")
        xbuf = torch.zeros(4096, d, device="cuda")

        def fwd(n):
            x = xbuf[:n]
            for _ in range(depth):
                h = x @ w1
                dst = torch.empty_like(h)
                h = break_here(torch.relu, dst, h)
                x = h @ w2
            return x

        pool = torch.cuda.graph_pool_handle()
        caps = []
        deltas = []
        for n in (4096, 2048, 4096):
            for _ in range(2):
                fwd(n)
            torch.cuda.synchronize()
            before = self._pool_mb()
            cap = BreakableCapture(pool=pool)  # shared default capture stream
            with cap:
                out = fwd(n)
            cap.replay()
            torch.cuda.synchronize()
            caps.append((cap, out))
            deltas.append(self._pool_mb() - before)
        # First capture claims ~peak-live; later ones must reuse it (small allowance).
        self.assertLess(deltas[1], max(2.0, deltas[0] * 0.1), f"deltas={deltas}")
        self.assertLess(deltas[2], max(2.0, deltas[0] * 0.1), f"deltas={deltas}")
        # Replays still valid after cross-capture reuse.
        for cap, _ in caps:
            cap.replay()
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
