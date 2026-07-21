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
from tokenspeed.runtime.utils.cuda_stream import StreamFork  # noqa: E402


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

    def test_multistream_segment_then_eager_break_reuses_events(self):
        """A joined graph fork may feed a break that reuses the same events.

        DeepSeek-V4 projects Q/KV on the capture stream and compressor inputs on
        an auxiliary stream, then its eager attention core reuses that
        ``StreamFork`` for cache writes. Exercise that exact ordering while also
        proving that both graph-produced break arguments refresh on replay.
        """
        stream_fork = StreamFork(torch.cuda.Stream())
        w2 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        w3 = torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
        calls = {"outer": 0, "core": 0}
        observed_rows = []

        class M:
            def forward(inner_self, x, ctx):
                calls["outer"] += 1
                with stream_fork.scope(enable=True) as fork:
                    q = x @ self.w0
                    with fork.branch():
                        score1 = x @ self.w1
                        score2 = score1 * 2.0
                return inner_self.core(q, score1, score2, ctx) @ w3

            @break_point
            def core(inner_self, q, score1, score2, ctx):
                calls["core"] += 1
                q, score1, score2 = slice_to_real_tokens(
                    ctx.real_tokens, q, score1, score2
                )
                observed_rows.append((q.shape[0], score1.shape[0], score2.shape[0]))
                with stream_fork.scope(enable=True) as fork:
                    main = torch.relu(q + score1) * ctx.scale
                    with fork.branch():
                        aux = (q - score2) @ w2
                return main + aux

        model = M()
        dummy = SimpleNamespace(scale=2.0, real_tokens=self.x_static.shape[0])

        for _ in range(3):
            with active_forward(dummy):
                model.forward(self.x_static, dummy)
        torch.cuda.synchronize()
        observed_rows.clear()

        cap = BreakableCapture()
        with active_forward(dummy):
            with cap:
                captured = model.forward(self.x_static, dummy)
        self.assertEqual(cap.num_segments, 3)
        outer_calls_after_capture = calls["outer"]
        core_calls_after_capture = calls["core"]

        for trial, (real_tokens, scale) in enumerate(((5, 3.0), (3, 5.0))):
            new_x = torch.randn(8, self.d, device=self.dev, dtype=self.dtype)
            self.x_static.copy_(new_x)
            live = SimpleNamespace(scale=scale, real_tokens=real_tokens)
            with active_forward(live):
                cap.replay()
            torch.cuda.synchronize()

            q = new_x @ self.w0
            score1 = new_x @ self.w1
            score2 = score1 * 2.0
            expected = (
                torch.relu(q[:real_tokens] + score1[:real_tokens]) * scale
                + (q[:real_tokens] - score2[:real_tokens]) @ w2
            ) @ w3
            self.assertEqual(captured.shape[0], self.x_static.shape[0])
            torch.testing.assert_close(
                captured[:real_tokens],
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"trial {trial}",
            )

        self.assertEqual(calls["outer"], outer_calls_after_capture)
        self.assertEqual(calls["core"], core_calls_after_capture + 2)
        self.assertEqual(
            observed_rows,
            [(8, 8, 8), (5, 5, 5), (3, 3, 3)],
        )

    def test_short_handoff_with_live_positions_and_same_shape_reuse(self):
        """Short breaks may share one handoff before row-local graph work.

        Mirrors DeepSeek-V4's eager attention-core -> graphed output-
        projection boundary.  Two same-shaped breaks reuse one handoff, each
        following graph segment consumes it before the next break overwrites it,
        and only the real prefix is refreshed on replay.  Poisoning the untouched
        tail proves that row-local work cannot contaminate real-token rows.
        """
        positions_static = torch.zeros(
            self.x_static.shape[0], device=self.dev, dtype=self.dtype
        )
        projection_weights = [
            torch.randn(self.d, self.d, device=self.dev, dtype=self.dtype)
            for _ in range(2)
        ]
        position_scales = (0.03125, -0.015625)
        observed_rows = []

        class M:
            def forward(inner_self, x, positions, ctx):
                h = x @ self.w0
                for layer, weight in enumerate(projection_weights):
                    h = inner_self.core(h, ctx, layer)
                    # The output projection remains in the following graph
                    # segment and reads the persistent, replay-refreshed positions.
                    h = (
                        torch.tanh(h + positions[:, None] * position_scales[layer])
                        @ weight
                    )
                return h

            @break_point
            def core(inner_self, h, ctx, layer):
                (h,) = slice_to_real_tokens(ctx.real_tokens, h)
                observed_rows.append((layer, h.shape[0]))
                return torch.sin(h + ctx.core_offsets[layer])

        def eager(x, positions, core_offsets):
            h = x @ self.w0
            for layer, weight in enumerate(projection_weights):
                h = torch.sin(h + core_offsets[layer])
                h = torch.tanh(h + positions[:, None] * position_scales[layer]) @ weight
            return h

        model = M()
        dummy = SimpleNamespace(
            real_tokens=self.x_static.shape[0], core_offsets=(0.125, -0.25)
        )
        positions_static.copy_(
            torch.arange(self.x_static.shape[0], device=self.dev, dtype=self.dtype)
        )

        for _ in range(3):
            with active_forward(dummy):
                model.forward(self.x_static, positions_static, dummy)
        torch.cuda.synchronize()
        observed_rows.clear()

        cap = BreakableCapture()
        with active_forward(dummy):
            with cap:
                captured = model.forward(self.x_static, positions_static, dummy)

        # graph -> break -> graph -> break -> graph, with both breaks sharing
        # the same shape-keyed destination.
        self.assertEqual(cap.num_segments, 5)
        self.assertEqual(len(cap._handoff), 1)
        handoff = next(iter(cap._handoff.values()))
        self.assertEqual(tuple(handoff.shape), tuple(self.x_static.shape))

        trials = (
            (5, (0.5, -0.75), "nan"),
            (3, (-0.375, 0.625), "inf"),
        )
        for trial, (real_tokens, core_offsets, poison_kind) in enumerate(trials):
            new_x = torch.randn_like(self.x_static)
            self.x_static.copy_(new_x)
            live_positions = (
                torch.arange(real_tokens, device=self.dev, dtype=self.dtype)
                * (trial + 2)
                + 17
            )
            positions_static.zero_()
            positions_static[:real_tokens].copy_(live_positions)

            tail = handoff[real_tokens:]
            if poison_kind == "nan":
                tail.fill_(float("nan"))
            else:
                row_poison = torch.where(
                    torch.arange(tail.shape[0], device=self.dev) % 2 == 0,
                    torch.tensor(float("inf"), device=self.dev),
                    torch.tensor(-float("inf"), device=self.dev),
                ).to(self.dtype)
                tail.copy_(row_poison[:, None].expand_as(tail))
            torch.cuda.synchronize()

            live = SimpleNamespace(real_tokens=real_tokens, core_offsets=core_offsets)
            with active_forward(live):
                cap.replay()
            torch.cuda.synchronize()

            expected = eager(new_x[:real_tokens], live_positions, core_offsets)
            self.assertEqual(captured.shape[0], self.x_static.shape[0])
            self.assertTrue(bool(torch.isfinite(captured[:real_tokens]).all()))
            torch.testing.assert_close(
                captured[:real_tokens],
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"trial {trial}",
            )
            self.assertTrue(bool((positions_static[real_tokens:] == 0).all()))
            if poison_kind == "nan":
                self.assertTrue(bool(torch.isnan(tail).all()))
            else:
                self.assertTrue(bool(torch.isinf(tail).all()))

        self.assertEqual(
            observed_rows,
            [(0, 8), (1, 8), (0, 5), (1, 5), (0, 3), (1, 3)],
        )

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


class TestPrefillDummyBatch(unittest.TestCase):
    """CPU-only checks for the prefill capture metadata wiring."""

    def test_dummy_batch_uses_input_lengths_for_paged_extend_metadata(self):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

        class RecordingBackend:
            uses_paged_cache_groups = True

            def init_forward_metadata(self, **kwargs):
                self.metadata = kwargs

        class RecordingDecodeWrapper:
            def _capture_paged_cache_block_tables(self, bs, token_to_kv_pool):
                self.call = (bs, token_to_kv_pool)
                return paged_cache_block_tables

        num_tokens = 4
        input_buffers = SimpleNamespace(
            input_ids_buf=torch.empty(num_tokens, dtype=torch.int32),
            out_cache_loc_buf=torch.empty(num_tokens, dtype=torch.int64),
            positions_buf=torch.empty(num_tokens, dtype=torch.int64),
            req_pool_indices_buf=torch.empty(1, dtype=torch.int32),
            seq_lens_buf=torch.empty(1, dtype=torch.int32),
            input_lengths_buf=torch.empty(1, dtype=torch.int32),
            extend_seq_lens_cpu=torch.empty(1, dtype=torch.int32),
            extend_prefix_lens_buf=torch.empty(1, dtype=torch.int32),
            extend_prefix_lens_cpu=torch.empty(1, dtype=torch.int32),
            dummy_kv_slot=7,
        )
        self.assertFalse(hasattr(input_buffers, "extend_seq_lens_buf"))

        backend = RecordingBackend()
        decode_wrapper = RecordingDecodeWrapper()
        token_to_kv_pool = object()
        paged_cache_block_tables = {"v4.swa_kv": torch.zeros(1, 1)}

        graph = PrefillGraph.__new__(PrefillGraph)
        graph.input_buffers = input_buffers
        graph.config = SimpleNamespace(device="cpu", world_size=1)
        graph.req_to_page = torch.ones(1, 2, dtype=torch.int32)
        graph.attn_backend = backend
        graph.token_to_kv_pool = token_to_kv_pool
        graph.drafter = None
        graph.dp_size = 1
        graph._prepare_replay_hook = None

        ctx = graph._make_dummy_batch(num_tokens, decode_wrapper)

        extend_seq_lens = backend.metadata["extend_seq_lens"]
        self.assertEqual(input_buffers.input_lengths_buf.tolist(), [num_tokens])
        self.assertEqual(extend_seq_lens.tolist(), [num_tokens])
        self.assertEqual(
            extend_seq_lens.data_ptr(), input_buffers.input_lengths_buf.data_ptr()
        )
        self.assertIs(extend_seq_lens._base, input_buffers.input_lengths_buf)
        self.assertEqual(input_buffers.extend_seq_lens_cpu.tolist(), [num_tokens])
        self.assertEqual(input_buffers.extend_prefix_lens_buf.tolist(), [0])
        self.assertEqual(input_buffers.extend_prefix_lens_cpu.tolist(), [0])
        self.assertEqual(backend.metadata["seq_lens"].tolist(), [num_tokens])

        self.assertEqual(ctx.bs, 1)
        self.assertEqual(ctx.num_extends, 1)
        self.assertEqual(ctx.input_num_tokens, num_tokens)
        self.assertEqual(ctx.forward_mode, ForwardMode.EXTEND)
        self.assertEqual(decode_wrapper.call, (1, token_to_kv_pool))
        self.assertIs(
            backend.metadata["paged_cache_block_tables"], paged_cache_block_tables
        )
        self.assertEqual(backend.metadata["num_tokens"], num_tokens)
        self.assertEqual(
            backend.metadata["positions"].tolist(), list(range(num_tokens))
        )
        # Capture-safety: the throwaway forward's KV writes must target the
        # reserved dummy slot and the dummy request's pages must point at
        # page 0, so capturing never touches real cache state.
        self.assertEqual(
            input_buffers.out_cache_loc_buf.tolist(),
            [input_buffers.dummy_kv_slot] * num_tokens,
        )
        self.assertEqual(input_buffers.input_ids_buf.tolist(), [1] * num_tokens)
        self.assertEqual(graph.req_to_page[0].tolist(), [0, 0])

    def test_replay_bucket_eligibility_gate(self):
        from tokenspeed.runtime.execution.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

        graph = PrefillGraph.__new__(PrefillGraph)
        graph.disable = False
        graph.dp_size = 1
        graph._captured_hidden_mode = CaptureHiddenMode.NULL
        graph.capture_buckets = [4]
        graph._captures = {4: object()}
        graph.config = SimpleNamespace(disable_cuda_graph_padding=False)

        def ctx(**over):
            base = dict(
                forward_mode=ForwardMode.EXTEND,
                num_extends=1,
                accept_lengths=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                input_num_tokens=4,
                global_num_tokens=None,
            )
            base.update(over)
            return SimpleNamespace(**base)

        # Eligible: pure extend, matching hidden mode, bucket captured.
        self.assertEqual(graph._replay_bucket(ctx()), 4)
        self.assertEqual(graph._replay_bucket(ctx(forward_mode=ForwardMode.MIXED)), 4)
        # Each ineligibility condition must force eager (None):
        self.assertIsNone(graph._replay_bucket(ctx(forward_mode=ForwardMode.DECODE)))
        self.assertIsNone(graph._replay_bucket(ctx(num_extends=0)))
        self.assertIsNone(graph._replay_bucket(ctx(accept_lengths=torch.tensor([1]))))
        self.assertIsNone(
            graph._replay_bucket(ctx(capture_hidden_mode=CaptureHiddenMode.FULL))
        )
        self.assertIsNone(graph._replay_bucket(ctx(input_num_tokens=9)))  # > bucket
        graph.disable = True
        self.assertIsNone(graph._replay_bucket(ctx()))

    def test_capture_unanimous_min_reduce_disables_on_any_rank_failure(self):
        from unittest import mock

        from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

        graph = PrefillGraph.__new__(PrefillGraph)
        graph.config = SimpleNamespace(world_group="w", world_size=2)

        # Single-rank / no group short-circuits to the local flag.
        solo = PrefillGraph.__new__(PrefillGraph)
        solo.config = SimpleNamespace(world_group=None, world_size=1)
        self.assertTrue(solo._capture_unanimous(True))
        self.assertFalse(solo._capture_unanimous(False))

        # Multi-rank: MIN all-reduce means any rank's False makes all False.
        for local_ok, peer_result, expected in ((True, 0, False), (True, 1, True)):
            with mock.patch(
                "tokenspeed.runtime.distributed.process_group_manager."
                "process_group_manager.get_process_group",
                return_value=object(),
            ), mock.patch("torch.distributed.all_reduce") as all_reduce:

                def fill(flag, op=None, group=None, _r=peer_result):
                    flag.fill_(_r)

                all_reduce.side_effect = fill
                self.assertEqual(graph._capture_unanimous(local_ok), expected)
                op = all_reduce.call_args.kwargs.get("op")
                self.assertIs(op, torch.distributed.ReduceOp.MIN)

    def test_prepare_finish_hooks_must_be_implemented_together(self):
        from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

        def resolve(model):
            g = PrefillGraph.__new__(PrefillGraph)
            g.inner_model = model
            g._prepare_replay_hook = getattr(
                model, "prepare_prefill_graph_replay", None
            )
            g._finish_replay_hook = getattr(model, "finish_prefill_graph_replay", None)
            if (g._prepare_replay_hook is None) != (g._finish_replay_hook is None):
                raise ValueError("both-or-neither")
            return g

        # Neither hook: fine (non-V4 models).
        self.assertIsNone(resolve(SimpleNamespace())._prepare_replay_hook)
        # Both hooks: fine.
        both = resolve(
            SimpleNamespace(
                prepare_prefill_graph_replay=lambda *a, **k: None,
                finish_prefill_graph_replay=lambda *a, **k: None,
            )
        )
        self.assertIsNotNone(both._prepare_replay_hook)
        # Exactly one: rejected.
        with self.assertRaises(ValueError):
            resolve(SimpleNamespace(prepare_prefill_graph_replay=lambda *a, **k: None))
        with self.assertRaises(ValueError):
            resolve(SimpleNamespace(finish_prefill_graph_replay=lambda *a, **k: None))

    def test_dummy_batch_calls_model_pregraph_hook_with_capture(self):
        from unittest import mock

        from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

        num_tokens = 4
        input_buffers = SimpleNamespace(
            input_ids_buf=torch.empty(num_tokens, dtype=torch.int32),
            out_cache_loc_buf=torch.empty(num_tokens, dtype=torch.int64),
            positions_buf=torch.empty(num_tokens, dtype=torch.int64),
            req_pool_indices_buf=torch.empty(1, dtype=torch.int32),
            seq_lens_buf=torch.empty(1, dtype=torch.int32),
            input_lengths_buf=torch.empty(1, dtype=torch.int32),
            extend_seq_lens_cpu=torch.empty(1, dtype=torch.int32),
            extend_prefix_lens_buf=torch.empty(1, dtype=torch.int32),
            extend_prefix_lens_cpu=torch.empty(1, dtype=torch.int32),
            dummy_kv_slot=7,
        )
        backend = SimpleNamespace(
            uses_paged_cache_groups=False,
            init_forward_metadata=mock.Mock(),
        )

        graph = PrefillGraph.__new__(PrefillGraph)
        graph.input_buffers = input_buffers
        graph.config = SimpleNamespace(device="cpu", world_size=1)
        graph.req_to_page = torch.ones(1, 2, dtype=torch.int32)
        graph.attn_backend = backend
        graph.token_to_kv_pool = object()
        graph.drafter = None
        graph.dp_size = 1
        graph._prepare_replay_hook = mock.Mock()

        ctx = graph._make_dummy_batch(num_tokens, None)

        hook = graph._prepare_replay_hook
        hook.assert_called_once()
        hook_ctx, hook_positions = hook.call_args.args
        self.assertIs(hook_ctx, ctx)
        self.assertEqual(hook_positions.shape[0], num_tokens)
        self.assertIs(hook_positions._base, input_buffers.positions_buf)
        self.assertEqual(hook.call_args.kwargs, {"capture": True})

    def test_replay_runs_pregraph_hooks_around_graph_and_clears_on_error(self):
        from unittest import mock

        from tokenspeed.runtime.execution import prefill_graph as prefill_graph_mod
        from tokenspeed.runtime.execution.context import ForwardContext
        from tokenspeed.runtime.execution.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from tokenspeed.runtime.execution.prefill_graph import (
            CapturedForward,
            PrefillGraph,
        )

        bucket = 4
        calls: list[str] = []
        input_buffers = SimpleNamespace(
            positions_buf=torch.arange(bucket, dtype=torch.int64),
        )

        graph = PrefillGraph.__new__(PrefillGraph)
        graph.disable = False
        graph.dp_size = 1
        graph.config = SimpleNamespace(
            device="cpu", world_size=1, disable_cuda_graph_padding=False
        )
        graph.capture_buckets = [bucket]
        graph.input_buffers = input_buffers
        graph._multimodal_input_embeds = None
        graph._embed_tokens = lambda ids: torch.zeros(ids.shape[0], 2)
        graph._input_embeds_buf = torch.zeros(bucket, 2)
        graph._captured_hidden_mode = CaptureHiddenMode.NULL
        graph._engaged_logged = {"text", "multimodal"}
        graph._captures = {
            bucket: SimpleNamespace(replay=lambda: calls.append("graph"))
        }
        graph._outputs = {
            bucket: CapturedForward(
                hidden_states=torch.zeros(bucket, 2), aux_hidden_states=None
            )
        }
        graph.text_model = SimpleNamespace(
            logits_processor=mock.Mock(return_value="logits"),
            lm_head=object(),
        )
        graph._prepare_replay_hook = mock.Mock(
            side_effect=lambda *a, **k: calls.append("prepare")
        )
        graph._finish_replay_hook = mock.Mock(
            side_effect=lambda ctx: calls.append("finish")
        )

        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=1,
            num_extends=1,
            input_num_tokens=bucket,
            forward_mode=ForwardMode.EXTEND,
        )
        with mock.patch.object(prefill_graph_mod, "LogitsMetadata"):
            result = graph.replay(ctx, torch.zeros(bucket, dtype=torch.int64))

        self.assertEqual(result, "logits")
        self.assertEqual(calls, ["prepare", "graph", "finish"])
        prepare = graph._prepare_replay_hook
        prepare_ctx, prepare_positions = prepare.call_args.args
        self.assertIs(prepare_ctx, ctx)
        self.assertIs(prepare_positions._base, input_buffers.positions_buf)
        self.assertEqual(prepare.call_args.kwargs, {"capture": False})
        graph._finish_replay_hook.assert_called_once_with(ctx)

        # A replay failure must still clear the armed flag via the finish hook.
        calls.clear()
        graph._captures[bucket] = SimpleNamespace(
            replay=mock.Mock(side_effect=RuntimeError("boom"))
        )
        with mock.patch.object(prefill_graph_mod, "LogitsMetadata"):
            with self.assertRaises(RuntimeError):
                graph.replay(ctx, torch.zeros(bucket, dtype=torch.int64))
        self.assertEqual(calls, ["prepare", "finish"])


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
