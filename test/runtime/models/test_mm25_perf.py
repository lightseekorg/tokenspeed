"""
MiniMax-M2.5 NVFP4 (TP2) single-request perf & quality tests.

Guards against "silly breakage" on the mm25 path by exercising:
  - baseline (overlap + cudagraph): stream decode TPS floor + non-stream e2e
    TPS floor + sampling (flashinfer) smoke
  - no cudagraph: short-gen exact-string match against baseline reference
  - no overlap: stream TPS strictly lower than overlap baseline + short-gen
    exact-string match
  - xgrammar JSON (poem schema): stream decode TPS floor + JSON validity
  - EAGLE3 spec: stream TPS floor + acceptance-length floor (≥ 2.0)

Targets B200 2-GPU runners (NVFP4 requires Blackwell).

Calibrated 2026-04-29 on 2×B200 running nvidia/MiniMax-M2.5-NVFP4;
thresholds set with ~5 TPS margin below measured values after the
trtllm decode-kernel-for-spec routing:
  - baseline stream decode TPS ≈ 217 → floor 212
  - baseline non-stream e2e (384 tok) ≈ 209 → floor 200
  - xgrammar JSON stream decode TPS ≈ 217 → floor 212
  - overlap vs no-overlap stream TPS ratio ≈ 0.78 → cap 0.85
  - EAGLE3 stream decode TPS ≈ 321, accept_len ≈ 2.94 → floors 300 / 2.0

Usage:
    cd test/runtime
    python3 -m unittest models.test_mm25_perf -v
    python3 -m unittest models.test_mm25_perf.TestMiniMaxM25Perf.test_baseline -v

Env overrides:
    MM25_MODEL                 default nvidia/MiniMax-M2.5-NVFP4
    MM25_DRAFT                 default thoughtworks/MiniMax-M2.5-Eagle3
    MM25_MIN_STREAM_TPS        default 212
    MM25_MIN_NONSTREAM_TPS     default 200
    MM25_MIN_XGRAMMAR_TPS      default 212
    MM25_MIN_SPEC_TPS          default 300
    MM25_MIN_ACCEPT_LEN        default 2.0
    MM25_MAX_NO_OVERLAP_RATIO  default 0.85
"""

import json
import os
import subprocess
import sys
import time
import unittest
from typing import Dict, List, Optional, Tuple

import requests

# /test on sys.path so "ci_system.ci_register" resolves from test/ci_system/.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=2400, suite="runtime-minimax-m2")

from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.utils.process import kill_process_tree  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────

MODEL = os.environ.get("MM25_MODEL", "nvidia/MiniMax-M2.5-NVFP4")
DRAFT = os.environ.get("MM25_DRAFT", "thoughtworks/MiniMax-M2.5-Eagle3")
WORLD_SIZE = 2
SERVER_LAUNCH_TIMEOUT = 900
REQUEST_TIMEOUT = 300

MIN_STREAM_TPS = float(os.environ.get("MM25_MIN_STREAM_TPS", "212"))
MIN_NONSTREAM_TPS = float(os.environ.get("MM25_MIN_NONSTREAM_TPS", "200"))
MIN_XGRAMMAR_TPS = float(os.environ.get("MM25_MIN_XGRAMMAR_TPS", "212"))
MIN_SPEC_TPS = float(os.environ.get("MM25_MIN_SPEC_TPS", "300"))
MIN_ACCEPT_LEN = float(os.environ.get("MM25_MIN_ACCEPT_LEN", "2.0"))
MAX_NO_OVERLAP_RATIO = float(os.environ.get("MM25_MAX_NO_OVERLAP_RATIO", "0.85"))

# Long enough to amortize TTFT and keep decode steady-state.
PERF_MAX_TOKENS = 384

# Broad, open-ended prompt that naturally produces long fluent output from a
# reasoning model (won't bottom out before PERF_MAX_TOKENS).
PERF_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Explain the history and cultural significance of the Renaissance "
            "period in Europe. Cover its origins, key figures, artistic "
            "innovations, scientific developments, and enduring legacy."
        ),
    }
]

# Quality prompts — use substring match. MiniMax-M2.5 is a reasoning model
# that emits a <think>…</think> prefix; needs ~200 tokens for the thinking to
# conclude and the answer to appear.
QUALITY_MAX_TOKENS = 256
QUALITY_CHECKS = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Reply with just the city name.",
            }
        ],
        "expected": "Paris",
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ],
        "expected": "4",
    },
]

# Determinism prompt — short, fixed; compared byte-exact across configs.
DETERMINISM_MESSAGES = [
    {"role": "user", "content": "Reply with exactly the single word: hello"}
]
DETERMINISM_MAX_TOKENS = 16

# Poem schema. With --reasoning-parser the engine wraps json_schema in
# a structural tag so the model thinks before emitting JSON.
POEM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["title", "content"],
    "additionalProperties": False,
}
POEM_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Write an original poem about the ocean at dusk, at least 12 "
            "lines. Return JSON with fields title (string) and content "
            "(string, the full poem with line breaks as \\n)."
        ),
    }
]
XGRAMMAR_MAX_TOKENS = 4096  # reasoning + JSON both fit; 1024 occasionally
# runs out before the JSON channel opens, leaving ``content=''``.
# Floor that guards "model actually reasoned before the JSON". Measured
# ~2000 tok on MiniMax-M2.5; 300 gives plenty of margin while still
# catching a regression that drops the structural-tag wrap (in which
# case xgrammar locks onto `{` at token 0 and we'd see ~30-150 tok).
MIN_XGRAMMAR_GEN_TOKENS = 300

# Base args. Notes:
#  - sampling-backend flashinfer: exercises the flashinfer sampling path on
#    every test.
#  - reasoning-parser minimax: MiniMax-M2.5 emits <think>…</think>. With
#    reasoning_parser set, xgrammar defers the response-format constraint
#    past the reasoning channel via a structural tag, so grammar-constrained
#    tests (test_xgrammar) still get to think freely before writing JSON.
#  - mem-fraction-static 0.50 / kvstore-ratio 1.0: shrink init footprint so
#    the server comes up in ~60s and leaves headroom for the EAGLE3 draft
#    model.
BASE_ARGS: Tuple[str, ...] = (
    "--trust-remote-code",
    "--attention-backend",
    "trtllm",
    "--block-size",
    "32",
    "--moe-backend",
    "flashinfer_trtllm",
    "--sampling-backend",
    "flashinfer",
    "--reasoning-parser",
    "minimax",
    "--max-num-seqs",
    "4",
    "--max-cudagraph-capture-size",
    "4",
    "--gpu-memory-utilization",
    "0.50",
    "--kvstore-ratio",
    "1.0",
)

_server_port = 23100


def _next_port() -> int:
    global _server_port
    p = _server_port
    _server_port += 1
    return p


# ── Server lifecycle ─────────────────────────────────────────────────


def _serve_server(port: int, extra_args=()) -> subprocess.Popen:
    # Use `python -m tokenspeed.cli serve` rather than the `ts` console
    # script so we don't depend on PATH setup in the CI runner.
    cmd = [
        sys.executable,
        "-m",
        "tokenspeed.cli",
        "serve",
        "--model",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--world-size",
        str(WORLD_SIZE),
        *BASE_ARGS,
        *extra_args,
    ]
    return subprocess.Popen(cmd, env=os.environ.copy())


def _wait_for_server(port: int, timeout: int = SERVER_LAUNCH_TIMEOUT) -> bool:
    url = f"http://127.0.0.1:{port}/readiness"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ── Request helpers ──────────────────────────────────────────────────


def _chat_nonstream(
    port: int,
    messages,
    max_tokens: int,
    response_format: Optional[Dict] = None,
    **sampling,
) -> Tuple[str, int, float, Dict]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        **sampling,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    t0 = time.time()
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    # With --reasoning-parser, content is post-</think>; for substring
    # quality checks we want either channel.
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    full = (reasoning + "\n" + content) if reasoning else content
    completion_tokens = data["usage"]["completion_tokens"]
    return full, completion_tokens, elapsed, data.get("usage", {})


def _chat_stream(
    port: int,
    messages,
    max_tokens: int,
    response_format: Optional[Dict] = None,
    **sampling,
) -> Tuple[str, int, float, float, Dict]:
    """
    Returns (content, completion_tokens, ttft_seconds, decode_elapsed_seconds, usage).
    decode_elapsed excludes the first-token window (measured from first content
    chunk timestamp to the last content chunk timestamp).
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sampling,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    t_start = time.time()
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    pieces: List[str] = []
    usage: Dict = {}

    with requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                # With --reasoning-parser, tokens arrive as either content or
                # reasoning_content (same decode cost). Count both so the TPS
                # reflects full decode throughput, not just the post-think
                # tail. Keep `pieces` as only the final content — callers
                # like the xgrammar test parse that as JSON.
                reasoning_piece = delta.get("reasoning_content")
                content_piece = delta.get("content")
                if reasoning_piece or content_piece:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    if content_piece:
                        pieces.append(content_piece)

    content = "".join(pieces)
    completion_tokens = int(usage.get("completion_tokens", 0))
    ttft = (t_first - t_start) if t_first else 0.0
    decode_elapsed = (
        (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    )
    return content, completion_tokens, ttft, decode_elapsed, usage


def _stream_decode_tps(completion_tokens: int, decode_elapsed: float) -> float:
    # Exclude the first token from the rate (TTFT window).
    if decode_elapsed <= 0 or completion_tokens <= 1:
        return 0.0
    return (completion_tokens - 1) / decode_elapsed


def _e2e_tps(completion_tokens: int, elapsed: float) -> float:
    if elapsed <= 0 or completion_tokens <= 0:
        return 0.0
    return completion_tokens / elapsed


def _run_quality_checks(self, port: int, label: str):
    for i, q in enumerate(QUALITY_CHECKS):
        content, _, _, _ = _chat_nonstream(
            port, q["messages"], max_tokens=QUALITY_MAX_TOKENS
        )
        self.assertIn(
            q["expected"],
            content,
            f"[{label}] quality check {i}: expected {q['expected']!r} "
            f"in reply {content!r}",
        )


# ── Tests ────────────────────────────────────────────────────────────


class TestMiniMaxM25Perf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: switch to amd/MiniMax-M2.5-MXFP4 on AMD.
        if current_platform().is_amd:
            raise unittest.SkipTest("Skip NVFP4 on AMD")

    def _with_server(self, extra_args, fn, launch_timeout=SERVER_LAUNCH_TIMEOUT):
        port = _next_port()
        proc = _serve_server(port, extra_args)
        try:
            if not _wait_for_server(port, timeout=launch_timeout):
                self.fail(
                    f"Server did not become ready within {launch_timeout}s "
                    f"(args={extra_args})"
                )
            return fn(port)
        finally:
            kill_process_tree(proc.pid)
            # Brief delay so the kernel releases GPU memory before next launch.
            time.sleep(10)

    # Baseline: overlap + cudagraph (defaults). TPS floors + quality + sampling.
    def test_baseline(self):
        def run(port):
            # Two full-length warmups: the first decode request after server
            # start runs noticeably slower (GPU state, prefix/kv cache not
            # populated) — reading those numbers as steady-state would be
            # noisy. Run a perf-sized non-stream and stream request before we
            # measure.
            _chat_nonstream(port, PERF_MESSAGES, max_tokens=PERF_MAX_TOKENS)
            _chat_stream(port, PERF_MESSAGES, max_tokens=PERF_MAX_TOKENS)

            # Stream decode TPS (excludes first token). Take best of 2 to
            # tolerate ~5-10% run-to-run variance from CUDA scheduling noise.
            stream_tps_runs = []
            for _ in range(2):
                _, tok_s, ttft, decode_elapsed, _ = _chat_stream(
                    port, PERF_MESSAGES, max_tokens=PERF_MAX_TOKENS
                )
                stream_tps_runs.append(
                    (
                        tok_s,
                        ttft,
                        decode_elapsed,
                        _stream_decode_tps(tok_s, decode_elapsed),
                    )
                )
            best = max(stream_tps_runs, key=lambda r: r[3])
            tok_s, ttft, decode_elapsed, tps_s = best
            for i, (t, f, d, x) in enumerate(stream_tps_runs):
                print(
                    f"[baseline stream r{i}] tok={t} ttft={f:.3f}s "
                    f"decode={d:.3f}s decode_tps={x:.1f}"
                )
            print(f"[baseline stream best] decode_tps={tps_s:.1f}")
            self.assertGreaterEqual(tok_s, PERF_MAX_TOKENS // 2)
            self.assertGreaterEqual(
                tps_s,
                MIN_STREAM_TPS,
                f"best-of-2 stream decode TPS {tps_s:.1f} < floor {MIN_STREAM_TPS}",
            )

            # Non-stream e2e TPS (includes TTFT). Best-of-2 as well.
            ns_runs = []
            for _ in range(2):
                _, tok_ns, elapsed_ns, _ = _chat_nonstream(
                    port, PERF_MESSAGES, max_tokens=PERF_MAX_TOKENS
                )
                ns_runs.append((tok_ns, elapsed_ns, _e2e_tps(tok_ns, elapsed_ns)))
            best_ns = max(ns_runs, key=lambda r: r[2])
            tok_ns, elapsed_ns, tps_ns = best_ns
            for i, (t, e, x) in enumerate(ns_runs):
                print(
                    f"[baseline non-stream r{i}] tok={t} elapsed={e:.3f}s "
                    f"e2e_tps={x:.1f}"
                )
            print(f"[baseline non-stream best] e2e_tps={tps_ns:.1f}")
            self.assertGreaterEqual(tok_ns, PERF_MAX_TOKENS // 2)
            self.assertGreaterEqual(
                tps_ns,
                MIN_NONSTREAM_TPS,
                f"best-of-2 non-stream e2e TPS {tps_ns:.1f} < floor {MIN_NONSTREAM_TPS}",
            )

            # Sampling (flashinfer backend): temperature > 0, top_p < 1.
            # Only asserts the path works & produces non-empty output.
            content_samp, tok_samp, _, _ = _chat_nonstream(
                port,
                PERF_MESSAGES,
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
            )
            print(
                f"[baseline sampling T=0.7 top_p=0.9] tok={tok_samp} "
                f"preview={content_samp[:80]!r}"
            )
            self.assertGreater(len(content_samp), 0)
            self.assertGreaterEqual(tok_samp, 32)

            _run_quality_checks(self, port, "baseline")

        self._with_server((), run)

    # Content-determinism helper: baseline short-gen output used as reference.
    def _capture_reference_short_gen(self) -> str:
        def run(port):
            content, _, _, _ = _chat_nonstream(
                port,
                DETERMINISM_MESSAGES,
                max_tokens=DETERMINISM_MAX_TOKENS,
            )
            return content

        return self._with_server((), run)

    # --enforce-eager: short-gen output must equal baseline reference.
    # No speed floor (eager is slower by design).
    def test_no_cudagraph(self):
        reference = self._capture_reference_short_gen()
        print(f"[no_cudagraph ref] {reference!r}")

        def run(port):
            content, _, _, _ = _chat_nonstream(
                port,
                DETERMINISM_MESSAGES,
                max_tokens=DETERMINISM_MAX_TOKENS,
            )
            print(f"[no_cudagraph actual] {content!r}")
            self.assertEqual(
                content,
                reference,
                "short-gen output under --enforce-eager must match baseline",
            )
            _run_quality_checks(self, port, "no_cudagraph")

        self._with_server(("--enforce-eager",), run)

    # --disable-overlap-schedule: TPS strictly below overlap + exact short-gen match.
    def test_overlap_vs_no_overlap(self):
        def measure(port):
            _, tok, _, decode_elapsed, _ = _chat_stream(
                port, PERF_MESSAGES, max_tokens=PERF_MAX_TOKENS
            )
            ref_short, _, _, _ = _chat_nonstream(
                port,
                DETERMINISM_MESSAGES,
                max_tokens=DETERMINISM_MAX_TOKENS,
            )
            return _stream_decode_tps(tok, decode_elapsed), ref_short

        overlap_tps, overlap_short = self._with_server((), measure)
        no_overlap_tps, no_overlap_short = self._with_server(
            ("--disable-overlap-schedule",), measure
        )

        print(
            f"[overlap vs no-overlap] overlap={overlap_tps:.1f} "
            f"no_overlap={no_overlap_tps:.1f} "
            f"ratio={no_overlap_tps / max(overlap_tps, 1e-6):.3f}"
        )
        print(f"[overlap short]    {overlap_short!r}")
        print(f"[no_overlap short] {no_overlap_short!r}")
        self.assertLess(
            no_overlap_tps,
            overlap_tps * MAX_NO_OVERLAP_RATIO,
            f"no-overlap TPS ({no_overlap_tps:.1f}) should be < "
            f"{MAX_NO_OVERLAP_RATIO:.2f} × overlap ({overlap_tps:.1f})",
        )
        self.assertEqual(
            no_overlap_short,
            overlap_short,
            "short-gen output under --disable-overlap-schedule must match overlap",
        )

    # xgrammar poem: stream decode TPS + JSON validity.
    def test_xgrammar(self):
        def run(port):
            _chat_nonstream(port, PERF_MESSAGES, max_tokens=64)  # warmup

            content, tok, ttft, decode_elapsed, _ = _chat_stream(
                port,
                POEM_MESSAGES,
                max_tokens=XGRAMMAR_MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "Poem", "schema": POEM_SCHEMA},
                },
            )
            tps = _stream_decode_tps(tok, decode_elapsed)
            print(
                f"[xgrammar poem stream] tok={tok} ttft={ttft:.3f}s "
                f"decode={decode_elapsed:.3f}s decode_tps={tps:.1f}"
            )
            print(f"[xgrammar poem content] {content[:200]!r}")
            self.assertGreaterEqual(
                tok,
                MIN_XGRAMMAR_GEN_TOKENS,
                f"xgrammar generation too short ({tok} tok) — structural-tag "
                f"wrap likely dropped; expected reasoning + JSON ≥"
                f"{MIN_XGRAMMAR_GEN_TOKENS} tok",
            )
            self.assertGreaterEqual(
                tps,
                MIN_XGRAMMAR_TPS,
                f"xgrammar JSON stream decode TPS {tps:.1f} < floor "
                f"{MIN_XGRAMMAR_TPS}",
            )

            try:
                obj = json.loads(content)
            except json.JSONDecodeError as e:
                self.fail(
                    f"xgrammar JSON output failed to parse: {e!r}; "
                    f"content={content!r}"
                )
            self.assertIsInstance(obj, dict)
            self.assertIn("title", obj)
            self.assertIn("content", obj)
            self.assertIsInstance(obj["title"], str)
            self.assertIsInstance(obj["content"], str)
            self.assertGreater(len(obj["title"]), 0, "poem title is empty")
            self.assertGreater(len(obj["content"]), 40, "poem content too short")

        self._with_server(("--grammar-backend", "xgrammar"), run)

    # EAGLE3 spec: stream decode TPS floor + acceptance-length floor.
    def test_eagle3_spec(self):
        spec_args = (
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DRAFT,
            "--speculative-num-steps",
            "3",
        )

        def run(port):
            _chat_nonstream(port, PERF_MESSAGES, max_tokens=64)  # warmup

            _, tok, ttft, decode_elapsed, _ = _chat_stream(
                port, PERF_MESSAGES, max_tokens=PERF_MAX_TOKENS
            )
            tps = _stream_decode_tps(tok, decode_elapsed)

            # accept_draft_tokens is the extras-per-verify rate; true "accept
            # length" (including the bonus token) = accept_draft + 1.
            _, _, _, usage_ns = _chat_nonstream(
                port, PERF_MESSAGES, max_tokens=PERF_MAX_TOKENS
            )
            accept_draft = usage_ns.get("accept_draft_tokens")
            accept_len = (accept_draft + 1) if accept_draft is not None else None
            print(
                f"[eagle3] tok={tok} ttft={ttft:.3f}s decode={decode_elapsed:.3f}s "
                f"decode_tps={tps:.1f} accept_draft={accept_draft} "
                f"accept_len={accept_len}"
            )
            self.assertGreaterEqual(tok, PERF_MAX_TOKENS // 2)
            self.assertGreaterEqual(
                tps,
                MIN_SPEC_TPS,
                f"EAGLE3 stream decode TPS {tps:.1f} < floor {MIN_SPEC_TPS}",
            )
            if accept_len is not None:
                self.assertGreaterEqual(
                    accept_len,
                    MIN_ACCEPT_LEN,
                    f"EAGLE3 accept length {accept_len:.2f} < floor {MIN_ACCEPT_LEN}",
                )

        self._with_server(spec_args, run, launch_timeout=SERVER_LAUNCH_TIMEOUT + 300)


if __name__ == "__main__":
    unittest.main()
