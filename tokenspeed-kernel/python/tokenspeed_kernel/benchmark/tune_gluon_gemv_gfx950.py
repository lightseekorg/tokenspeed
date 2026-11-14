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

"""Tune the gfx950 BF16 decode GEMV routing tables for one model.

``gemm.mm`` sends a BF16 decode projection ``[M, K] @ [N, K].T`` to one of two
Gluon GEMVs only for measured ``(N, K)`` shapes:

- ``_PREFERRED_BUCKETS`` (``gemv.py``): the dense decode GEMV, per M bucket
  (1, 2-4, 5-16).
- ``_PREFERRED_SHAPES`` (``decode_gemv.py``): the row-CTA GEMV for one row.
- ``_ROWS_CONFIGS`` (``decode_gemv.py``): the row-CTA GEMV for 2 or 3-4 rows,
  with its tile (outputs per wave, waves per program).

This tool derives the projections one tensor-parallel rank runs from a model's
``config.json``, times each routing through ``gemm.mm`` itself under graph
replay with cold weights, and writes the entries that won. A routing enters
the tables only when it leads the routing it replaces by ``min_lead`` in each
of ``sweeps`` sweeps, each sweep the median of ``trials`` runs alternating
the two. Run it on an otherwise idle GPU::

    python -m tokenspeed_kernel.benchmark.tune_gluon_gemv_gfx950 \\
        --model-config config.json --tp 1 --output tuned.json \\
        --min-weight-bytes 2e9 --repeats 20 --replays 7 --seed 0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch

# Both GEMVs address the weight with 32-bit byte offsets.
MAX_WEIGHT_BYTES = 1 << 31

# Row counts each decision is timed at, as the hand-made entries were: the
# row-CTA tiles at 2 and 4 rows, the dense GEMV's 2-4 row bucket by the summed
# time at 2 and 4 rows, and its 5-16 row bucket at 8, 12 and 16 rows.
SINGLE_ROW_MS = (1,)
ROWS_BUCKET_MS = {2: (2,), 4: (4,)}
GEMV_BUCKET4_MS = (2, 4)
GEMV_BUCKET16_MS = (8, 12, 16)

# The few-row tile the tables use unless another measured faster by the gate.
REFERENCE_ROWS_TILE = (4, 4)
_ROWS_OUTPUTS_PER_WAVE = (1, 2, 4)
_ROWS_NUM_WARPS = (1, 2, 4, 8)
# FP32 accumulators per lane: BLOCK_M * outputs per wave * BLOCK_K / (64 * waves).
_ROWS_MAX_ACCUMULATORS = 256
_SUBGROUP_SIZE = 64


@dataclass(frozen=True)
class Projection:
    """A dense projection weight ``[n, k]`` on one rank and the layers using it."""

    n: int
    k: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class Routing:
    """One ``(N, K)`` shape's entries in the three routing tables."""

    gemv_buckets: frozenset[int]
    # Sorted ((row bucket, (outputs per wave, waves per program)), ...).
    rows: tuple[tuple[int, tuple[int, int]], ...]
    single_row: bool

    def with_rows(self, bucket: int, tile: tuple[int, int]) -> Routing:
        rows = dict(self.rows)
        rows[bucket] = tile
        return Routing(self.gemv_buckets, tuple(sorted(rows.items())), self.single_row)

    def to_json(self) -> dict:
        return {
            "gemv_buckets": sorted(self.gemv_buckets),
            "rows": {str(bucket): list(tile) for bucket, tile in self.rows},
            "single_row": self.single_row,
        }


DEFAULT_ROUTING = Routing(frozenset(), (), False)


@dataclass(frozen=True)
class GatePolicy:
    """How much evidence a routing needs to replace another."""

    trials: int
    sweeps: int
    min_lead: float
    screen_trials: int


# The rule the hand-measured table entries used.
GATE = GatePolicy(trials=6, sweeps=3, min_lead=0.015, screen_trials=3)


@dataclass(frozen=True)
class GateResult:
    challenger: Routing
    incumbent: Routing
    ms: tuple[int, ...]
    leads: tuple[float, ...]
    passed: bool

    def to_json(self) -> dict:
        return {
            "challenger": self.challenger.to_json(),
            "incumbent": self.incumbent.to_json(),
            "ms": list(self.ms),
            "leads": [round(lead, 4) for lead in self.leads],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ShapeDecision:
    projection: Projection
    routing: Routing
    gates: tuple[GateResult, ...]


# Time per call in microseconds of one routing, summed over the row counts.
Timer = Callable[[Routing, tuple[int, ...]], float]


def _split(size: int, tp: int, what: str) -> int:
    if size % tp:
        raise ValueError(f"{what} size {size} does not split across TP {tp}")
    return size // tp


def _attention(
    add: Callable[[str, int, int], None],
    text: dict,
    tp: int,
    *,
    qkv_input: int,
    output_gate: bool,
) -> None:
    hidden = text["hidden_size"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text.get("head_dim", hidden // heads)
    local_heads = _split(heads, tp, "attention heads")
    # QKVParallelLinear replicates KV heads when TP exceeds their count.
    local_kv_heads = max(1, kv_heads // tp)
    q_heads = local_heads * (2 if output_gate else 1)
    add("attn.qkv_proj", (q_heads + 2 * local_kv_heads) * head_dim, qkv_input)
    add("attn.o_proj", hidden, local_heads * head_dim)


def _mlp(add: Callable[[str, int, int], None], text: dict, tp: int) -> None:
    hidden = text["hidden_size"]
    intermediate = _split(text["intermediate_size"], tp, "MLP intermediate")
    add("mlp.gate_up_proj", 2 * intermediate, hidden)
    add("mlp.down_proj", hidden, intermediate)


def derive_projections(config: dict, tp: int) -> tuple[list[Projection], list[str]]:
    """Return the dense BF16 projections one TP rank of ``config`` runs.

    Column-parallel layers split N and row-parallel layers split K. Weights of
    2 GiB or more (an LM head) are left out: neither GEMV addresses them.
    Labels of an EAGLE draft layer start with ``draft.``.

    Returns:
        The projections, merged by shape, and the labels left out.
    """
    if tp < 1:
        raise ValueError(f"TP must be positive, got {tp}")
    text = config.get("text_config", config)
    hidden = text["hidden_size"]
    architectures = tuple(config.get("architectures", ()))
    shapes: dict[tuple[int, int], list[str]] = {}
    draft = any("Eagle" in name for name in architectures)
    prefix = "draft." if draft else ""

    def add(label: str, n: int, k: int) -> None:
        shapes.setdefault((n, k), []).append(f"{prefix}{label}")

    if draft:
        # EAGLE draft layer: attention reads [embedding, hidden] unless the
        # draft projects them back to one hidden width first (attn_norm).
        if text.get("attn_norm", False):
            add("eh_proj", hidden, 2 * hidden)
            qkv_input = hidden
        else:
            qkv_input = 2 * hidden
        _attention(add, text, tp, qkv_input=qkv_input, output_gate=False)
        _mlp(add, text, tp)
        if "num_aux_hidden_states" in text:
            aux_states = text["num_aux_hidden_states"]
        else:
            aux_states = len(text["eagle_config"]["eagle_aux_hidden_state_layer_ids"])
        add("fc", _split(hidden, tp, "fc output"), hidden * aux_states)
    elif text["model_type"] in ("qwen3_5", "qwen3_5_text"):
        if text.get("num_experts"):
            raise ValueError("MoE Qwen3.5 configs are not supported; pass --shape")
        layer_types = set(text["layer_types"])
        if "full_attention" in layer_types:
            _attention(
                add,
                text,
                tp,
                qkv_input=hidden,
                output_gate=text["attn_output_gate"],
            )
        if "linear_attention" in layer_types:
            key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
            value_heads = text["linear_num_value_heads"]
            value_dim = value_heads * text["linear_value_head_dim"]
            # in_proj_qkvzba packs q, k, v, z, b and a.
            fused = 2 * key_dim + 2 * value_dim + 2 * value_heads
            add("gdn.in_proj_qkvzba", _split(fused, tp, "GDN in_proj"), hidden)
            add("gdn.out_proj", hidden, _split(value_dim, tp, "GDN out_proj"))
        _mlp(add, text, tp)
    elif text["model_type"] in ("llama", "mistral", "qwen2", "qwen3"):
        _attention(add, text, tp, qkv_input=hidden, output_gate=False)
        _mlp(add, text, tp)
    else:
        raise ValueError(
            f"unsupported model_type {text['model_type']!r}; pass --shape instead"
        )

    projections = []
    skipped = []
    for (n, k), labels in shapes.items():
        if n * k * 2 >= MAX_WEIGHT_BYTES:
            skipped.extend(labels)
            continue
        projections.append(Projection(n, k, tuple(labels)))
    return projections, skipped


def merge_projections(groups: Sequence[Sequence[Projection]]) -> list[Projection]:
    """Merge projections of several models that share a weight shape."""
    merged: dict[tuple[int, int], list[str]] = {}
    for group in groups:
        for projection in group:
            merged.setdefault((projection.n, projection.k), []).extend(
                projection.labels
            )
    return [Projection(n, k, tuple(labels)) for (n, k), labels in merged.items()]


def rows_tiles(
    n: int, k: int, bucket: int, *, block_k: int, waves: int
) -> list[tuple[int, int]]:
    """Few-row tiles ``(outputs per wave, waves per program)`` that cover
    ``[n, k]`` completely with the single-row geometry ``(block_k, waves)``,
    as ``decode_gemv._rows_config`` requires."""
    if k % block_k or block_k % (_SUBGROUP_SIZE * waves):
        return []
    tiles = []
    for outputs_per_wave in _ROWS_OUTPUTS_PER_WAVE:
        for num_warps in _ROWS_NUM_WARPS:
            if num_warps % waves or n % (num_warps // waves * outputs_per_wave):
                continue
            accumulators = (
                bucket * outputs_per_wave * block_k // (_SUBGROUP_SIZE * waves)
            )
            if accumulators > _ROWS_MAX_ACCUMULATORS:
                continue
            tiles.append((outputs_per_wave, num_warps))
    return tiles


def gate(
    timer: Timer,
    challenger: Routing,
    incumbent: Routing,
    ms: tuple[int, ...],
    policy: GatePolicy,
) -> GateResult:
    """Whether ``challenger`` leads ``incumbent`` by ``policy.min_lead`` in
    every sweep, each sweep comparing medians of alternating runs."""
    leads = []
    for _ in range(policy.sweeps):
        challenger_us: list[float] = []
        incumbent_us: list[float] = []
        for trial in range(policy.trials):
            runs = [(challenger, challenger_us), (incumbent, incumbent_us)]
            if trial % 2:
                runs.reverse()
            for routing, samples in runs:
                samples.append(timer(routing, ms))
        lead = 1.0 - statistics.median(challenger_us) / statistics.median(incumbent_us)
        leads.append(lead)
    passed = all(lead >= policy.min_lead for lead in leads)
    return GateResult(challenger, incumbent, ms, tuple(leads), passed)


def _fastest(
    timer: Timer,
    candidates: Sequence[Routing],
    ms: tuple[int, ...],
    policy: GatePolicy,
) -> Routing:
    """Screen candidates by the median of a few runs; ties keep list order."""
    best = candidates[0]
    best_us = math.inf
    for routing in candidates:
        median_us = statistics.median(
            timer(routing, ms) for _ in range(policy.screen_trials)
        )
        if median_us < best_us:
            best, best_us = routing, median_us
    return best


def tune_shape(
    projection: Projection,
    timer: Timer,
    policy: GatePolicy,
    *,
    rowcta_geometry: tuple[int, int],
) -> ShapeDecision:
    """Choose one shape's table entries.

    Args:
        projection: The weight shape.
        timer: Times a routing of this shape through ``gemm.mm``.
        policy: Gate a routing must pass to replace the one before it.
        rowcta_geometry: ``(BLOCK_K, waves)`` of the single-row GEMV here;
            the few-row tiles keep it so each row stays bitwise the
            single-row result.
    """
    n, k = projection.n, projection.k
    gates: list[GateResult] = []

    # One row: the dense GEMV's bucket 1 or the row-CTA GEMV against mm's own
    # path; the faster one in screening must pass the gate.
    single = _fastest(
        timer,
        [
            Routing(frozenset({1}), (), False),
            Routing(frozenset(), (), True),
        ],
        SINGLE_ROW_MS,
        policy,
    )
    result = gate(timer, single, DEFAULT_ROUTING, SINGLE_ROW_MS, policy)
    gates.append(result)
    gemv_buckets = set(single.gemv_buckets) if result.passed else set()
    single_row = single.single_row and result.passed

    # Two and three-four rows: the row-CTA GEMV's few-row tiles, each bucket
    # on its own. A tile other than the reference needs the gate over it.
    block_k, waves = rowcta_geometry
    rows_routing = DEFAULT_ROUTING
    for bucket, ms in ROWS_BUCKET_MS.items():
        tiles = rows_tiles(n, k, bucket, block_k=block_k, waves=waves)
        if not tiles:
            continue
        candidates = [DEFAULT_ROUTING.with_rows(bucket, tile) for tile in tiles]
        best = _fastest(timer, candidates, ms, policy)
        if REFERENCE_ROWS_TILE in tiles:
            reference = DEFAULT_ROUTING.with_rows(bucket, REFERENCE_ROWS_TILE)
            if best != reference:
                result = gate(timer, best, reference, ms, policy)
                gates.append(result)
                if not result.passed:
                    best = reference
        result = gate(timer, best, DEFAULT_ROUTING, ms, policy)
        gates.append(result)
        if result.passed:
            rows_routing = rows_routing.with_rows(bucket, dict(best.rows)[bucket])

    # The dense GEMV's bucket 4 serves all of 2-4 rows and outranks the
    # row-CTA tiles, so it must beat the routing they leave.
    result = gate(
        timer,
        Routing(frozenset({4}), (), False),
        rows_routing,
        GEMV_BUCKET4_MS,
        policy,
    )
    gates.append(result)
    if result.passed:
        gemv_buckets.add(4)
        rows_routing = DEFAULT_ROUTING

    result = gate(
        timer,
        Routing(frozenset({16}), (), False),
        DEFAULT_ROUTING,
        GEMV_BUCKET16_MS,
        policy,
    )
    gates.append(result)
    if result.passed:
        gemv_buckets.add(16)

    routing = Routing(frozenset(gemv_buckets), rows_routing.rows, single_row)
    return ShapeDecision(projection, routing, tuple(gates))


def format_tables(decisions: Sequence[ShapeDecision]) -> str:
    """Render the decisions as entries of the three source tables."""
    ordered = sorted(decisions, key=lambda d: (d.projection.n, d.projection.k))
    buckets = []
    rows = []
    shapes = []
    for decision in ordered:
        n, k = decision.projection.n, decision.projection.k
        comment = ", ".join(decision.projection.labels)
        routing = decision.routing
        if routing.gemv_buckets:
            values = ", ".join(str(b) for b in sorted(routing.gemv_buckets))
            buckets.append(f"    ({n}, {k}): frozenset({{{values}}}),  # {comment}")
        if routing.rows:
            values = ", ".join(
                f"{bucket}: ({tile[0]}, {tile[1]})" for bucket, tile in routing.rows
            )
            rows.append(f"    ({n}, {k}): {{{values}}},  # {comment}")
        if routing.single_row:
            shapes.append(f"        ({n}, {k}),  # {comment}")
    sections = [
        ("_PREFERRED_BUCKETS (gemv.py)", buckets),
        ("_ROWS_CONFIGS (decode_gemv.py)", rows),
        ("_PREFERRED_SHAPES (decode_gemv.py)", shapes),
    ]
    lines = []
    for title, entries in sections:
        lines.append(f"# {title}")
        lines.extend(entries or ["    # no entries"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU measurement


def current_routing(n: int, k: int) -> Routing:
    """The tables' routing for ``[n, k]`` in this checkout."""
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import decode_gemv, gemv

    rows = decode_gemv._ROWS_CONFIGS.get((n, k), {})
    return Routing(
        frozenset(gemv._PREFERRED_BUCKETS.get((n, k), frozenset())),
        tuple(sorted(rows.items())),
        (n, k) in decode_gemv._PREFERRED_SHAPES,
    )


@contextlib.contextmanager
def routed(n: int, k: int, routing: Routing) -> Iterator[None]:
    """Make ``gemm.mm`` route ``[n, k]`` as ``routing`` inside the block."""
    from tokenspeed_kernel.ops.gemm import _select_mv
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import decode_gemv, gemv

    key = (n, k)
    saved_buckets = gemv._PREFERRED_BUCKETS.pop(key, None)
    saved_rows = decode_gemv._ROWS_CONFIGS.pop(key, None)
    saved_shapes = decode_gemv._PREFERRED_SHAPES
    if routing.gemv_buckets:
        gemv._PREFERRED_BUCKETS[key] = routing.gemv_buckets
    if routing.rows:
        decode_gemv._ROWS_CONFIGS[key] = dict(routing.rows)
    if routing.single_row:
        decode_gemv._PREFERRED_SHAPES = saved_shapes | {key}
    else:
        decode_gemv._PREFERRED_SHAPES = saved_shapes - {key}
    # The mv selection caches its choice per shape.
    _select_mv.cache_clear()
    try:
        yield
    finally:
        gemv._PREFERRED_BUCKETS.pop(key, None)
        decode_gemv._ROWS_CONFIGS.pop(key, None)
        if saved_buckets is not None:
            gemv._PREFERRED_BUCKETS[key] = saved_buckets
        if saved_rows is not None:
            decode_gemv._ROWS_CONFIGS[key] = saved_rows
        decode_gemv._PREFERRED_SHAPES = saved_shapes
        _select_mv.cache_clear()


class ColdWeightTimer:
    """Time ``gemm.mm`` on ``[n, k]`` under graph replay with cold weights.

    One graph cycles enough weight copies (``min_weight_bytes`` in total) that
    every call streams its weight from HBM, as in a decode step; a sample is
    one replay divided by its call count.
    """

    def __init__(
        self,
        n: int,
        k: int,
        *,
        min_weight_bytes: int,
        repeats: int,
        replays: int,
        seed: int,
    ) -> None:
        from tokenspeed_kernel.benchmark.graph import (
            GraphBenchmarkConfig,
            GraphTimer,
        )

        self.n = n
        self.k = k
        self.repeats = repeats
        self.replays = replays
        copies = max(2, math.ceil(min_weight_bytes / (n * k * 2)))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.weights = [
            torch.randn(
                (n, k), dtype=torch.bfloat16, device="cuda", generator=generator
            )
            / math.sqrt(k)
            for _ in range(copies)
        ]
        self.generator = generator
        self.inputs: dict[int, torch.Tensor] = {}
        self.graph_timer = GraphTimer(
            GraphBenchmarkConfig(eager_warmup_iterations=1, replay_warmup_iterations=3)
        )
        self.validated: set[tuple[Routing, int]] = set()
        # Every per-call time taken, by routing and row count, for the report.
        self.samples_us: dict[tuple[Routing, int], list[float]] = {}

    def _input(self, m: int) -> torch.Tensor:
        if m not in self.inputs:
            self.inputs[m] = torch.randn(
                (m, self.k),
                dtype=torch.bfloat16,
                device="cuda",
                generator=self.generator,
            )
        return self.inputs[m]

    def _validate(self, routing: Routing, m: int) -> None:
        from tokenspeed_kernel.ops.gemm import mm

        x = self._input(m)
        reference = x.float() @ self.weights[0].float().T
        torch.testing.assert_close(
            mm(x, self.weights[0]).float(), reference, atol=3e-2, rtol=3e-2
        )
        self.validated.add((routing, m))

    def __call__(self, routing: Routing, ms: tuple[int, ...]) -> float:
        from tokenspeed_kernel.benchmark.graph import PreparedInvocation
        from tokenspeed_kernel.ops.gemm import mm

        total_us = 0.0
        calls = self.repeats * len(self.weights)
        for m in ms:
            x = self._input(m)

            def invoke() -> None:
                for _ in range(self.repeats):
                    for weight in self.weights:
                        mm(x, weight)

            with routed(self.n, self.k, routing):
                if (routing, m) not in self.validated:
                    self._validate(routing, m)
                measurement = self.graph_timer.measure(
                    PreparedInvocation(invoke=invoke),
                    cold_cache=False,
                    measurement_blocks=self.replays,
                )
            call_us = measurement.median_us / calls
            self.samples_us.setdefault((routing, m), []).append(call_us)
            total_us += call_us
        return total_us


def _parse_shape(raw: str) -> Projection:
    n, k = (int(value) for value in raw.split(","))
    if n * k * 2 >= MAX_WEIGHT_BYTES:
        raise argparse.ArgumentTypeError(f"[{n}, {k}] weight is 2 GiB or more")
    return Projection(n, k, (f"shape {n}x{k}",))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-config",
        action="append",
        default=[],
        help="config.json of a model or EAGLE draft (repeatable)",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        type=_parse_shape,
        help="extra weight shape N,K (repeatable)",
    )
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output", required=True, help="JSON result path")
    parser.add_argument("--min-weight-bytes", type=float, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--replays", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import load_builtin_kernels

    if not current_platform().is_cdna4:
        raise SystemExit("this tool tunes gfx950 (CDNA4) tables")
    load_builtin_kernels()
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        gluon_rowcta_bf16_gemv_geometry_gfx950,
    )

    groups: list[list[Projection]] = [list(args.shape)]
    skipped: list[str] = []
    for path in args.model_config:
        with open(path) as file:
            config = json.load(file)
        projections, left_out = derive_projections(config, args.tp)
        groups.append(projections)
        skipped.extend(left_out)

    decisions = []
    records = []
    for projection in merge_projections(groups):
        n, k = projection.n, projection.k
        print(f"[{n}, {k}] {', '.join(projection.labels)}", flush=True)
        timer = ColdWeightTimer(
            n,
            k,
            min_weight_bytes=int(args.min_weight_bytes),
            repeats=args.repeats,
            replays=args.replays,
            seed=args.seed,
        )
        decision = tune_shape(
            projection,
            timer,
            GATE,
            rowcta_geometry=gluon_rowcta_bf16_gemv_geometry_gfx950(n, k),
        )
        current = current_routing(n, k)
        decisions.append(decision)
        records.append(
            {
                "n": n,
                "k": k,
                "labels": list(projection.labels),
                "routing": decision.routing.to_json(),
                "current": current.to_json(),
                "matches_current": decision.routing == current,
                "gates": [result.to_json() for result in decision.gates],
                "median_us": [
                    {
                        "routing": routing.to_json(),
                        "m": m,
                        "us": round(statistics.median(samples), 3),
                        "runs": len(samples),
                    }
                    for (routing, m), samples in timer.samples_us.items()
                ],
            }
        )
        print(
            f"  tuned {decision.routing.to_json()} "
            f"current {current.to_json()} "
            f"{'match' if decision.routing == current else 'DIFFER'}",
            flush=True,
        )
        del timer
        torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(0)
    result = {
        "device": properties.name,
        "gcn_arch": properties.gcnArchName,
        "cu_count": properties.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "tp": args.tp,
        "models": args.model_config,
        "skipped": skipped,
        "policy": {
            "trials": GATE.trials,
            "sweeps": GATE.sweeps,
            "min_lead": GATE.min_lead,
            "screen_trials": GATE.screen_trials,
        },
        "shapes": records,
        "tables": format_tables(decisions),
    }
    with open(args.output, "w") as file:
        json.dump(result, file, indent=2)
    print(result["tables"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
