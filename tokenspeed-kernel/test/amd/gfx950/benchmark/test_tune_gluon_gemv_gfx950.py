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

import pytest
from tokenspeed_kernel.benchmark.tune_gluon_gemv_gfx950 import (
    DEFAULT_ROUTING,
    GATE,
    GatePolicy,
    Projection,
    Routing,
    derive_projections,
    format_tables,
    gate,
    merge_projections,
    rows_tiles,
    tune_shape,
)

QWEN38_27B = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "vocab_size": 248320,
    },
}

QWEN38_27B_EAGLE4 = {
    "architectures": ["LlamaForCausalLMEagle4"],
    "model_type": "llama",
    "hidden_size": 5120,
    "intermediate_size": 16384,
    "num_attention_heads": 40,
    "num_key_value_heads": 40,
    "head_dim": 128,
    "num_aux_hidden_states": 3,
    "vocab_size": 248320,
}


def _shapes(projections: list[Projection]) -> dict[tuple[int, int], set[str]]:
    return {(p.n, p.k): set(p.labels) for p in projections}


def test_qwen38_27b_projections_at_tp1() -> None:
    projections, skipped = derive_projections(QWEN38_27B, 1)

    assert _shapes(projections) == {
        (14336, 5120): {"attn.qkv_proj"},
        (5120, 6144): {"attn.o_proj", "gdn.out_proj"},
        (16480, 5120): {"gdn.in_proj_qkvzba"},
        (34816, 5120): {"mlp.gate_up_proj"},
        (5120, 17408): {"mlp.down_proj"},
    }
    assert skipped == []


def test_eagle_draft_projections_at_tp1() -> None:
    projections, _ = derive_projections(QWEN38_27B_EAGLE4, 1)

    assert _shapes(projections) == {
        (15360, 10240): {"draft.attn.qkv_proj"},
        (5120, 5120): {"draft.attn.o_proj"},
        (32768, 5120): {"draft.mlp.gate_up_proj"},
        (5120, 16384): {"draft.mlp.down_proj"},
        (5120, 15360): {"draft.fc"},
    }


def test_tensor_parallel_splits_columns_and_rows() -> None:
    projections, _ = derive_projections(QWEN38_27B, 2)

    # 12 gated query heads, 2 KV heads of 256 per rank.
    assert _shapes(projections) == {
        (7168, 5120): {"attn.qkv_proj"},
        (5120, 3072): {"attn.o_proj", "gdn.out_proj"},
        (8240, 5120): {"gdn.in_proj_qkvzba"},
        (17408, 5120): {"mlp.gate_up_proj"},
        (5120, 8704): {"mlp.down_proj"},
    }


def test_weights_of_two_gib_are_left_out() -> None:
    config = {
        "model_type": "llama",
        "hidden_size": 16384,
        "intermediate_size": 49152,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
    }

    projections, skipped = derive_projections(config, 1)

    assert skipped == ["mlp.gate_up_proj"]
    assert all(p.n * p.k * 2 < 1 << 31 for p in projections)
    assert (16384, 49152) in _shapes(projections)


def test_merging_models_joins_labels_of_one_shape() -> None:
    target, _ = derive_projections(QWEN38_27B, 1)
    draft, _ = derive_projections(QWEN38_27B_EAGLE4, 1)
    shared = Projection(5120, 6144, ("extra",))

    merged = _shapes(merge_projections([target, draft, [shared]]))

    assert len(merged) == 10
    assert merged[(5120, 6144)] == {"attn.o_proj", "gdn.out_proj", "extra"}


def test_rows_tiles_tile_the_weight_with_the_single_row_geometry() -> None:
    tiles = rows_tiles(5120, 6144, 4, block_k=512, waves=4)

    assert (4, 4) in tiles
    for outputs_per_wave, num_warps in tiles:
        assert num_warps % 4 == 0
        assert 5120 % (num_warps // 4 * outputs_per_wave) == 0
    # A K that the chunk does not divide has no few-row tile.
    assert rows_tiles(5120, 6000, 4, block_k=512, waves=4) == []


class _Device:
    """Deterministic per-call times (us) for each kernel ``gemm.mm`` reaches.

    The dispatch order mirrors ``gemm.mv``: the dense GEMV on its buckets,
    then the few-row tiles, then the single-row GEMV, else mm's own path.
    """

    def __init__(
        self,
        default: dict[int, float],
        gemv: dict[int, float],
        rowcta: float,
        rows: dict[int, float],
        tile_scale: dict[tuple[int, int], float],
    ) -> None:
        self.default = default
        self.gemv = gemv
        self.rowcta = rowcta
        self.rows = rows
        self.tile_scale = tile_scale

    def __call__(self, routing: Routing, ms: tuple[int, ...]) -> float:
        return sum(self._time(routing, m) for m in ms)

    def _time(self, routing: Routing, m: int) -> float:
        bucket = 1 if m == 1 else 4 if m <= 4 else 16
        if bucket in routing.gemv_buckets:
            return self.gemv[m]
        rows = dict(routing.rows)
        rows_bucket = 1 << (m - 1).bit_length()
        if m >= 2 and rows_bucket in rows:
            return self.rows[m] * self.tile_scale.get(rows[rows_bucket], 1.01)
        if m == 1 and routing.single_row:
            return self.rowcta
        return self.default[m]


# Rounded medians the hand-made Qwen3.8-27B entries came from.
_DOWN = _Device(
    default={1: 34.0, 2: 34.2, 4: 36.0, 8: 40.0, 12: 42.0, 16: 44.0},
    gemv={1: 29.4, 2: 32.0, 4: 33.4, 8: 37.0, 12: 40.0, 16: 44.3},
    rowcta=34.7,
    rows={2: 30.5, 4: 35.7},
    tile_scale={(4, 4): 1.0},
)
_O_PROJ = _Device(
    default={1: 20.0, 2: 21.0, 4: 22.0, 8: 24.0, 12: 26.0, 16: 28.0},
    gemv={1: 14.8, 2: 17.5, 4: 18.2, 8: 19.4, 12: 22.6, 16: 25.3},
    rowcta=12.9,
    rows={2: 14.6, 4: 18.3},
    tile_scale={(4, 4): 1.0},
)
_GEOMETRY = (512, 4)


def test_down_projection_takes_the_dense_gemv_through_16_rows() -> None:
    decision = tune_shape(
        Projection(5120, 17408, ("mlp.down_proj",)),
        _DOWN,
        GATE,
        rowcta_geometry=_GEOMETRY,
    )

    # The 2-row tile wins alone, but the dense GEMV's 2-4 row bucket beats the
    # tile plus mm's path at 4 rows, and 5-16 rows win on the summed time
    # although 16 rows alone are slower.
    assert decision.routing == Routing(frozenset({1, 4, 16}), (), False)


def test_output_projection_takes_the_row_cta_gemv_through_4_rows() -> None:
    decision = tune_shape(
        Projection(5120, 6144, ("attn.o_proj",)),
        _O_PROJ,
        GATE,
        rowcta_geometry=_GEOMETRY,
    )

    assert decision.routing == Routing(
        frozenset({16}), ((2, (4, 4)), (4, (4, 4))), True
    )
    assert format_tables([decision]).splitlines() == [
        "# _PREFERRED_BUCKETS (gemv.py)",
        "    (5120, 6144): frozenset({16}),  # attn.o_proj",
        "# _ROWS_CONFIGS (decode_gemv.py)",
        "    (5120, 6144): {2: (4, 4), 4: (4, 4)},  # attn.o_proj",
        "# _PREFERRED_SHAPES (decode_gemv.py)",
        "        (5120, 6144),  # attn.o_proj",
    ]


def test_a_tile_replaces_the_reference_only_past_the_gate() -> None:
    def device(scale: float) -> _Device:
        return _Device(
            default=_O_PROJ.default,
            gemv=_O_PROJ.gemv,
            rowcta=_O_PROJ.rowcta,
            rows=_O_PROJ.rows,
            tile_scale={(4, 4): 1.0, (2, 8): scale},
        )

    close = tune_shape(
        Projection(5120, 6144, ()), device(0.99), GATE, rowcta_geometry=_GEOMETRY
    )
    faster = tune_shape(
        Projection(5120, 6144, ()), device(0.95), GATE, rowcta_geometry=_GEOMETRY
    )

    assert close.routing.rows == ((2, (4, 4)), (4, (4, 4)))
    assert faster.routing.rows == ((2, (2, 8)), (4, (2, 8)))


def test_gate_needs_the_lead_in_every_sweep() -> None:
    policy = GatePolicy(trials=6, sweeps=3, min_lead=0.015, screen_trials=3)
    challenger = Routing(frozenset({1}), (), False)
    runs = iter(range(1000))

    def noisy(routing: Routing, ms: tuple[int, ...]) -> float:
        # The challenger is 3% faster except in the second sweep.
        index = next(runs)
        sweep = index // (2 * policy.trials)
        if routing == DEFAULT_ROUTING:
            return 100.0
        return 99.5 if sweep == 1 else 97.0

    result = gate(noisy, challenger, DEFAULT_ROUTING, (1,), policy)

    assert not result.passed
    assert [round(lead, 3) for lead in result.leads] == [0.03, 0.005, 0.03]


def test_no_entry_without_a_lead() -> None:
    even = _Device(
        default={m: 10.0 for m in (1, 2, 4, 8, 12, 16)},
        gemv={m: 9.95 for m in (1, 2, 4, 8, 12, 16)},
        rowcta=9.95,
        rows={2: 9.95, 4: 9.95},
        tile_scale={(4, 4): 1.0},
    )

    decision = tune_shape(
        Projection(4096, 4096, ()), even, GATE, rowcta_geometry=_GEOMETRY
    )

    assert decision.routing == DEFAULT_ROUTING
    assert all(not result.passed for result in decision.gates)


@pytest.mark.parametrize("tp", [0, -1])
def test_tp_must_be_positive(tp: int) -> None:
    with pytest.raises(ValueError, match="TP must be positive"):
        derive_projections(QWEN38_27B, tp)
