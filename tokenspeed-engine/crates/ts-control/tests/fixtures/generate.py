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

"""How the Python engine normalizes sampling parameters off the msgpack wire,
and how it publishes KV cache events.

``sampling.hex`` holds one case per line: the raw ``SamplingParams`` array a
frontend sends, the request id, the array after the engine's ingress
(decode, ``resolve_seed``, ``normalize``), and ``ok`` or ``invalid`` for
``verify``. ``kv_events.hex`` is one ``KVEventBatch`` payload as the ZMQ KV
event publisher sends it. ``eplb.json`` holds expert loads with the placement
the EPLB solver derives for them and every rank's slot moves into it.
``ts-control``'s ``sampling``, ``kv_events`` and ``eplb`` modules must
reproduce them;
``test/runtime/test_ts_control_fixtures.py`` checks the file still matches
Python. Rewrite it after an intentional change:

    python tokenspeed-engine/crates/ts-control/tests/fixtures/generate.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import msgspec
import torch

from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
from tokenspeed.runtime.moe.expert_rebalance import plan_slot_moves
from tokenspeed.runtime.moe.placement_maps import compute_placement_maps
from tokenspeed.runtime.pd.kv_events import BlockRemoved, BlockStored, KVEventBatch
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

FIXTURE_DIR = Path(__file__).resolve().parent
VOCAB_SIZE = 1000

_FIELDS = [field.name for field in msgspec.structs.fields(SamplingParams)]
_DEFAULTS = {
    field.name: field.default for field in msgspec.structs.fields(SamplingParams)
}


def _raw(**overrides) -> bytes:
    """The array a frontend puts on the wire: every field, unnormalized."""
    unknown = set(overrides) - set(_FIELDS)
    if unknown:
        raise KeyError(unknown)
    values = [overrides.get(name, _DEFAULTS[name]) for name in _FIELDS]
    return msgspec.msgpack.encode(
        [sorted(v) if isinstance(v, set) else v for v in values]
    )


_CASES: list[tuple[str, bytes]] = [
    ("r-default", _raw()),
    ("r-greedy", _raw(temperature=0.0, seed=7)),
    ("r-top-k-one", _raw(top_k=1, seed=9)),
    ("r-sampling", _raw(temperature=0.7, top_p=0.9, top_k=20, seed=123)),
    ("r-crc-seed", _raw(temperature=0.5)),
    ("rid-ünicode", _raw(temperature=0.5)),
    ("r-stop-str", _raw(stop="</s>")),
    ("r-stop-list", _raw(stop=["ab", "çdé", ""], stop_token_ids=[9, 3, 3])),
    ("r-already-normalized", _raw(temperature=0.0, top_k=-1, is_normalized=True)),
    ("r-bad-top-p", _raw(top_p=0.0)),
    ("r-bad-top-k", _raw(top_k=0)),
    ("r-wide-top-k", _raw(top_k=128)),
    ("r-ok-top-k", _raw(top_k=127)),
    ("r-negative-temperature", _raw(temperature=-0.5)),
    ("r-min-p", _raw(min_p=1.5)),
    ("r-penalties", _raw(frequency_penalty=2.5)),
    ("r-presence", _raw(presence_penalty=-2.0, repetition_penalty=2.0)),
    ("r-repetition", _raw(repetition_penalty=0.0)),
    ("r-min-over-max", _raw(max_new_tokens=2, min_new_tokens=3)),
    ("r-negative-max", _raw(max_new_tokens=-1)),
    ("r-logit-bias", _raw(logit_bias={"5": 1.0, "999": -1.0})),
    ("r-logit-bias-oov", _raw(logit_bias={"1000": 1.0})),
    ("r-logprobs", _raw(logprobs=0)),
    ("r-top-logprobs", _raw(logprobs=2)),
    ("r-two-grammars", _raw(regex="a", ebnf="b")),
    ("r-int-floats", _raw(temperature=1, top_p=1)),
]


def _expected(rid: str, raw: bytes) -> tuple[bytes, str]:
    params = msgspec.msgpack.Decoder(SamplingParams).decode(raw)
    params.resolve_seed(rid)
    params.normalize(tokenizer=None)
    try:
        params.verify(VOCAB_SIZE)
        verdict = "ok"
    except ValueError:
        verdict = "invalid"
    if isinstance(params.stop_token_ids, set):
        params.stop_token_ids = sorted(params.stop_token_ids)
    return msgspec.msgpack.encode(params), verdict


#: A KV event batch: a stored chain under a parent, a root block whose hash
#: needs 64 bits, and a removal.
_KV_EVENT_BATCH = KVEventBatch(
    ts=1_700_000_000.25,
    events=[
        BlockStored(
            block_hashes=[11, 2**63 + 5],
            parent_block_hash=7,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_size=4,
        ),
        BlockStored(
            block_hashes=[2**64 - 1],
            parent_block_hash=None,
            token_ids=[9, 10, 11, 12],
            block_size=4,
        ),
        BlockRemoved(block_hashes=[11, 3]),
    ],
    attn_dp_rank=1,
)


#: EPLB geometries: (layers, logical, physical, ep_size, groups, nodes,
#: algorithm, per-rank nodes).
_EPLB_CASES = [
    (3, 16, 24, 4, 4, 2, "deepseek_hierarchical", [0, 0, 1, 1]),
    (2, 16, 24, 4, None, 1, "deepseek", [0, 0, 0, 0]),
    (2, 8, 16, 8, 2, 2, "deepseek_hierarchical", [0, 0, 0, 0, 1, 1, 1, 1]),
]


def _eplb_cases() -> list[dict]:
    rng = random.Random(11)
    cases = []
    for (
        layers,
        logical,
        physical,
        ep_size,
        groups,
        nodes,
        algorithm,
        rank_nodes,
    ) in _EPLB_CASES:
        # Skewed integer loads with ties, as route counts are.
        load = [
            [
                rng.choice([0, 1, 5, 5, 40, 300, rng.randrange(1000)])
                for _ in range(logical)
            ]
            for _ in range(layers)
        ]
        new_map, _ = compute_placement_maps(
            torch.tensor(load, dtype=torch.int64),
            num_physical_experts=physical,
            ep_size=ep_size,
            num_groups=groups,
            num_nodes=nodes,
            algorithm=EplbAlgorithm[algorithm],
        )
        old_map = [[p % logical for p in range(physical)] for _ in range(layers)]
        num_local = physical // ep_size
        moves = [
            [
                {
                    field: [list(pair) for pair in getattr(m, field)]
                    for field in ("recv", "send", "local_copy", "free_rider")
                }
                for m in (
                    plan_slot_moves(
                        old_map[layer],
                        new_map[layer].tolist(),
                        ep_rank=rank,
                        ep_rank_nodes=rank_nodes,
                        num_local=num_local,
                    )
                    for layer in range(layers)
                )
            ]
            for rank in range(ep_size)
        ]
        cases.append(
            {
                "load": load,
                "num_physical": physical,
                "ep_size": ep_size,
                "num_groups": groups,
                "num_nodes": nodes,
                "algorithm": algorithm,
                "ep_rank_nodes": rank_nodes,
                "old_map": old_map,
                "new_map": new_map.tolist(),
                "moves": moves,
            }
        )
    return cases


def build_fixtures() -> dict[str, str]:
    """Each fixture file's expected content, keyed by file name."""
    lines = []
    for rid, raw in _CASES:
        expected, verdict = _expected(rid, raw)
        lines.append(f"{raw.hex()} {rid.encode().hex()} {expected.hex()} {verdict}\n")
    return {
        "sampling.hex": "".join(lines),
        "kv_events.hex": msgspec.msgpack.encode(_KV_EVENT_BATCH).hex() + "\n",
        "eplb.json": json.dumps(_eplb_cases()) + "\n",
    }


def main() -> None:
    for name, content in build_fixtures().items():
        (FIXTURE_DIR / name).write_text(content)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
