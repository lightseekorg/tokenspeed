"""CPU-only coverage for MiniMax-M3's fused QKV+indexer projection.

Validates ``MinimaxM3QKVParallelLinearWithIndexer`` -- a single column-parallel
GEMM emitting ``[q | k | v | index_q | index_k]`` -- against independent
per-projection references: forward correctness at TP=1, sharding across ranks
(including the KV-head replication path and the fully-replicated single
``index_k`` head), and the quantized ``load_qkv_weight`` delegation. Also checks
that the ``modelopt_mixed`` resolver folds the index members into ``qkv_proj``
without disturbing dense / non-M3 layers.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tokenspeed.runtime.layers.parameter import ModelWeightParameter
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.models.minimax_m3 import (
    MinimaxM3QKVParallelLinearWithIndexer,
)

# Small M3-shaped dims: 8 q heads, 2 kv heads, 2 index heads (== kv heads).
# index_head_size deliberately differs from head_size to cover the general case.
_NH, _NKV, _NIDX, _HEAD, _IDX_HEAD, _H = 8, 2, 2, 8, 16, 64


def _build(tp: int, rank: int) -> MinimaxM3QKVParallelLinearWithIndexer:
    return MinimaxM3QKVParallelLinearWithIndexer(
        hidden_size=_H,
        head_size=_HEAD,
        total_num_heads=_NH,
        total_num_kv_heads=_NKV,
        total_num_index_heads=_NIDX,
        index_head_size=_IDX_HEAD,
        bias=False,
        quant_config=None,
        prefix="m",
        tp_rank=rank,
        tp_size=tp,
        tp_group=tuple(range(tp)),
    ).to(torch.float32)


def _ref_weights(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(_NH * _HEAD, _H, generator=g),
        "k": torch.randn(_NKV * _HEAD, _H, generator=g),
        "v": torch.randn(_NKV * _HEAD, _H, generator=g),
        "index_q": torch.randn(_NIDX * _IDX_HEAD, _H, generator=g),
        "index_k": torch.randn(_IDX_HEAD, _H, generator=g),
    }


def test_output_sizes_layout():
    layer = _build(tp=1, rank=0)
    assert layer.output_sizes == [
        _NH * _HEAD,
        _NKV * _HEAD,
        _NKV * _HEAD,
        _NIDX * _IDX_HEAD,
        _IDX_HEAD,
    ]
    assert layer.num_index_heads == layer.num_kv_heads


def test_forward_matches_independent_projections_tp1():
    layer = _build(tp=1, rank=0)
    weights = _ref_weights()
    param = dict(layer.named_parameters())["weight"]
    for shard, w in weights.items():
        layer.weight_loader(param, w, shard)

    x = torch.randn(4, _H)
    out, _ = layer(x)
    q, k, v, iq, ik = out.split(
        [_NH * _HEAD, _NKV * _HEAD, _NKV * _HEAD, _NIDX * _IDX_HEAD, _IDX_HEAD],
        dim=-1,
    )
    got = {"q": q, "k": k, "v": v, "index_q": iq, "index_k": ik}
    for name, w in weights.items():
        torch.testing.assert_close(got[name], F.linear(x, w))


@pytest.mark.parametrize("tp", [2, 4])
def test_sharding_across_ranks(tp: int):
    """tp=2: no replication; tp=4: KV replication factor 2; index_k replicated."""
    weights = _ref_weights()
    for rank in range(tp):
        layer = _build(tp=tp, rank=rank)
        param = dict(layer.named_parameters())["weight"]
        for shard, w in weights.items():
            layer.weight_loader(param, w, shard)

        nh, nkv, nidx, rep = (
            layer.num_heads,
            layer.num_kv_heads,
            layer.num_index_heads,
            layer.num_kv_head_replicas,
        )
        kv_rank = rank // rep
        expected = {
            "q": weights["q"].narrow(0, rank * nh * _HEAD, nh * _HEAD),
            "k": weights["k"].narrow(0, kv_rank * nkv * _HEAD, nkv * _HEAD),
            "v": weights["v"].narrow(0, kv_rank * nkv * _HEAD, nkv * _HEAD),
            "index_q": weights["index_q"].narrow(
                0, kv_rank * nidx * _IDX_HEAD, nidx * _IDX_HEAD
            ),
            "index_k": weights["index_k"],  # replicated to every rank
        }
        sizes = [
            ("q", nh * _HEAD),
            ("k", nkv * _HEAD),
            ("v", nkv * _HEAD),
            ("index_q", nidx * _IDX_HEAD),
            ("index_k", _IDX_HEAD),
        ]
        offset = 0
        for name, size in sizes:
            got = param.data.narrow(0, offset, size)
            torch.testing.assert_close(got, expected[name])
            offset += size


def test_quantized_load_qkv_weight_delegation_replicates_index_k():
    """The MXFP8 path routes through param.load_qkv_weight; index_k passes
    num_heads=tp_size so every rank picks shard 0 (full replication)."""
    tp = 4
    weights = _ref_weights()
    nh, nkv, nidx, rep = _NH // tp, 1, 1, tp // _NKV
    sizes = [
        ("q", nh * _HEAD),
        ("k", nkv * _HEAD),
        ("v", nkv * _HEAD),
        ("index_q", nidx * _IDX_HEAD),
        ("index_k", _IDX_HEAD),
    ]
    total = sum(sz for _, sz in sizes)
    for rank in range(tp):
        param = ModelWeightParameter(
            data=torch.zeros(total, _H), input_dim=1, output_dim=0, weight_loader=None
        )
        offsets, offset = {}, 0
        for name, sz in sizes:
            offsets[name] = offset
            offset += sz
        for name, sz in sizes:
            num_heads = tp if name == "index_k" else rep
            param.load_qkv_weight(
                loaded_weight=weights[name],
                num_heads=num_heads,
                shard_id=name,
                shard_offset=offsets[name],
                shard_size=sz,
                tp_rank=rank,
                use_presharded_weights=False,
            )
        kv_rank = rank // rep
        expected = {
            "q": weights["q"].narrow(0, rank * nh * _HEAD, nh * _HEAD),
            "k": weights["k"].narrow(0, kv_rank * nkv * _HEAD, nkv * _HEAD),
            "v": weights["v"].narrow(0, kv_rank * nkv * _HEAD, nkv * _HEAD),
            "index_q": weights["index_q"].narrow(
                0, kv_rank * nidx * _IDX_HEAD, nidx * _IDX_HEAD
            ),
            "index_k": weights["index_k"],
        }
        for name, sz in sizes:
            torch.testing.assert_close(
                param.data.narrow(0, offsets[name], sz), expected[name]
            )


def test_invariants_rejected():
    with pytest.raises(ValueError, match="total_num_index_heads == total_num_kv_heads"):
        MinimaxM3QKVParallelLinearWithIndexer(
            _H, _HEAD, _NH, _NKV, _NKV + 1, _IDX_HEAD, tp_rank=0, tp_size=1
        )


def _mixed(layers: dict[str, str]) -> ModelOptMixedConfig:
    return ModelOptMixedConfig(
        quantized_layers=layers, kv_cache_quant_algo=None, group_size=16
    )


def test_resolver_folds_index_members_into_qkv():
    base = "model.layers.5.self_attn"
    sparse = _mixed(
        {
            f"{base}.{m}": "MXFP8"
            for m in ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj")
        }
    )
    assert sparse._resolve_quant_algo(f"{base}.qkv_proj") == "MXFP8"


def test_resolver_skips_absent_index_members_for_dense():
    base = "model.layers.0.self_attn"
    dense = _mixed({f"{base}.{m}": "MXFP8" for m in ("q_proj", "k_proj", "v_proj")})
    assert dense._resolve_quant_algo(f"{base}.qkv_proj") == "MXFP8"


def test_resolver_raises_on_index_member_disagreement():
    base = "model.layers.5.self_attn"
    mixed = _mixed(
        {
            f"{base}.q_proj": "MXFP8",
            f"{base}.k_proj": "MXFP8",
            f"{base}.v_proj": "MXFP8",
            f"{base}.index_q_proj": "NVFP4",
            f"{base}.index_k_proj": "MXFP8",
        }
    )
    with pytest.raises(ValueError, match="Mixed quant_algo within fused layer"):
        mixed._resolve_quant_algo(f"{base}.qkv_proj")
