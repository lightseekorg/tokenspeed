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

"""``--prefill-context-parallel-size``: the layouts the first landing refuses."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.utils.server_args import prepare_server_args, validate_qcp

BASE = [
    "--model",
    "x",
    "--attn-tp-size",
    "2",
    "--prefill-context-parallel-size",
    "2",
    "--disaggregation-mode",
    "prefill",
    "--disable-prefill-graph",
    "--attention-backend",
    "dsa",
]


def test_the_prefill_role_accepts_a_full_tp_query_shard():
    args = prepare_server_args(BASE)
    assert args.mapping.attn.qcp_size == 2
    assert args.mapping.attn.has_qcp
    args.mapping.rank = 1
    assert args.mapping.attn.qcp_group == args.mapping.attn.tp_group == (0, 1)
    assert args.mapping.attn.qcp_rank == 1
    # DCP equal to the shard group is the one sharded-page layout allowed
    # (and it inherits DCP's Host KVStore refusal).
    args = prepare_server_args(
        BASE + ["--decode-context-parallel-size", "2", "--disable-kvstore"]
    )
    assert args.mapping.attn.dcp_size == 2


def test_off_by_default_everywhere():
    args = prepare_server_args(["--model", "x", "--attn-tp-size", "2"])
    assert args.prefill_context_parallel_size == 1
    assert not args.mapping.attn.has_qcp


@pytest.mark.parametrize(
    "argv,match",
    [
        (
            # The mapping refuses the partial shard first (a query shard spans
            # the whole group); validate_qcp repeats the rule for its callers.
            ["--attn-tp-size", "4", "--prefill-context-parallel-size", "2"],
            "must be 1 or the attention TP size",
        ),
        (["--disaggregation-mode", "null"], "requires --disaggregation-mode prefill"),
        (["--disaggregation-mode", "decode"], "requires --disaggregation-mode prefill"),
        (["--enable-mixed-batch"], "--enable-mixed-batch"),
        (["--attention-backend", "flashmla"], "DSA-family attention backend"),
        (["--kv-cache-dtype", "fp8_e4m3"], "bf16 KV cache"),
        (["--kv-cache-dtype", "mxfp8"], "bf16 KV cache"),
        (["--kv-cache-quant-method", "per_token_head"], "bf16 KV cache"),
    ],
)
def test_refusals(argv, match):
    # Later flags override earlier ones, so BASE + argv applies the override.
    with pytest.raises(ValueError, match=match):
        prepare_server_args(BASE + argv)


def test_refuses_dcp_narrower_than_the_shard_group():
    argv = [flag if flag != "2" else "4" for flag in BASE]
    with pytest.raises(ValueError, match="--decode-context-parallel-size must be 1"):
        prepare_server_args(argv + ["--decode-context-parallel-size", "2"])


def test_refuses_the_prefill_graph():
    argv = [flag for flag in BASE if flag != "--disable-prefill-graph"]
    with pytest.raises(ValueError, match="--disable-prefill-graph"):
        prepare_server_args(argv)


def test_refuses_attention_dp():
    with pytest.raises(ValueError, match="attention DP 1"):
        prepare_server_args(BASE + ["--data-parallel-size", "2"])


def test_dense_and_moe_groups_are_one_or_the_attention_width():
    """The attention weights are head-replicated, so the drafter's replicated
    decode rows are never scattered: a dense or MoE group narrower than the
    attention TP would gather rows nobody scattered."""
    wide = [flag if flag != "2" else "4" for flag in BASE]
    for argv in (["--dense-tp-size", "1"], ["--dense-tp-size", "4"]):
        args = prepare_server_args(wide + argv)
        assert args.mapping.dense.tp_size == int(argv[1])
    args = prepare_server_args(wide + ["--dense-tp-size", "1", "--ep-size", "4"])
    assert args.mapping.moe.tp_ep_size == 4
    with pytest.raises(ValueError, match="1 or the attention TP width"):
        prepare_server_args(wide + ["--dense-tp-size", "2"])


def test_the_default_attention_weights_are_head_replicated():
    args = prepare_server_args(BASE)
    args.mapping.rank = 1
    assert args.mapping.attn.head_tp_size == 1
    assert not args.mapping.attn.has_head_tp
    assert args.mapping.attn.head_tp_group == (1,)


def test_head_tp_over_the_query_shards():
    """``--attn-head-tp-size`` equal to the shard group is the prefill-role
    head-TP layout: the head group is the shard group, ``--tp-batch-invariant
    attn`` selects the column-parallel o_proj, and none of the decode-only
    gates (role, decode-shaped autotune, generation budget) apply -- they key
    on ``head_tp_serves_decode_only``, False here."""
    args = prepare_server_args(BASE + ["--attn-head-tp-size", "2"])
    args.mapping.rank = 1
    attn = args.mapping.attn
    assert attn.has_head_tp and attn.has_qcp
    assert not attn.head_tp_serves_decode_only
    assert attn.head_tp_group == attn.qcp_group == (0, 1)
    assert args.disaggregation_mode == "prefill"
    assert args.tp_batch_invariant == "none"

    bi = prepare_server_args(
        BASE + ["--attn-head-tp-size", "2", "--tp-batch-invariant", "attn"]
    )
    assert bi.tp_batch_invariant == "attn"
    assert bi.mapping.attn.has_head_tp


@pytest.mark.parametrize(
    "argv,match",
    [
        (["--attn-head-tp-size", "4"], "must equal qcp_size"),
        (
            ["--attn-head-tp-size", "1", "--tp-batch-invariant", "attn"],
            "attn-head-tp-size",
        ),
    ],
)
def test_head_tp_on_the_prefill_role_is_the_shard_group(argv, match):
    with pytest.raises(ValueError, match=match):
        prepare_server_args(BASE + argv)


@pytest.mark.parametrize("dense_tp", ["1", "2"])
def test_the_batch_invariant_dense_tail_has_no_layout_under_query_sharding(dense_tp):
    """``--tp-batch-invariant attn+dense`` replaces the token reduce-scatter
    of a dense group wider than attention TP; under query sharding the dense
    group is 1 or the attention TP width, so the selection is refused at
    argument resolution (CommManager would refuse it at construction)."""
    argv = BASE + [
        "--attn-head-tp-size",
        "2",
        "--dense-tp-size",
        dense_tp,
        "--tp-batch-invariant",
        "attn+dense",
    ]
    with pytest.raises(ValueError, match="only --tp-batch-invariant attn applies"):
        prepare_server_args(argv)
    args = prepare_server_args(argv[:-1] + ["attn"])
    assert args.tp_batch_invariant == "attn"
    assert args.mapping.dense.tp_size == int(dense_tp)


def test_head_tp_on_the_prefill_role_still_needs_the_shard():
    """Without a query shard the attention TP ranks hold the same rows, so
    head TP on the prefill role is the decode-only layout, refused as before."""
    with pytest.raises(ValueError, match="attention TP 1"):
        prepare_server_args(
            [
                "--model",
                "x",
                "--attn-tp-size",
                "2",
                "--disaggregation-mode",
                "prefill",
                "--attention-backend",
                "dsa",
                "--attn-head-tp-size",
                "4",
            ]
        )
    with pytest.raises(ValueError, match="disaggregation-mode decode"):
        prepare_server_args(
            [
                "--model",
                "x",
                "--attn-tp-size",
                "1",
                "--data-parallel-size",
                "2",
                "--disaggregation-mode",
                "prefill",
                "--attn-head-tp-size",
                "2",
            ]
        )


def test_validate_qcp_rejects_a_shard_below_the_tp_width():
    with pytest.raises(ValueError, match="must equal the attention TP size"):
        validate_qcp(
            qcp_size=2,
            attn_tp_size=4,
            attn_dp_size=1,
            dense_tp_size=1,
            moe_tp_ep_size=4,
            dcp_size=1,
            disaggregation_mode="prefill",
            disable_prefill_graph=True,
            enable_mixed_batch=False,
            attention_backend="dsa",
            kv_cache_dtype="auto",
            kv_cache_quant_method="none",
        )
    # An unset backend resolves to the architecture's default; the attention
    # config pins it to GPU DSA when the model is known.
    validate_qcp(
        qcp_size=4,
        attn_tp_size=4,
        attn_dp_size=1,
        dense_tp_size=4,
        moe_tp_ep_size=4,
        dcp_size=4,
        disaggregation_mode="prefill",
        disable_prefill_graph=True,
        enable_mixed_batch=False,
        attention_backend=None,
        kv_cache_dtype="bfloat16",
        kv_cache_quant_method="none",
    )


def _dsa_attn_config(**overrides):
    import torch

    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig

    spec = DSAConfig(
        num_attention_heads=4,
        num_kv_heads=1,
        head_dim=576,
        attn_tp_size=2,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=576,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=64,
        index_k_format="fp8_scaled",
    )
    kwargs = dict(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=64,
        kernel_page_size=64,
        context_len=4096,
        max_bs=4,
        qcp_size=2,
        qcp_rank=0,
        qcp_group=(0, 1),
        components=(spec,),
    )
    kwargs.update(overrides)
    return AttnConfig(**kwargs)


def test_the_attention_config_pins_query_sharding_to_a_bf16_native_cache():
    """The sharded KV write stores gathered rows with ``latent_store``, which
    writes native rows only: an FP8 / MXFP8 / per-token-head cache is refused
    where the config is built, not at the first write."""
    import torch

    assert _dsa_attn_config().qcp_size == 2
    for overrides in (
        {"kv_cache_dtype": torch.float8_e4m3fn},
        {"kv_cache_dtype": torch.float8_e4m3fn, "kv_cache_mxfp8": True},
        {"kv_cache_quant_method": "per_token_head"},
    ):
        with pytest.raises(ValueError, match="bf16 KV cache"):
            _dsa_attn_config(**overrides)
    # Off, the same caches are allowed.
    _dsa_attn_config(
        kv_cache_dtype=torch.float8_e4m3fn, qcp_size=1, qcp_rank=0, qcp_group=(0,)
    )
