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

"""A rank's scheduler config, from real ServerArgs and a device build.

CPU-only. ``scheduler_config_from_args`` is the one translation of the
server arguments and ``DeviceSpecs`` into ``make_config``'s arguments; these
tests run it with the arguments as an engine resolves them, so a knob that is
unset on some engines (the L3 prefetch threshold without a store) is caught
here rather than at the binding on startup.
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

ts = pytest.importorskip("tokenspeed_scheduler")

from test.runtime.cache_pool_test_utils import (  # noqa: E402
    make_layer_group_ids,
    specs_for_layers,
)

from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig  # noqa: E402
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.engine.rank_setup import (  # noqa: E402
    rank_process_title,
    scheduler_config_from_args,
)
from tokenspeed.runtime.engine.scheduler_utils import make_config  # noqa: E402
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (  # noqa: E402
    cache_group_config,
)
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    ServerArgs,
    prepare_server_args,
)


def _specs(*, num_host_pages: int, num_snapshot_pages: int, max_retracted: int):
    return SimpleNamespace(
        cache_geometry=SimpleNamespace(num_device_pages=64, prefix_granularity=16),
        cache_groups=[],
        host_group_packing=(),
        num_host_pages=num_host_pages,
        num_snapshot_pages=num_snapshot_pages,
        max_retracted_requests=max_retracted,
    )


def _config(args: ServerArgs, specs):
    storage_config = None
    if args.kvstore_storage_backend is not None:
        storage_config = StorageRuntimeConfig.from_extra_config(
            None, host_blocks=specs.num_host_pages - 1, prefix_granularity=16
        ).scheduler
    return scheduler_config_from_args(
        args,
        specs,
        storage_config=storage_config,
        max_scheduled_tokens=args.chunked_prefill_size,
        max_batch_size=args.retraction_snapshot_max_requests or 8,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        enable_kv_cache_events=False,
        prefix_replay_tokens=0,
    )


def test_an_engine_without_l3_passes_no_prefetch_threshold():
    # The knob is unset (None) without a store; the binding takes 0.
    args = ServerArgs(model="x", max_num_seqs=8)
    cfg = _config(
        args, _specs(num_host_pages=17, num_snapshot_pages=5, max_retracted=8)
    )
    assert cfg.enable_l3_storage is False
    assert cfg.storage_config is None
    assert cfg.disable_l2_cache is False and cfg.num_host_pages == 17
    assert cfg.host_write_policy == ts.SchedulerConfig.HostWritePolicy.WriteThrough
    assert (cfg.num_snapshot_pages, cfg.max_retracted_requests) == (5, 8)


def test_make_config_requires_and_maps_the_host_write_policy():
    fields = dict(
        num_device_pages=32,
        max_scheduled_tokens=64,
        max_batch_size=8,
        prefix_granularity=2,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=1,
        host_group_packing=(),
        max_retracted_requests=0,
        storage_config=None,
    )
    # Keyword-only and required, with or without a Host tier.
    with pytest.raises(TypeError, match="host_write_policy"):
        make_config(**fields)
    policy = ts.SchedulerConfig.HostWritePolicy
    cfg = make_config(**fields, host_write_policy="write_through")
    assert cfg.host_write_policy == policy.WriteThrough
    # Mapped as given: the scheduler, not make_config, decides what it runs.
    cfg = make_config(**fields, host_write_policy="write_back")
    assert cfg.host_write_policy == policy.WriteBack


@pytest.mark.parametrize("tokens,pages", [(None, 0), (1, 1), (16, 1), (17, 2)])
def test_prefill_kv_budget_is_translated_to_scheduler_pages(tokens, pages):
    flags = [] if tokens is None else ["--chunk-prefill-kv-size", str(tokens)]
    args = prepare_server_args(["--model", "x", "--max-num-seqs", "8", *flags])
    cfg = _config(
        args, _specs(num_host_pages=17, num_snapshot_pages=5, max_retracted=8)
    )
    assert cfg.max_kv_page_num == pages


@pytest.mark.parametrize("tokens", [0, -1])
def test_prefill_kv_budget_must_be_positive_when_supplied(tokens):
    with pytest.raises(ValueError, match="chunk-prefill-kv-size"):
        ServerArgs(model="x", max_num_seqs=8, chunk_prefill_kv_size=tokens)


def test_an_engine_with_l3_passes_its_prefetch_threshold():
    args = ServerArgs(model="x", max_num_seqs=8, kvstore_storage_backend="memory")
    cfg = _config(
        args, _specs(num_host_pages=17, num_snapshot_pages=5, max_retracted=8)
    )
    assert cfg.enable_l3_storage is True
    assert cfg.storage_config is not None


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_roles_configure_their_pools_as_resolved(role):
    # The prefill role resolves to no pool and no forced retraction; the
    # decode role keeps both.
    args = ServerArgs(
        model="x",
        max_num_seqs=8,
        disaggregation_mode=role,
        debug_force_retraction_interval=3,
        disaggregation_transfer_backend="mooncake",
    )
    pool = (1, 0) if role == "prefill" else (5, 8)
    cfg = _config(
        args,
        _specs(num_host_pages=0, num_snapshot_pages=pool[0], max_retracted=pool[1]),
    )
    assert (cfg.num_snapshot_pages, cfg.max_retracted_requests) == pool
    assert cfg.debug_force_retraction_interval == (0 if role == "prefill" else 3)


@pytest.mark.parametrize(
    "flags, enabled",
    [([], False), (["--enable-state-checkpoint-in-forward"], True)],
)
def test_the_in_forward_state_checkpoint_is_off_unless_the_flag_is_given(
    flags, enabled
):
    args = prepare_server_args(["--model", "x", "--max-num-seqs", "8", *flags])
    cfg = _config(
        args, _specs(num_host_pages=17, num_snapshot_pages=5, max_retracted=8)
    )
    assert cfg.state_checkpoint_in_forward is enabled


def test_the_in_forward_state_checkpoint_is_refused_without_prefix_caching():
    # Its checkpoint is only reachable through a prefix hit.
    with pytest.raises(ValueError, match="requires prefix caching"):
        ServerArgs(
            model="x",
            max_num_seqs=8,
            disable_kvstore=True,
            enable_prefix_caching=False,
            enable_state_checkpoint_in_forward=True,
        )


@pytest.mark.parametrize(
    "flags, policy, pool",
    [
        ([], ts.SchedulerConfig.HostWritePolicy.WriteThrough, (5, 8)),
        (
            [
                "--disaggregation-mode",
                "prefill",
                "--kvstore-write-policy",
                "write_back",
            ],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (1, 0),
        ),
        (
            [
                "--disaggregation-mode",
                "prefill",
                "--kvstore-write-policy",
                "write_back",
                "--attn-tp-size",
                "2",
            ],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (1, 0),
        ),
        (
            [
                "--disaggregation-mode",
                "prefill",
                "--kvstore-write-policy",
                "write_back",
                "--attn-tp-size",
                "2",
                "--kv-parallel-size",
                "2",
            ],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (1, 0),
        ),
        (
            [
                "--disaggregation-mode",
                "prefill",
                "--kvstore-write-policy",
                "write_back",
                "--attn-tp-size",
                "4",
                "--kv-parallel-size",
                "2",
            ],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (1, 0),
        ),
        (
            ["--kvstore-write-policy", "write_back"],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (5, 8),
        ),
        (
            ["--kvstore-write-policy", "write_back", "--attn-tp-size", "2"],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (5, 8),
        ),
        (
            [
                "--kvstore-write-policy",
                "write_back",
                "--attn-tp-size",
                "8",
                "--kv-parallel-size",
                "8",
                "--decode-context-parallel-size",
                "8",
            ],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (5, 8),
        ),
        (
            [
                "--kvstore-write-policy",
                "write_back",
                "--attn-tp-size",
                "8",
                "--kv-parallel-size",
                "4",
                "--decode-context-parallel-size",
                "4",
            ],
            ts.SchedulerConfig.HostWritePolicy.WriteBack,
            (5, 8),
        ),
    ],
    ids=[
        "default",
        "prefill_write_back",
        "prefill_write_back_attention_tp",
        "prefill_write_back_kvp",
        "prefill_write_back_tp_wider_than_kvp",
        "fused_write_back",
        "fused_write_back_attention_tp",
        "fused_write_back_dcp",
        "fused_write_back_tp_wider_than_dcp",
    ],
)
def test_the_host_write_policy_follows_the_flag(flags, policy, pool):
    args = prepare_server_args(["--model", "x", "--max-num-seqs", "8", *flags])
    cfg = _config(
        args,
        _specs(num_host_pages=17, num_snapshot_pages=pool[0], max_retracted=pool[1]),
    )
    assert cfg.host_write_policy == policy


@pytest.mark.parametrize(
    "flags, refused",
    [
        (["--disaggregation-mode", "decode"], "--disaggregation-mode decode"),
        (
            [
                "--disaggregation-mode",
                "decode",
                "--attn-tp-size",
                "8",
                "--kv-parallel-size",
                "8",
                "--decode-context-parallel-size",
                "8",
            ],
            "--disaggregation-mode decode",
        ),
        (
            ["--disaggregation-mode", "prefill", "--disable-kvstore"],
            "--disable-kvstore (no Host L2)",
        ),
        # An encode engine runs without the KVStore.
        (["--disaggregation-mode", "encode"], "--disable-kvstore (no Host L2)"),
        (
            ["--disaggregation-mode", "prefill", "--kvstore-storage-backend", "memory"],
            "--kvstore-storage-backend",
        ),
        (
            [
                "--disaggregation-mode",
                "prefill",
                "--attn-tp-size",
                "2",
                "--kv-parallel-size",
                "2",
                "--query-context-parallel-size",
                "2",
                "--attention-backend",
                "dsa",
            ],
            "--query-context-parallel-size > 1",
        ),
        (
            ["--disaggregation-mode", "prefill", "--pipeline-parallel-size", "2"],
            "--pipeline-parallel-size > 1",
        ),
        (["--pipeline-parallel-size", "2"], "--pipeline-parallel-size > 1"),
    ],
    ids=["decode", "decode_dcp", "no_host", "encode", "l3", "qcp", "pp", "fused_pp"],
)
def test_write_back_is_refused_without_its_prerequisites(flags, refused, monkeypatch):
    # The scheduler writes back on every role but decode; the runtime with
    # Host L2 and no L3, on one pipeline stage, and not yet under a query
    # shard. Attention TP and KV parallelism are accepted (above). Each case
    # breaks one prerequisite, which the write_back check names. The debug
    # switch that runs a pipeline on the fused role does not let write_back
    # through.
    monkeypatch.setenv("TS_PP_DEBUG_ALLOW_NON_PREFILL", "1")
    with pytest.raises(
        ValueError, match=re.escape(f"write_back does not support {refused}")
    ):
        prepare_server_args(
            [
                "--model",
                "x",
                "--max-num-seqs",
                "8",
                "--kvstore-write-policy",
                "write_back",
                *flags,
            ]
        )


@pytest.mark.parametrize(
    "flags, refused",
    [
        (
            ["--attn-tp-size", "8", "--kv-parallel-size", "8"],
            "--kv-parallel-size > 1 on --disaggregation-mode null requires "
            "--decode-context-parallel-size 8",
        ),
        (
            [
                "--attn-tp-size",
                "8",
                "--kv-parallel-size",
                "8",
                "--decode-context-parallel-size",
                "8",
                "--enable-mixed-batch",
            ],
            "Aggregated DCP does not support --enable-mixed-batch",
        ),
        (
            [
                "--disaggregation-mode",
                "prefill",
                "--attn-tp-size",
                "8",
                "--kv-parallel-size",
                "8",
                "--decode-context-parallel-size",
                "8",
            ],
            "--decode-context-parallel-size > 1 requires --disaggregation-mode "
            "null or decode",
        ),
    ],
    ids=["fused_kvp_without_dcp", "fused_dcp_mixed_batch", "prefill_dcp"],
)
def test_write_back_leaves_the_context_parallel_rules_as_they_are(flags, refused):
    # A fused engine writes back over sharded pages only as the DCP engine
    # the context-parallel rules make of it, and a prefill engine takes no
    # DCP: write_back opens none of those layouts.
    with pytest.raises(ValueError, match=re.escape(refused)):
        prepare_server_args(
            [
                "--model",
                "x",
                "--max-num-seqs",
                "8",
                "--kvstore-write-policy",
                "write_back",
                *flags,
            ]
        )


def test_write_back_beside_a_sliding_window_group_starts():
    # The server arguments cannot see the model's cache groups; the scheduler
    # built from the device side's groups takes a sliding-window one under
    # either policy.
    layer_types = ("full_attention", "sliding_attention")
    group_specs = specs_for_layers(
        layer_types=layer_types,
        group_ids=make_layer_group_ids(
            layer_num=2, layer_types=layer_types, sliding_window_tokens=32
        ),
        sliding_window_tokens=32,
        prefix_granularity=16,
        pd_disaggregation_enabled=True,
    )
    specs = _specs(num_host_pages=17, num_snapshot_pages=1, max_retracted=0)
    specs.cache_groups = [
        cache_group_config(spec, total_pages=64, cache_blocks_per_lcm_block=1)
        for spec in group_specs
    ]
    specs.host_group_packing = (1, 1)

    def config(policy: str):
        args = prepare_server_args(
            [
                "--model",
                "x",
                "--max-num-seqs",
                "8",
                "--disaggregation-mode",
                "prefill",
                "--kvstore-write-policy",
                policy,
            ]
        )
        return _config(args, specs)

    ts.Scheduler(config("write_through"))
    ts.Scheduler(config("write_back"))


@pytest.mark.parametrize(
    "role_flags, pool",
    [
        (["--disaggregation-mode", "prefill"], (1, 0)),
        (["--decode-context-parallel-size", "2"], (5, 8)),
    ],
    ids=["prefill", "fused"],
)
def test_a_kv_parallel_engine_writes_back_its_sharded_group(role_flags, pool):
    # A KV-parallel prefill or fused engine (the fused one decoding the
    # sharded pages with DCP and retracting into a snapshot pool) shards its
    # full-attention group over the KVP ranks, as the Kimi K3 recipe does;
    # the scheduler built from it takes write-back as it takes write-through.
    layer_types = ("full_attention",)
    (full,) = specs_for_layers(
        layer_types=layer_types,
        group_ids=make_layer_group_ids(layer_num=1, layer_types=layer_types),
        prefix_granularity=16,
        pd_disaggregation_enabled="prefill" in role_flags,
    )
    specs = _specs(num_host_pages=17, num_snapshot_pages=pool[0], max_retracted=pool[1])
    specs.cache_groups = [
        cache_group_config(
            dataclasses.replace(full, shard_count=2),
            total_pages=1 + 63 * 2,
            cache_blocks_per_lcm_block=2,
        )
    ]
    specs.host_group_packing = (2,)
    policies = ts.SchedulerConfig.HostWritePolicy
    for flag, policy in (
        ("write_back", policies.WriteBack),
        ("write_through", policies.WriteThrough),
    ):
        args = prepare_server_args(
            [
                "--model",
                "x",
                "--max-num-seqs",
                "8",
                *role_flags,
                "--attn-tp-size",
                "2",
                "--kv-parallel-size",
                "2",
                "--kvstore-write-policy",
                flag,
            ]
        )
        cfg = _config(args, specs)
        assert cfg.host_write_policy == policy
        assert cfg.cache_groups[0].shard_count == 2
        ts.Scheduler(cfg)


@pytest.mark.parametrize(
    "world, attn_tp, attn_dp, kvp",
    [(8, 8, 1, 8), (8, 8, 1, 4), (16, 8, 2, 8), (8, 8, 1, 1), (4, 2, 2, 2)],
)
def test_every_kv_group_lies_in_its_attention_tp_group(world, attn_tp, attn_dp, kvp):
    # A segment written back publishes once every rank of the replica holds
    # its ACK, and without a pipeline rust_host.py passes the attention TP
    # group as ``replica_ranks`` (this checks the Mapping, not that wiring).
    # The ranks that own a page between them, its KVP group, must all be in
    # that group, or the segment could publish before an owner copied its
    # rows.
    for rank in range(world):
        mapping = Mapping(
            rank=rank,
            world_size=world,
            attn_tp_size=attn_tp,
            attn_dp_size=attn_dp,
            attn_kvp_size=kvp,
        )
        assert len(mapping.attn.kvp_group) == kvp
        assert set(mapping.attn.kvp_group) <= set(mapping.attn.tp_group)


def test_an_unknown_host_write_policy_is_refused():
    # The command line offers the two policies; the dataclass checks the same.
    with pytest.raises(ValueError, match="--kvstore-write-policy must be one of"):
        ServerArgs(model="x", max_num_seqs=8, kvstore_write_policy="write_around")


def test_a_decode_server_takes_the_in_forward_state_checkpoint_flag_as_a_no_op():
    # A decode server computes no prompt, so a PD launcher may pass the flag
    # to both roles while disabling prefix caching only on decode; neither
    # its scheduler nor its executor (whose prefill kernels a decode node may
    # lack) turns the checkpoint on.
    args = ServerArgs(
        model="x",
        max_num_seqs=8,
        disaggregation_mode="decode",
        disaggregation_transfer_backend="mooncake",
        disable_kvstore=True,
        enable_prefix_caching=False,
        enable_state_checkpoint_in_forward=True,
    )
    assert not args.writes_state_checkpoints_in_forward
    cfg = _config(args, _specs(num_host_pages=0, num_snapshot_pages=5, max_retracted=8))
    assert not cfg.state_checkpoint_in_forward


@pytest.mark.parametrize(
    ("mapping", "title"),
    [
        (Mapping(rank=0, world_size=1), "tokenspeed::scheduler_tp0"),
        (Mapping(rank=3, world_size=4, moe_ep_size=4), "tokenspeed::scheduler_tp3_ep3"),
        (Mapping(rank=5, world_size=8, pp_size=2), "tokenspeed::scheduler_tp1_pp1"),
        (
            Mapping(
                rank=6, world_size=8, attn_tp_size=2, attn_dp_size=4, moe_ep_size=8
            ),
            "tokenspeed::scheduler_tp0_ep6_dp3",
        ),
    ],
)
def test_a_rank_is_titled_by_its_parallel_ranks(mapping, title):
    # The attention TP rank always shows; another dimension only when it
    # has more than one rank.
    assert rank_process_title(mapping) == title
