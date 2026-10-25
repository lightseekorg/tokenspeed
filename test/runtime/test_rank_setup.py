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

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig  # noqa: E402
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.engine.rank_setup import (  # noqa: E402
    rank_process_title,
    scheduler_config_from_args,
)
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    ServerArgs,
    prepare_server_args,
)


def _specs(*, num_host_pages: int, num_snapshot_pages: int, max_retracted: int):
    return SimpleNamespace(
        cache_geometry=SimpleNamespace(num_device_pages=64, prefix_granularity=16),
        cache_groups=[],
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
    assert (cfg.num_snapshot_pages, cfg.max_retracted_requests) == (5, 8)


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
