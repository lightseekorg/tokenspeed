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

"""CPU-only scheduler_utils contract for shifted-input prefix hashing."""

import inspect

import pytest
from tokenspeed_scheduler import CacheGroupConfig, Scheduler

from tokenspeed.runtime.engine.scheduler_utils import make_config


@pytest.mark.parametrize(
    "algorithm,lookahead",
    [(None, 0), ("DFLASH", 0), ("DSPARK", 0), ("MTP", 1), ("EAGLE3", 1)],
)
@pytest.mark.parametrize("role", ["fused", "prefill", "decode"])
def test_config_selects_shifted_hash_coverage_without_changing_geometry(
    algorithm, lookahead, role
):
    cfg = make_config(
        num_device_pages=32,
        max_scheduled_tokens=256,
        max_batch_size=4,
        prefix_granularity=64,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role=role,
        speculative_algorithm=algorithm,
        decode_input_tokens=4,
        prefix_replay_tokens=128,
    )
    assert cfg.prefix_hash_lookahead_tokens == lookahead
    assert cfg.prefix_granularity == 64
    assert cfg.prefix_replay_tokens == 128
    assert cfg.decode_input_tokens == 4


def test_speculative_algorithm_is_required_and_old_extension_is_not_silently_supported():
    assert (
        inspect.signature(make_config).parameters["speculative_algorithm"].default
        is inspect.Parameter.empty
    )
    cfg = make_config(
        num_device_pages=32,
        max_scheduled_tokens=256,
        max_batch_size=4,
        prefix_granularity=64,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        speculative_algorithm="MTP",
        cache_groups=[
            CacheGroupConfig(group_id="full", block_granularity=64, total_pages=32)
        ],
    )
    scheduler = Scheduler(cfg)
    tokens = list(range(65))
    changed = tokens[:-1] + [1000]
    assert scheduler.prefix_hashes_for_tokens(
        tokens
    ) != scheduler.prefix_hashes_for_tokens(changed)
