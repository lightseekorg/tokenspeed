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

"""The KV event wire schema (``pd/kv_events.py``), which the Rust publisher
writes byte for byte (``ts-control``'s ``kv_events`` fixture)."""

import os
import sys

import msgspec

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.pd.kv_events import (  # noqa: E402
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)


def test_kv_event_batch_msgpack_shape_is_dynamo_compatible() -> None:
    payload = msgspec.msgpack.encode(
        KVEventBatch(
            ts=1.5,
            events=[
                BlockStored(
                    block_hashes=[123],
                    parent_block_hash=None,
                    token_ids=[1, 2],
                    block_size=2,
                ),
                BlockRemoved(block_hashes=[123]),
            ],
            attn_dp_rank=2,
        )
    )

    decoded = msgspec.msgpack.decode(payload)

    # Every slot of the reference publisher's layout is written; the ones
    # this engine does not fill ride as None.
    assert decoded == [
        1.5,
        [
            [
                "BlockStored",
                [123],
                None,
                [1, 2],
                2,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ["BlockRemoved", [123], None, None],
        ],
        2,
    ]


def test_kv_events_name_their_cache_group() -> None:
    """A hybrid model's groups publish the same boundary each under its own
    group: the History chain, the State checkpoint (``mamba``) and the window;
    the group slots sit after four unused ones, where the reference consumer
    reads them."""
    payload = msgspec.msgpack.encode(
        KVEventBatch(
            ts=1.5,
            events=[
                BlockStored(
                    block_hashes=[123],
                    parent_block_hash=None,
                    token_ids=[1, 2],
                    block_size=2,
                    group_idx=0,
                    kv_cache_spec_kind="full_attention",
                ),
                BlockStored(
                    block_hashes=[123],
                    parent_block_hash=None,
                    token_ids=[1, 2],
                    block_size=2,
                    group_idx=1,
                    kv_cache_spec_kind="mamba",
                ),
                BlockStored(
                    block_hashes=[123],
                    parent_block_hash=None,
                    token_ids=[1, 2],
                    block_size=2,
                    group_idx=2,
                    kv_cache_spec_kind="sliding_window",
                    kv_cache_spec_sliding_window=512,
                ),
                BlockRemoved(block_hashes=[123], group_idx=1),
            ],
            attn_dp_rank=0,
        )
    )

    decoded = msgspec.msgpack.decode(payload)

    assert decoded == [
        1.5,
        [
            [
                "BlockStored",
                [123],
                None,
                [1, 2],
                2,
                None,
                None,
                None,
                None,
                0,
                "full_attention",
                None,
            ],
            [
                "BlockStored",
                [123],
                None,
                [1, 2],
                2,
                None,
                None,
                None,
                None,
                1,
                "mamba",
                None,
            ],
            [
                "BlockStored",
                [123],
                None,
                [1, 2],
                2,
                None,
                None,
                None,
                None,
                2,
                "sliding_window",
                512,
            ],
            ["BlockRemoved", [123], None, 1],
        ],
        0,
    ]
    typed = msgspec.msgpack.decode(payload, type=KVEventBatch)
    assert [event.group_idx for event in typed.events] == [0, 1, 2, 1]
    assert typed.events[2].kv_cache_spec_sliding_window == 512
