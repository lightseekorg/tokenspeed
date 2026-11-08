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

"""The KV event wire schema (``pd/kv_events.py``), which the scheduler's frame
drain writes byte for byte (``ts-control``'s ``kv_events`` fixture)."""

import os
import sys

import msgspec

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

    # Every slot of the reference publisher's layout is written, this
    # engine's group mask after the last one; the slots it does not fill
    # ride as None.
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
                None,
                None,
                None,
                None,
            ],
            ["BlockRemoved", [123], None, None, None, None, None],
        ],
        2,
    ]


def test_kv_events_name_their_cache_groups_in_one_event_per_boundary() -> None:
    """A hybrid model publishes a boundary once: the store names the groups
    that hold it in ``group_mask`` and the lowest newly holding group in the
    reference slots (its kind, its window); a removal names the groups that
    dropped it. The group slots sit after four unused ones, where the
    reference consumer reads them, and the mask after the reference layout's
    last slot."""
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
                    group_mask=0b011,
                ),
                BlockStored(
                    block_hashes=[123],
                    parent_block_hash=None,
                    token_ids=[1, 2],
                    block_size=2,
                    group_idx=2,
                    kv_cache_spec_kind="sliding_window",
                    kv_cache_spec_sliding_window=512,
                    group_mask=0b111,
                ),
                BlockRemoved(block_hashes=[123], group_idx=1, group_mask=0b010),
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
                None,
                None,
                None,
                0b011,
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
                None,
                None,
                None,
                0b111,
            ],
            ["BlockRemoved", [123], None, 1, None, None, 0b010],
        ],
        0,
    ]
    typed = msgspec.msgpack.decode(payload, type=KVEventBatch)
    assert [event.group_idx for event in typed.events] == [0, 2, 1]
    assert [event.group_mask for event in typed.events] == [0b011, 0b111, 0b010]
    assert typed.events[1].kv_cache_spec_sliding_window == 512
