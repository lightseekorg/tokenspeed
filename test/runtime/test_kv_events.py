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
    MEDIUM_HOST,
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
                BlockStored(
                    block_hashes=[124],
                    parent_block_hash=123,
                    token_ids=[3, 4],
                    block_size=2,
                    medium=MEDIUM_HOST,
                ),
                BlockRemoved(block_hashes=[124], medium=MEDIUM_HOST),
            ],
            attn_dp_rank=2,
        )
    )

    decoded = msgspec.msgpack.decode(payload)

    # vLLM's layout: a store's lora_id (None) then its medium, a removal's
    # medium; None is the GPU cache.
    assert decoded == [
        1.5,
        [
            ["BlockStored", [123], None, [1, 2], 2, None, None],
            ["BlockRemoved", [123], None],
            ["BlockStored", [124], 123, [3, 4], 2, None, "CPU_PINNED"],
            ["BlockRemoved", [124], "CPU_PINNED"],
        ],
        2,
    ]
