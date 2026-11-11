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

"""Cache binding bounds int32 router slots before allocating cache storage."""

import pytest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


def contract(parents, *, degree, entry_stride):
    spec = CacheGroupSpec(
        group_id="history",
        attn_kind="full_history",
        rows_per_page=128,
        entry_stride_tokens=entry_stride,
        shard_count=degree,
        replayable=False,
    )
    return CacheRuntimeContract(
        prefix_granularity=spec.block_granularity,
        num_lcm_blocks=parents,
        token_capacity=parents * degree * spec.block_granularity,
        group_specs=(spec,),
        group_page_counts={spec.group_id: parents + 1},
        group_packing={spec.group_id: 1},
    )


@pytest.mark.parametrize("degree", [1, 2, 8])
@pytest.mark.parametrize("entry_stride", [1, 4])
def test_virtual_slot_limit_includes_null_block_and_shard_owners(degree, entry_stride):
    span = 128 * entry_stride
    parents = ((1 << 31) // span - 1) // degree
    bound = contract(parents, degree=degree, entry_stride=entry_stride)
    assert bound.virtual_block_counts["history"] * span - 1 < (1 << 31)
    with pytest.raises(ValueError, match="cache slots.*exceed int32"):
        contract(parents + 1, degree=degree, entry_stride=entry_stride)
