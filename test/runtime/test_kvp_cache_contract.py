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

"""KV parallelism: the virtual-block contract on the CPU.

The scheduler addresses a sharded group by virtual block ID; each KVP rank
owns every D-th block and translates before it reads, writes or zeroes. These
tests pin the contract's arithmetic, the translation, and the configuration
checks -- none of which needs a GPU or a second process.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.cache.transfer.layout import (
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation
from tokenspeed.runtime.distributed.mapping import AttentionLayerMapping
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.virtual_blocks import (
    local_pages,
    local_pages_by_group,
    owned_local_pages,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
)


def _full_history_spec(group_id: str, *, shard_count: int) -> CacheGroupSpec:
    return CacheGroupSpec(
        group_id=group_id,
        attn_kind="full_history",
        rows_per_page=64,
        entry_stride_tokens=4,
        sliding_window_tokens=None,
        shard_count=shard_count,
        replayable=False,
    )


def _contract(*, parents: int, packing: int, shard_count: int) -> CacheRuntimeContract:
    """One sharded and one replicated group over the same parents."""
    specs = (
        _full_history_spec("sharded", shard_count=shard_count),
        _full_history_spec("replicated", shard_count=1),
    )
    return CacheRuntimeContract(
        prefix_granularity=256,
        num_lcm_blocks=parents,
        token_capacity=parents * packing * shard_count * 256,
        group_specs=specs,
        group_page_counts={spec.group_id: 1 + parents * packing for spec in specs},
        group_packing={spec.group_id: packing for spec in specs},
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_host_ownership_uses_each_tiers_packing_and_bounds(rank):
    layout = CacheTransferLayout(
        num_lcm_blocks=2,
        groups=(
            CacheGroupLayout("sharded", 4, (CacheField("kv", 0, 0, 32, 32),)),
            CacheGroupLayout("replicated", 4, (CacheField("state", 0, 0, 80, 80),)),
        ),
        buffers=(object(),),
        consumers=(("kv", "state"),),
    )
    contract = _contract(parents=2, packing=4, shard_count=2)
    owners = BlockOwnerTranslation.for_host_pool(
        layout,
        contract,
        num_host_lcm_blocks=2,
        host_group_packing=(2, 1),
        rank=rank,
    )
    rows = [(0, 16, 8), (1, 8, 2)]
    expected = [(1, (1, 8, 2))]
    if rank == 1:
        expected.insert(0, (0, (0, 8, 4)))
    assert owners.owned_positions(rows) == expected
    for row in [(0, 17, 7), (0, 15, 9), (1, 8, 3)]:
        with pytest.raises(IndexError):
            owners.owned_positions([row])
    for packing in [(2,), (2, 0), (2, -1)]:
        with pytest.raises(ValueError):
            BlockOwnerTranslation.for_host_pool(
                layout,
                contract,
                num_host_lcm_blocks=2,
                host_group_packing=packing,
                rank=rank,
            )


class TestRuntimeContractTest:
    def test_virtual_counts_scale_physical_pages_by_shard_count(self, subtests):
        for shard_count in (1, 2, 4, 8):
            with subtests.test(shard_count=shard_count):
                contract = _contract(parents=5, packing=3, shard_count=shard_count)
                assert (contract.virtual_block_counts) == (
                    {"sharded": 1 + 5 * 3 * shard_count, "replicated": 1 + 5 * 3}
                )
                assert (contract.virtual_packing) == (
                    {"sharded": 3 * shard_count, "replicated": 3}
                )
                # Physical facts are untouched by the placement.
                assert (contract.group_page_counts["sharded"]) == (1 + 5 * 3)
                assert (contract.group_packing["sharded"]) == (3)

    def test_token_capacity_may_use_the_sharded_group_but_not_exceed_it(self):
        _contract(parents=4, packing=2, shard_count=4)  # 4 * 2 * 4 * 256 tokens fit.
        with pytest.raises((ValueError), match=("child-page capacity")):
            CacheRuntimeContract(
                prefix_granularity=256,
                num_lcm_blocks=4,
                token_capacity=4 * 2 * 4 * 256 + 1,
                group_specs=(_full_history_spec("sharded", shard_count=4),),
                group_page_counts={"sharded": 1 + 4 * 2},
                group_packing={"sharded": 2},
            )

    def test_virtual_ids_must_fit_int32_even_when_local_storage_does(self):
        spec = _full_history_spec("sharded", shard_count=8)
        pages = (1 << 29) + 1  # local IDs fit; 8 owners overflow int32
        with pytest.raises((ValueError), match=("int32")):
            CacheRuntimeContract(
                prefix_granularity=256,
                num_lcm_blocks=pages - 1,
                token_capacity=256,
                group_specs=(spec,),
                group_page_counts={"sharded": pages},
                group_packing={"sharded": 1},
            )


class TestLocalPagesTest:
    def test_owned_blocks_partition_every_non_null_virtual_block(self, subtests):
        for shard_count in (1, 2, 3, 4):
            with subtests.test(shard_count=shard_count):
                count = 1 + 6 * shard_count
                virtual = list(range(count))
                seen = []
                for rank in range(shard_count):
                    owned = local_pages(
                        virtual,
                        shard_count=shard_count,
                        rank=rank,
                        virtual_block_count=count,
                    )
                    # Each rank owns exactly the local pages 1..6, in order.
                    assert (owned.tolist()) == (list(range(1, 7)))
                    seen.extend(
                        block
                        for block in virtual
                        if block > 0 and (block - 1) % shard_count == rank
                    )
                assert (sorted(seen)) == (virtual[1:])

    def test_null_block_and_duplicates(self):
        assert (
            local_pages(
                [0, 3, 3, 0, 1], shard_count=2, rank=0, virtual_block_count=9
            ).tolist()
        ) == ([2, 2, 1])
        assert (
            local_pages([0, 0], shard_count=1, rank=0, virtual_block_count=9).tolist()
        ) == ([])
        # The scheduler's zero-copy export is a read-only int32 array.
        exported = np.asarray([0, 3, 3, 0, 1], dtype=np.int32)
        exported.setflags(write=False)
        assert (
            local_pages(exported, shard_count=2, rank=0, virtual_block_count=9).tolist()
        ) == ([2, 2, 1])

    def test_owner_mask_pairs_owned_pages_with_their_input_positions(self):
        owned, local = owned_local_pages(
            [0, 3, 3, 0, 1, 4], shard_count=2, rank=0, virtual_block_count=9
        )
        assert (owned.tolist()) == ([False, True, True, False, True, False])
        assert (local.tolist()) == ([2, 2, 1])
        owned, local = owned_local_pages(
            [], shard_count=2, rank=0, virtual_block_count=9
        )
        assert ((owned.tolist(), local.tolist())) == (([], []))

    def test_by_group_reads_shard_count_and_bounds_from_the_contract(self):
        contract = _contract(parents=2, packing=2, shard_count=2)
        translated = local_pages_by_group(
            {"sharded": [0, 1, 2, 3, 4, 8], "replicated": [0, 1, 4]},
            contract=contract,
            rank=1,
        )
        # Sharded: rank 1 owns virtual 2, 4, 6, 8 -> local 1, 2, 3, 4.
        assert ({group: pages.tolist() for group, pages in translated.items()}) == (
            {"sharded": [1, 2, 4], "replicated": [1, 4]}
        )
        with pytest.raises((IndexError)):
            local_pages_by_group({"replicated": [5]}, contract=contract, rank=0)


class TestMappingTest:
    def test_kvp_subgroups_are_consecutive_within_attention_tp(self):
        for rank in range(8):
            mapping = AttentionLayerMapping(
                rank=rank, world_size=8, tp_size=8, dp_size=1, kvp_size=4
            )
            assert mapping.has_kvp
            assert (mapping.kvp_rank) == (rank % 4)
            assert (mapping.kvp_replica_rank) == (rank // 4)
            assert (mapping.kvp_group) == (
                tuple(range(rank - rank % 4, rank - rank % 4 + 4))
            )
        plain = AttentionLayerMapping(
            rank=3, world_size=8, tp_size=8, dp_size=1, kvp_size=1
        )
        assert not (plain.has_kvp)
        assert (plain.kvp_group) == ((3,))


class TestConfigurationTest:
    @staticmethod
    def _component(backend_name: str) -> SoftmaxAttnConfig:
        return SoftmaxAttnConfig(
            backend_name=backend_name,
            num_attention_heads=8,
            num_kv_heads=1,
            head_dim=512,
            attn_tp_size=2,
        )

    def _config(self, **overrides) -> AttnConfig:
        fields = dict(
            device="cpu",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_quant_method="none",
            kernel_page_size=64,
            context_len=4096,
            max_bs=4,
            kvp_size=2,
            kvp_rank=0,
            kvp_group=(0, 1),
            components=(self._component("tokenspeed_mla"),),
        )
        fields.update(overrides)
        return AttnConfig(**fields)

    def test_kvp_requires_a_dcp_capable_backend(self):
        with pytest.raises((ValueError), match=("DSA, FlashMLA")):
            self._config(components=(self._component("mha"),))

    def test_kvp_keeps_host_kvstore_and_allows_l3(self):
        # Host blocks are allocated in their Device block's residue class, so
        # the KVStore (and the retraction snapshot pool) serve a sharded
        # engine. L3 keys use a replica identity independent of allocation.
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "decode"
        args.kv_parallel_size = 2
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args.kvstore_storage_backend = None
        args._handle_kvstore()
        assert args.enable_kvstore, "KVStore is on by default"
        args.validate_cache_options()
        args.kvstore_storage_backend = "mooncake"
        args.validate_cache_options()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
