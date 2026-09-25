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

"""Decode context parallelism: the virtual-block contract on the CPU.

The scheduler addresses a sharded group by virtual block ID; each DCP rank
owns every D-th block and translates before it reads, writes or zeroes. These
tests pin the contract's arithmetic, the translation at every layer that
performs it, the recipe's group declarations, and the configuration checks --
none of which needs a GPU or a second process.
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.distributed.mapping import AttentionLayerMapping
from tokenspeed.runtime.layers.attention.configs import base as configs_base
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_KV_GROUP_ID,
    parse_v4_compressed_kv_group_id,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.virtual_blocks import (
    local_pages,
    local_pages_by_group,
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
        retention="full_history",
        rows_per_page=64,
        entry_stride_tokens=4,
        sliding_window_tokens=None,
        family="history",
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


def _recipe(*, dcp_size: int, fp4: bool, draft: bool) -> DeepseekV4Recipe:
    # DeepSeek V4 Flash's layer ratios: 43 target layers plus one draft layer.
    hf = SimpleNamespace(
        compress_ratios=(0, 0) + (4, 128) * 20 + (4, 0),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )
    return DeepseekV4Recipe(
        server_args=SimpleNamespace(
            max_total_tokens=None,
            chunked_prefill_size=8192,
            disaggregation_mode="null",
            enable_prefix_caching=True,
            attention_use_fp4_indexer_cache=fp4,
        ),
        model_config=SimpleNamespace(hf_config=hf, num_attention_layers=43),
        attn_config=SimpleNamespace(
            prefix_granularity=256,
            max_bs=16,
            context_len=4096,
            pd_disaggregation_enabled=False,
            dcp_size=dcp_size,
        ),
        draft_model_config=(
            SimpleNamespace(hf_config=hf, num_attention_layers=1) if draft else None
        ),
        draft_attn_config=SimpleNamespace(dcp_size=dcp_size) if draft else None,
        cache_budget_bytes=8 << 30,
        decode_input_tokens=4,
        overlap_schedule_depth=0,
    )


class RuntimeContractTest(unittest.TestCase):
    def test_virtual_counts_scale_physical_pages_by_shard_count(self):
        for shard_count in (1, 2, 4, 8):
            with self.subTest(shard_count=shard_count):
                contract = _contract(parents=5, packing=3, shard_count=shard_count)
                self.assertEqual(
                    contract.virtual_block_counts,
                    {"sharded": 1 + 5 * 3 * shard_count, "replicated": 1 + 5 * 3},
                )
                self.assertEqual(
                    contract.virtual_packing,
                    {"sharded": 3 * shard_count, "replicated": 3},
                )
                # Physical facts are untouched by the placement.
                self.assertEqual(contract.group_page_counts["sharded"], 1 + 5 * 3)
                self.assertEqual(contract.group_packing["sharded"], 3)

    def test_token_capacity_may_use_the_sharded_group_but_not_exceed_it(self):
        _contract(parents=4, packing=2, shard_count=4)  # 4 * 2 * 4 * 256 tokens fit.
        with self.assertRaisesRegex(ValueError, "child-page capacity"):
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
        with self.assertRaisesRegex(ValueError, "int32"):
            CacheRuntimeContract(
                prefix_granularity=256,
                num_lcm_blocks=pages - 1,
                token_capacity=256,
                group_specs=(spec,),
                group_page_counts={"sharded": pages},
                group_packing={"sharded": 1},
            )

    def test_shard_count_must_be_a_positive_integer(self):
        for value in (0, -1, 2.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _full_history_spec("g", shard_count=value)


class LocalPagesTest(unittest.TestCase):
    def test_owned_blocks_partition_every_non_null_virtual_block(self):
        for shard_count in (1, 2, 3, 4):
            with self.subTest(shard_count=shard_count):
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
                    self.assertEqual(owned, list(range(1, 7)))
                    seen.extend(
                        block
                        for block in virtual
                        if block > 0 and (block - 1) % shard_count == rank
                    )
                self.assertEqual(sorted(seen), virtual[1:])

    def test_null_block_and_duplicates(self):
        self.assertEqual(
            local_pages([0, 3, 3, 0, 1], shard_count=2, rank=0, virtual_block_count=9),
            [2, 2, 1],
        )
        self.assertEqual(
            local_pages([0, 0], shard_count=1, rank=0, virtual_block_count=9), []
        )

    def test_out_of_range_ids_and_ranks_are_rejected(self):
        with self.assertRaises(IndexError):
            local_pages([9], shard_count=2, rank=0, virtual_block_count=9)
        with self.assertRaises(IndexError):
            local_pages([-1], shard_count=2, rank=0, virtual_block_count=9)
        with self.assertRaises(ValueError):
            local_pages([1], shard_count=2, rank=2, virtual_block_count=9)
        with self.assertRaises(ValueError):
            local_pages([1], shard_count=0, rank=0, virtual_block_count=9)

    def test_by_group_reads_shard_count_and_bounds_from_the_contract(self):
        contract = _contract(parents=2, packing=2, shard_count=2)
        translated = local_pages_by_group(
            {"sharded": [0, 1, 2, 3, 4, 8], "replicated": [0, 1, 4]},
            contract=contract,
            rank=1,
        )
        # Sharded: rank 1 owns virtual 2, 4, 6, 8 -> local 1, 2, 3, 4.
        self.assertEqual(translated, {"sharded": [1, 2, 4], "replicated": [1, 4]})
        with self.assertRaises(IndexError):
            local_pages_by_group({"replicated": [5]}, contract=contract, rank=0)


def _metadata(*, dcp_size: int, dcp_rank: int, table: torch.Tensor):
    group_id = v4_compressed_kv_group_id(4)
    specs = (
        replace(_full_history_spec(group_id, shard_count=dcp_size)),
        _full_history_spec(V4_INDEXER_KV_GROUP_ID, shard_count=1),
    )
    # Eight virtual blocks shared by the owners: 8 // dcp_size local pages each.
    contract = SimpleNamespace(
        group_specs=specs,
        virtual_block_counts={group_id: 9, V4_INDEXER_KV_GROUP_ID: 9},
    )
    return DeepseekV4CacheMetadata.from_group_tables(
        page_size=64,
        page_table=torch.zeros((table.shape[0], 1), dtype=torch.int32),
        block_tables={group_id: table, V4_INDEXER_KV_GROUP_ID: table.clone()},
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        runtime_contract=contract,
    )


class CacheMetadataTranslationTest(unittest.TestCase):
    def test_read_tables_mark_null_and_foreign_pages_and_refresh_in_place(self):
        table = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]], dtype=torch.int32
        )
        for dcp_size in (1, 2, 4):
            for rank in range(dcp_size):
                with self.subTest(dcp_size=dcp_size, rank=rank):
                    metadata = _metadata(dcp_size=dcp_size, dcp_rank=rank, table=table)
                    metadata.refresh_page_tables()
                    read = metadata.compressed_page_table(4)
                    virtual = table.long()
                    owned = (virtual > 0) & ((virtual - 1) % dcp_size == rank)
                    expected = torch.where(
                        owned,
                        (virtual - 1) // dcp_size + 1,
                        torch.full_like(virtual, -1),
                    )
                    self.assertTrue(torch.equal(read.long(), expected))
                    # Compressed KV and the independent indexer group share
                    # the placement contract, not physical page IDs.
                    self.assertEqual(
                        set(metadata.compressed_page_tables),
                        {v4_compressed_kv_group_id(4), V4_INDEXER_KV_GROUP_ID},
                    )
                    table[0, 0] = 2
                    metadata.refresh_page_tables()
                    self.assertEqual(
                        metadata.compressed_page_table(4).data_ptr(),
                        read.data_ptr(),
                        "graph-captured read tables must be refreshed, not replaced",
                    )
                    table[0, 0] = 1

    def test_write_slots_translate_and_mask_for_every_dcp_size(self):
        table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        slots = torch.tensor(
            [0, 63, 64, 128, 200, 256, 511, 576, -1], dtype=torch.int64
        )
        for dcp_size in (1, 2, 4):
            for rank in range(dcp_size):
                with self.subTest(dcp_size=dcp_size, rank=rank):
                    metadata = _metadata(dcp_size=dcp_size, dcp_rank=rank, table=table)
                    local, owned = metadata.local_compressed_write_slots(slots, 4)
                    block = slots.clamp_min(0) // 64
                    expected_owned = (slots >= 64) & (block < 9)
                    expected_owned &= (block - 1) % dcp_size == rank
                    self.assertTrue(torch.equal(owned, expected_owned))
                    expected_local = (
                        (block - 1) // dcp_size + 1
                    ) * 64 + slots.clamp_min(0) % 64
                    self.assertTrue(
                        torch.equal(local[owned], expected_local[expected_owned])
                    )
                    self.assertTrue((local[~owned] == 0).all())

    def test_slot_mapping_never_targets_the_null_block(self):
        # Request 1's second compressed page is the null block: its tokens
        # write nothing instead of clobbering page 0.
        table = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
        metadata = _metadata(dcp_size=1, dcp_rank=0, table=table)
        positions = torch.tensor([3, 255, 259, 3, 259], dtype=torch.int64)
        token_to_req = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        for indexer in (False, True):
            with self.subTest(indexer=indexer):
                slots = metadata.compressed_slot_mapping(
                    positions,
                    4,
                    token_to_req_indices=token_to_req,
                    query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
                    seq_lens=torch.tensor([260, 260], dtype=torch.int32),
                    indexer=indexer,
                    kv_cache_block_size=64,
                )
                self.assertEqual(slots.tolist(), [64, 127, 128, 192, -1])

    def test_indexer_table_is_its_own_sharded_group(self):
        table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        metadata = _metadata(dcp_size=4, dcp_rank=3, table=table)
        self.assertTrue(torch.equal(metadata.indexer_block_table(), table))
        metadata.refresh_page_tables()
        self.assertEqual(metadata.indexer_page_table().tolist(), [[-1, -1, -1, 1]])
        self.assertEqual(
            metadata.local_indexer_write_slots(
                torch.tensor([64, 128, 192, 256, -1]), 64
            ).tolist(),
            [-1, -1, -1, 64, -1],
        )
        with self.assertRaisesRegex(RuntimeError, "missing cache-group block table"):
            _metadata(dcp_size=4, dcp_rank=3, table=table).compressed_block_table(128)

    def test_request_slices_keep_placement_and_read_views(self):
        table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
        metadata = _metadata(dcp_size=2, dcp_rank=1, table=table)
        metadata.refresh_page_tables()
        sliced = metadata.slice_requests(1, 3)
        self.assertEqual((sliced.dcp_size, sliced.dcp_rank), (2, 1))
        self.assertIs(sliced.runtime_contract, metadata.runtime_contract)
        self.assertTrue(
            torch.equal(
                sliced.compressed_page_table(4),
                metadata.compressed_page_table(4)[1:3],
            )
        )
        self.assertEqual(
            sliced.compressed_page_table(4).data_ptr(),
            metadata.compressed_page_table(4)[1:3].data_ptr(),
        )


class RecipeDeclarationTest(unittest.TestCase):
    def test_compressed_and_indexer_groups_are_sharded_and_states_replicated(self):
        for dcp_size in (1, 2, 4):
            with self.subTest(dcp_size=dcp_size):
                specs = [
                    spec
                    for spec, _ in _recipe(
                        dcp_size=dcp_size, fp4=True, draft=True
                    ).groups()
                ]
                by_id = {spec.group_id: spec for spec in specs}
                self.assertIn(V4_INDEXER_KV_GROUP_ID, by_id)
                self.assertEqual(by_id[V4_INDEXER_KV_GROUP_ID].shard_count, dcp_size)
                self.assertEqual(
                    by_id[V4_INDEXER_KV_GROUP_ID].retention, "full_history"
                )
                for spec in specs:
                    expected = (
                        dcp_size
                        if (
                            parse_v4_compressed_kv_group_id(spec.group_id)
                            or spec.group_id == V4_INDEXER_KV_GROUP_ID
                        )
                        else 1
                    )
                    self.assertEqual(spec.shard_count, expected, spec.group_id)

    def test_group_set_does_not_depend_on_the_dcp_size(self):
        ids = {
            dcp_size: [
                spec.group_id
                for spec, _ in _recipe(
                    dcp_size=dcp_size, fp4=False, draft=False
                ).groups()
            ]
            for dcp_size in (1, 4)
        }
        self.assertEqual(ids[1], ids[4])

    def test_capacity_is_bounded_by_the_replicated_groups(self):
        for fp4 in (False, True):
            with self.subTest(fp4=fp4):
                recipe = _recipe(dcp_size=4, fp4=fp4, draft=True)
                groups = recipe.groups()
                layout = pack(
                    groups,
                    prefix_granularity=recipe.prefix_granularity,
                    cache_blocks_per_lcm_block=recipe.packing(groups),
                    alignment=recipe.alignment,
                    max_padding_fraction=recipe.max_padding_fraction,
                )
                parents = recipe.num_lcm_blocks(layout)
                capacity = recipe.token_capacity(layout, parents)
                # Every group can hold the admitted tokens: the recipe's parent
                # demand at token_capacity never exceeds the parents it planned,
                # and one more token's worth would. Replicated SWA and state
                # demands still constrain the shared physical budget.
                self.assertLessEqual(recipe.parents_needed(layout, capacity), parents)
                self.assertGreater(
                    recipe.parents_needed(layout, capacity + 256), parents
                )

    def test_dcp_raises_capacity_without_growing_the_arena(self):
        base = _recipe(dcp_size=1, fp4=True, draft=True).setup().spec
        sharded = _recipe(dcp_size=4, fp4=True, draft=True).setup().spec
        self.assertGreater(sharded.token_capacity, base.token_capacity)
        self.assertLessEqual(
            sharded.memory_plan.arena_bytes,
            _recipe(dcp_size=4, fp4=True, draft=True).cache_budget_bytes,
        )
        # Physical geometry is identical: every group packs the same physical
        # children into one parent, and only the scheduler's virtual view
        # widens. The parent count itself may differ by the capacity search's
        # rounding, so compare the per-parent packing rather than raw pages.
        for spec in base.cache_group_specs:
            self.assertEqual(
                (base.memory_plan.group(spec.group_id).page_count - 1)
                // base.memory_plan.num_lcm_blocks,
                (sharded.memory_plan.group(spec.group_id).page_count - 1)
                // sharded.memory_plan.num_lcm_blocks,
                spec.group_id,
            )


class PrefillExchangePlanTest(unittest.TestCase):
    def test_history_rows_are_routed_to_their_owner_and_restore_request_order(self):
        from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4 import (
            DeepseekV4AttentionBackend,
        )
        from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
            DeepseekV4ForwardMetadata,
        )

        degree = 2
        # Two requests: 300 tokens (75 compressed rows over pages 1, 2) and
        # 64 tokens (16 rows on page 5). Page 5's owner is rank 0, 1 -> 0, 2 -> 1.
        table = torch.tensor([[1, 2], [5, 0]], dtype=torch.int32)
        seq_lens = torch.tensor([300, 64], dtype=torch.int32)
        query_lens = torch.tensor([4, 64], dtype=torch.int32)
        for rank in range(degree):
            with self.subTest(rank=rank):
                cache = _metadata(dcp_size=degree, dcp_rank=rank, table=table)
                metadata = DeepseekV4ForwardMetadata(
                    seq_lens=seq_lens,
                    query_lens=query_lens,
                    query_start_loc=torch.tensor([0, 4, 68], dtype=torch.int32),
                    token_to_req_indices=torch.repeat_interleave(
                        torch.arange(2, dtype=torch.int32), query_lens
                    ),
                    cache=cache,
                    is_valid_token=None,
                    seq_lens_cpu=seq_lens,
                    query_lens_cpu=query_lens,
                    forward_mode=None,
                    num_prefill_reqs=2,
                    num_prefill_tokens=68,
                )
                chunks = DeepseekV4AttentionBackend._build_dcp_prefill_chunks(
                    metadata, 4, table, chunk_size=8, window_size=128
                )
                self.assertEqual(list(chunks), [(0, 2)])
                chunk = chunks[0, 2]
                width = chunk.workspace_width
                # Request 0: rows 0..63 live on virtual page 1 (owner 0), rows
                # 64..74 on page 2 (owner 1). Request 1: rows 0..15 on page 5
                # (owner 0). Row destinations are workspace-relative.
                expected_by_owner = {
                    0: list(range(0, 64)) + [width + row for row in range(16)],
                    1: list(range(64, 75)),
                }
                self.assertEqual(
                    chunk.counts, [len(expected_by_owner[0]), len(expected_by_owner[1])]
                )
                self.assertEqual(
                    chunk.local_destinations.tolist(), expected_by_owner[rank]
                )
                self.assertEqual(
                    chunk.destinations.tolist(),
                    expected_by_owner[0] + expected_by_owner[1],
                )


class MappingTest(unittest.TestCase):
    def test_dcp_subgroups_are_consecutive_within_attention_tp(self):
        for rank in range(8):
            mapping = AttentionLayerMapping(
                rank=rank, world_size=8, tp_size=8, cp_size=1, dp_size=1, dcp_size=4
            )
            self.assertTrue(mapping.has_dcp)
            self.assertEqual(mapping.dcp_rank, rank % 4)
            self.assertEqual(mapping.dcp_replica_rank, rank // 4)
            self.assertEqual(
                mapping.dcp_group, tuple(range(rank - rank % 4, rank - rank % 4 + 4))
            )
        plain = AttentionLayerMapping(
            rank=3, world_size=8, tp_size=8, cp_size=1, dp_size=1, dcp_size=1
        )
        self.assertFalse(plain.has_dcp)
        self.assertEqual(plain.dcp_group, (3,))

    def test_dcp_must_divide_attention_tp(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            AttentionLayerMapping(
                rank=0, world_size=8, tp_size=8, cp_size=1, dp_size=1, dcp_size=3
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            AttentionLayerMapping(
                rank=0, world_size=8, tp_size=8, cp_size=1, dp_size=1, dcp_size=0
            )


class ConfigurationTest(unittest.TestCase):
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
            prefix_granularity=256,
            kernel_page_size=64,
            context_len=4096,
            max_bs=4,
            dcp_size=2,
            dcp_rank=0,
            dcp_group=(0, 1),
            components=(self._component("deepseek_v4"),),
        )
        fields.update(overrides)
        return AttnConfig(**fields)

    def test_dcp_requires_the_deepseek_v4_backend(self):
        with self.assertRaisesRegex(ValueError, "DeepSeek V4"):
            self._config(components=(self._component("mha"),))

    def test_dcp_requires_a_partial_capable_decode_kernel(self):
        with patch.object(
            configs_base, "dsv4_decode_supports_partials", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "no-sink LSE"):
                self._config()
        with patch.object(
            configs_base, "dsv4_decode_supports_partials", return_value=True
        ):
            self.assertEqual(self._config().dcp_size, 2)
        # A DCP size of one asks nothing of the kernel registry.
        with patch.object(
            configs_base,
            "dsv4_decode_supports_partials",
            side_effect=AssertionError("queried for dcp_size=1"),
        ):
            self._config(dcp_size=1, dcp_group=(0,))

    def test_dcp_rejects_kvstore_after_its_default_is_applied(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "null"
        args.decode_context_parallel_size = 2
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args._handle_kvstore()
        self.assertTrue(args.enable_kvstore, "KVStore is on by default")
        with self.assertRaisesRegex(ValueError, "KVStore"):
            args.validate_cache_options()
        args.disable_kvstore = True
        args.enable_kvstore = False
        args._handle_kvstore()
        args.validate_cache_options()


if __name__ == "__main__":
    unittest.main()
