from __future__ import annotations

import os
import sys
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import build_v4_cache_specs
from tokenspeed.runtime.configs.paged_cache_spec import validate_flat_scheduler_config
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends import deepseek_v4 as v4_backend
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    _group_slot_mapping_from_raw,
)
from tokenspeed.runtime.layers.attention.registry import (
    _validate_shared_flat_group_geometry,
)


def _config(
    *,
    device: str,
    is_draft: bool = False,
    context_len: int = 4096,
    speculative_tokens: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        page_size=64,
        device=device,
        num_attention_heads=64,
        num_kv_heads=1,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        is_draft=is_draft,
        speculative_num_draft_tokens=speculative_tokens,
        speculative_num_steps=1,
        head_dim=512,
        qk_rope_head_dim=64,
        context_len=context_len,
    )


def _backend(
    *,
    flat: bool,
    device: str = "cpu",
    is_draft: bool = False,
    context_len: int = 4096,
    speculative_tokens: int = 4,
) -> DeepseekV4AttentionBackend:
    with patch.object(v4_backend, "scheduler_ext_flat_kvcache", return_value=flat):
        return DeepseekV4AttentionBackend(
            _config(
                device=device,
                is_draft=is_draft,
                context_len=context_len,
                speculative_tokens=speculative_tokens,
            )
        )


def _target_specs():
    return tuple(
        build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(1, 4, 128),
        )
    )


def _draft_specs():
    return tuple(
        build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(1,),
        )
    )


def _tables(specs, *, rows: int = 2, cols: int = 2, device: str = "cpu"):
    return OrderedDict(
        (
            str(spec.group_id),
            torch.full(
                (rows, cols),
                index + 1,
                dtype=torch.int32,
                device=device,
            ),
        )
        for index, spec in enumerate(specs)
    )


def _page_counts(specs, count: int = 4096):
    return {str(spec.group_id): count for spec in specs}


class DeepseekV4FlatGroupUnitTest(unittest.TestCase):
    def test_cache_spec_identity_and_order_are_deterministic(self):
        first = _target_specs()
        second = _target_specs()
        expected = (
            "v4.swa_kv",
            "v4.c4a.compressor_state",
            "v4.c4a.compressed_kv",
            "v4.c128a.compressor_state",
            "v4.c128a.compressed_kv",
            "v4.c4a.indexer_compressor_state",
        )
        self.assertEqual(tuple(spec.group_id for spec in first), expected)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple((spec.block_size, spec.cache_blocks_per_lcm_block) for spec in first),
            ((64, 4), (4, 64), (256, 1), (8, 32), (256, 1), (4, 64)),
        )

    def test_eager_flat_maps_all_target_groups_in_decode_extend_and_mixed(self):
        specs = _target_specs()
        tables = _tables(specs)
        cases = (
            (
                ForwardMode.DECODE,
                dict(num_tokens=2),
            ),
            (
                ForwardMode.EXTEND,
                dict(
                    num_tokens=4,
                    extend_seq_lens_cpu=torch.tensor([2, 2], dtype=torch.int32),
                ),
            ),
            (
                ForwardMode.MIXED,
                dict(
                    num_tokens=3,
                    num_extends=1,
                    extend_seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
                ),
            ),
        )
        for mode, extras in cases:
            with self.subTest(mode=mode):
                backend = _backend(flat=True, speculative_tokens=1)
                backend.init_cuda_graph_state(
                    2,
                    paged_cache_group_specs=specs,
                    paged_cache_group_page_counts=_page_counts(specs),
                )
                backend.init_forward_metadata(
                    bs=2,
                    req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                    seq_lens=torch.tensor([2, 2], dtype=torch.int32),
                    forward_mode=mode,
                    req_to_page=torch.full((2, 4), 999, dtype=torch.int32),
                    flat_block_tables=tables,
                    **extras,
                )
                metadata = backend.forward_metadata
                assert metadata is not None
                self.assertEqual(
                    tuple(metadata.cache.paged_cache_block_tables), tuple(tables)
                )
                self.assertEqual(
                    metadata.cache.paged_cache_block_table_base_offsets,
                    {},
                )
                self.assertIs(
                    metadata.cache.swa_block_table,
                    metadata.cache.paged_cache_block_tables["v4.swa_kv"],
                )
                self.assertIs(
                    metadata.cache.compressor_state_block_tables[4],
                    metadata.cache.paged_cache_block_tables["v4.c4a.compressor_state"],
                )
                self.assertIs(
                    metadata.cache.compressor_state_block_tables[128],
                    metadata.cache.paged_cache_block_tables[
                        "v4.c128a.compressor_state"
                    ],
                )
                self.assertIs(
                    metadata.cache.indexer_state_block_table,
                    metadata.cache.paged_cache_block_tables[
                        "v4.c4a.indexer_compressor_state"
                    ],
                )

    def test_flat_absolute_swa_crosses_pages_without_radix_base_offsets(self):
        absolute = _group_slot_mapping_from_raw(
            positions=torch.tensor([63, 64, 127, 128], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            block_table=torch.tensor([[10, 11, 0]], dtype=torch.int32),
            rows_per_page=64,
            base_offsets=None,
        )
        self.assertTrue(torch.equal(absolute, torch.tensor([703, 704, 767, 0])))
        compressed = _group_slot_mapping_from_raw(
            positions=torch.tensor([255, 256, 511, 512], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            block_table=torch.tensor([[10, 11, 0]], dtype=torch.int32),
            rows_per_page=64,
            entry_stride_tokens=4,
            base_offsets=None,
        )
        self.assertTrue(torch.equal(compressed, torch.tensor([703, 704, 767, 0])))
        compact = _group_slot_mapping_from_raw(
            positions=torch.tensor([128, 129], dtype=torch.int64),
            req_indices=torch.tensor([0, 0], dtype=torch.int32),
            block_table=torch.tensor([[10, 11]], dtype=torch.int32),
            rows_per_page=64,
            base_offsets=torch.tensor([2], dtype=torch.int32),
        )
        self.assertTrue(torch.equal(compact, torch.tensor([640, 641])))

    def test_flat_payload_validation_fails_closed(self):
        specs = _target_specs()
        backend = _backend(flat=True)
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=specs,
            paged_cache_group_page_counts=_page_counts(specs),
        )
        valid = _tables(specs)
        common = dict(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.zeros((2, 4), dtype=torch.int32),
        )

        malformed = []
        missing = OrderedDict(valid)
        missing.pop(next(iter(missing)))
        malformed.append((missing, "group mismatch"))
        extra = OrderedDict(valid)
        extra["v4.unknown"] = torch.zeros((2, 1), dtype=torch.int32)
        malformed.append((extra, "group mismatch"))
        reordered = OrderedDict(reversed(tuple(valid.items())))
        malformed.append((reordered, "wrong order"))
        wrong_rows = OrderedDict(valid)
        wrong_rows[next(iter(wrong_rows))] = torch.zeros((1, 2), dtype=torch.int32)
        malformed.append((wrong_rows, "rows"))
        wrong_rank = OrderedDict(valid)
        wrong_rank[next(iter(wrong_rank))] = torch.zeros(2, dtype=torch.int32)
        malformed.append((wrong_rank, "rank 2"))
        wrong_width = OrderedDict(valid)
        wrong_width[next(iter(wrong_width))] = torch.zeros((2, 0), dtype=torch.int32)
        malformed.append((wrong_width, "zero width"))
        wrong_dtype = OrderedDict(valid)
        wrong_dtype[next(iter(wrong_dtype))] = torch.zeros((2, 2), dtype=torch.int64)
        malformed.append((wrong_dtype, "torch.int32"))
        wrong_device = OrderedDict(valid)
        wrong_device[next(iter(wrong_device))] = torch.zeros(
            (2, 2), dtype=torch.int32, device="meta"
        )
        malformed.append((wrong_device, "expected cpu"))

        for tables, error in malformed:
            with self.subTest(error=error):
                with self.assertRaisesRegex(RuntimeError, error):
                    backend.init_forward_metadata(
                        **common,
                        flat_block_tables=tables,
                    )

        out_of_capacity = OrderedDict(valid)
        out_of_capacity[next(iter(out_of_capacity))] = torch.tensor(
            [[4096, -1], [1, -1]], dtype=torch.int32
        )
        with self.assertRaisesRegex(RuntimeError, "outside -1..4095"):
            backend.init_forward_metadata(
                **common,
                flat_block_tables=out_of_capacity,
            )

        too_short = OrderedDict(valid)
        too_short[next(iter(too_short))] = torch.ones((2, 1), dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "missing a real page"):
            backend.init_forward_metadata(
                **{**common, "seq_lens": torch.tensor([65, 2], dtype=torch.int32)},
                flat_block_tables=too_short,
            )

        null_current_page = OrderedDict(valid)
        null_current_page[next(iter(null_current_page))] = torch.tensor(
            [[1, 0], [2, 3]], dtype=torch.int32
        )
        with self.assertRaisesRegex(RuntimeError, "missing a real page"):
            backend.init_forward_metadata(
                **{**common, "seq_lens": torch.tensor([65, 2], dtype=torch.int32)},
                flat_block_tables=null_current_page,
            )

        with self.assertRaisesRegex(RuntimeError, "radix paged-cache"):
            backend.init_forward_metadata(
                **common,
                flat_block_tables=valid,
                paged_cache_block_tables=valid,
            )
        with self.assertRaisesRegex(RuntimeError, "base offsets"):
            backend.init_forward_metadata(
                **common,
                flat_block_tables=valid,
                paged_cache_block_table_base_offsets={
                    "v4.swa_kv": torch.zeros(2, dtype=torch.int32)
                },
            )

    def test_duplicate_specs_and_capture_ids_fail_closed(self):
        spec = _draft_specs()[0]
        backend = _backend(flat=True, is_draft=True)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            backend.init_cuda_graph_state(
                2,
                paged_cache_group_specs=(spec, spec),
                paged_cache_group_page_counts=_page_counts((spec,)),
            )

        backend = _backend(flat=True, is_draft=True)
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=(spec,),
            paged_cache_group_page_counts=_page_counts((spec,)),
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            backend.init_forward_metadata_capture_cuda_graph(
                bs=2,
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens=torch.ones(2, dtype=torch.int32),
                forward_mode=ForwardMode.DECODE,
                flat_cache_group_ids=(spec.group_id, spec.group_id),
            )

    def test_idle_zero_batch_and_padded_replay_are_safe(self):
        specs = _draft_specs()
        group_ids = tuple(spec.group_id for spec in specs)
        backend = _backend(flat=True, is_draft=True, speculative_tokens=1)
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=specs,
            paged_cache_group_page_counts=_page_counts(specs),
            max_tokens_per_req=1,
        )
        backend.init_forward_metadata(
            bs=0,
            req_pool_indices=torch.empty(0, dtype=torch.int32),
            seq_lens=torch.empty(0, dtype=torch.int32),
            forward_mode=ForwardMode.IDLE,
            req_to_page=torch.zeros((1, 2), dtype=torch.int32),
        )
        assert backend.forward_metadata is not None
        self.assertEqual(backend.forward_metadata.seq_lens.numel(), 0)

        common = dict(
            bs=2,
            req_pool_indices=torch.tensor([0, 0], dtype=torch.int32),
            seq_lens=torch.ones(2, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )
        backend.init_forward_metadata_capture_cuda_graph(
            **common,
            flat_cache_group_ids=group_ids,
        )
        idle_tables = OrderedDict(
            ((group_ids[0], torch.tensor([[1], [0]], dtype=torch.int32)),)
        )
        backend.init_forward_metadata_replay_cuda_graph(
            **common,
            actual_bs=0,
            req_to_page=torch.zeros((1, 64), dtype=torch.int32),
            flat_block_tables=idle_tables,
        )
        assert backend.forward_metadata is not None
        self.assertFalse(bool(backend.forward_metadata.is_valid_token.any()))

        backend.init_forward_metadata_replay_cuda_graph(
            **common,
            actual_bs=1,
            req_to_page=torch.zeros((1, 64), dtype=torch.int32),
            flat_block_tables=idle_tables,
        )
        self.assertEqual(
            backend.forward_metadata.is_valid_token.tolist(),
            [True, False],
        )
        for invalid_actual_bs in (-1, 3):
            with self.subTest(actual_bs=invalid_actual_bs):
                with self.assertRaisesRegex(RuntimeError, "within 0..2"):
                    backend.init_forward_metadata_replay_cuda_graph(
                        **common,
                        actual_bs=invalid_actual_bs,
                        req_to_page=torch.zeros((1, 64), dtype=torch.int32),
                        flat_block_tables=idle_tables,
                    )

    def test_target_and_mtp_route_same_identity_including_state(self):
        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.draft_attn_backend = SimpleNamespace(
            uses_flat_cache_groups=True,
            flat_cache_consumer_families=frozenset({"history", "state"}),
        )
        wrapper.draft_token_to_kv_pool = SimpleNamespace(
            paged_cache_group_specs=_draft_specs()
        )
        wrapper.token_to_kv_pool = SimpleNamespace(
            paged_cache_group_specs=_target_specs()
        )
        self.assertEqual(wrapper._draft_flat_group_ids(), ("v4.swa_kv",))

        target_tables = _tables(_target_specs())
        draft_tables = wrapper._draft_flat_tables(target_tables)
        assert draft_tables is not None
        self.assertEqual(tuple(draft_tables), ("v4.swa_kv",))
        self.assertIs(draft_tables["v4.swa_kv"], target_tables["v4.swa_kv"])
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            wrapper._draft_flat_tables(
                {
                    key: value
                    for key, value in target_tables.items()
                    if key != "v4.swa_kv"
                }
            )

    def test_target_and_mtp_share_flat_page_id_geometry(self):
        target_specs = _target_specs()
        draft_specs = _draft_specs()
        target = SimpleNamespace(
            paged_cache_group_specs=target_specs,
            paged_cache_group_page_counts={spec.group_id: 17 for spec in target_specs},
        )
        draft = SimpleNamespace(
            paged_cache_group_specs=draft_specs,
            paged_cache_group_page_counts={"v4.swa_kv": 17},
        )
        _validate_shared_flat_group_geometry(target, draft)

        draft.paged_cache_group_page_counts = {"v4.swa_kv": 16}
        with self.assertRaisesRegex(RuntimeError, "page-id capacity"):
            _validate_shared_flat_group_geometry(target, draft)

    def test_flat_graph_refreshes_every_group_for_ten_replays(self):
        specs = _target_specs()
        group_ids = tuple(spec.group_id for spec in specs)
        backend = _backend(flat=True, context_len=4096, speculative_tokens=4)
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=specs,
            paged_cache_group_page_counts=_page_counts(specs),
            max_tokens_per_req=4,
        )
        common = dict(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([64, 128], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            num_tokens=8,
        )
        backend.init_forward_metadata_capture_cuda_graph(
            **common,
            flat_cache_group_ids=group_ids,
        )
        for step in range(10):
            live = OrderedDict(
                (
                    group_id,
                    torch.full(
                        (
                            2,
                            backend._cuda_graph_paged_cache_block_tables[
                                group_id
                            ].shape[1],
                        ),
                        step + index + 1,
                        dtype=torch.int32,
                    ),
                )
                for index, group_id in enumerate(group_ids)
            )
            backend.init_forward_metadata_replay_cuda_graph(
                **common,
                actual_bs=1,
                req_to_page=torch.zeros((2, 128), dtype=torch.int32),
                flat_block_tables=live,
            )
            metadata = backend.forward_metadata
            assert metadata is not None
            for group_id, table in live.items():
                self.assertTrue(
                    torch.equal(
                        metadata.cache.paged_cache_block_tables[group_id][
                            :, : table.shape[1]
                        ],
                        table,
                    )
                )
        with self.assertRaisesRegex(RuntimeError, "missing live"):
            backend.init_forward_metadata_replay_cuda_graph(
                **common,
                req_to_page=torch.zeros((2, 128), dtype=torch.int32),
            )

    def test_flat_graph_width_is_absolute_but_radix_stays_compact(self):
        spec = _draft_specs()[0]
        flat = _backend(flat=True, is_draft=True, context_len=4096)
        flat.init_cuda_graph_state(
            1,
            paged_cache_group_specs=(spec,),
            paged_cache_group_page_counts=_page_counts((spec,)),
            max_tokens_per_req=4,
        )
        self.assertEqual(
            flat._cuda_graph_paged_cache_block_tables[spec.group_id].shape,
            (1, 65),
        )
        self.assertEqual(flat._cuda_graph_paged_cache_base_offsets, {})

        radix = _backend(flat=False, is_draft=True, context_len=4096)
        radix.init_cuda_graph_state(
            1,
            paged_cache_group_specs=(spec,),
            max_tokens_per_req=4,
        )
        self.assertEqual(
            radix._cuda_graph_paged_cache_block_tables[spec.group_id].shape,
            (1, 4),
        )
        self.assertEqual(
            tuple(radix._cuda_graph_paged_cache_base_offsets),
            (spec.group_id,),
        )

    def test_radix_capture_and_replay_keep_base_offsets(self):
        spec = _draft_specs()[0]
        backend = _backend(flat=False, is_draft=True, speculative_tokens=1)
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=(spec,),
            paged_cache_group_page_counts=_page_counts((spec,)),
            max_tokens_per_req=1,
        )
        table = {spec.group_id: torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)}
        offsets = {spec.group_id: torch.tensor([2, 3], dtype=torch.int32)}
        common = dict(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([200, 80], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )
        backend.init_forward_metadata_capture_cuda_graph(
            **common,
            paged_cache_block_tables=table,
            paged_cache_block_table_base_offsets=offsets,
        )
        backend.init_forward_metadata_replay_cuda_graph(
            **common,
            req_to_page=torch.zeros((2, 64), dtype=torch.int32),
            paged_cache_block_tables=table,
            paged_cache_block_table_base_offsets=offsets,
        )
        metadata = backend.forward_metadata
        assert metadata is not None
        self.assertTrue(
            torch.equal(metadata.cache.swa_block_table[:, :2], table[spec.group_id])
        )
        self.assertTrue(
            torch.equal(metadata.cache.swa_base_logical_page, offsets[spec.group_id])
        )

    def test_startup_gate_accepts_completed_v4_flat_backend(self):
        specs = _target_specs()
        backend = _backend(flat=True)
        validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=specs,
            attn_backend=backend,
            kv_pool=SimpleNamespace(),
            speculative_algorithm="MTP",
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DeepseekV4FlatCudaGraphTest(unittest.TestCase):
    def test_target_mtp_graph_replay_refresh_and_memory_stability(self):
        device = "cuda"
        target_specs = _target_specs()
        draft_specs = _draft_specs()
        target_ids = tuple(spec.group_id for spec in target_specs)
        draft_ids = tuple(spec.group_id for spec in draft_specs)
        target = _backend(
            flat=True,
            device=device,
            context_len=128,
            speculative_tokens=4,
        )
        draft = _backend(
            flat=True,
            device=device,
            is_draft=True,
            context_len=128,
            speculative_tokens=4,
        )
        target.init_cuda_graph_state(
            2,
            paged_cache_group_specs=target_specs,
            paged_cache_group_page_counts=_page_counts(target_specs),
            max_tokens_per_req=4,
        )
        draft.init_cuda_graph_state(
            2,
            paged_cache_group_specs=draft_specs,
            paged_cache_group_page_counts=_page_counts(draft_specs),
            max_tokens_per_req=4,
        )
        common = dict(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
            seq_lens=torch.tensor([64, 1], dtype=torch.int32, device=device),
            forward_mode=ForwardMode.DECODE,
            num_tokens=8,
        )
        target.init_forward_metadata_capture_cuda_graph(
            **common,
            flat_cache_group_ids=target_ids,
        )
        draft.init_forward_metadata_capture_cuda_graph(
            **common,
            flat_cache_group_ids=draft_ids,
        )
        target_metadata = target.forward_metadata
        draft_metadata = draft.forward_metadata
        assert target_metadata is not None
        assert draft_metadata is not None

        observed = torch.empty(len(target_ids) + 1, dtype=torch.int32, device=device)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for index, group_id in enumerate(target_ids):
                observed[index].copy_(
                    target_metadata.cache.paged_cache_block_tables[group_id][0, 0]
                )
            observed[-1].copy_(draft_metadata.cache.swa_block_table[0, 0])

        live = OrderedDict(
            (
                group_id,
                torch.zeros(
                    (
                        2,
                        target._cuda_graph_paged_cache_block_tables[group_id].shape[1],
                    ),
                    dtype=torch.int32,
                    device=device,
                ),
            )
            for group_id in target_ids
        )
        req_to_page = torch.zeros((2, 8), dtype=torch.int32, device=device)

        def replay(step: int) -> None:
            for index, table in enumerate(live.values()):
                table[0].fill_(step + index + 1)
                table[1].zero_()
            target.init_forward_metadata_replay_cuda_graph(
                **common,
                actual_bs=1,
                req_to_page=req_to_page,
                flat_block_tables=live,
            )
            draft.init_forward_metadata_replay_cuda_graph(
                **common,
                actual_bs=1,
                req_to_page=req_to_page,
                flat_block_tables=OrderedDict((("v4.swa_kv", live["v4.swa_kv"]),)),
            )
            graph.replay()

        for step in range(10):
            replay(step)
            torch.cuda.synchronize()
            expected = [step + index + 1 for index in range(len(target_ids))]
            expected.append(step + 1)
            self.assertEqual(observed.cpu().tolist(), expected)
        self.assertEqual(
            target.forward_metadata.is_valid_token.tolist(),
            [True, True, True, True, False, False, False, False],
        )
        self.assertEqual(
            draft.forward_metadata.is_valid_token.tolist(),
            [True, True, True, True, False, False, False, False],
        )

        torch.cuda.synchronize()
        reserved_before = torch.cuda.memory_reserved()
        for step in range(50):
            replay(step + 100)
        torch.cuda.synchronize()
        reserved_after = torch.cuda.memory_reserved()
        self.assertLessEqual(reserved_after, reserved_before)


if __name__ == "__main__":
    unittest.main()
