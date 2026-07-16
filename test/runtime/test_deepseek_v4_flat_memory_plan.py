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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


with mock.patch.dict(sys.modules):
    _contract = _load(
        "tokenspeed.runtime.configs.flat_kv_contract", "flat_kv_contract.py"
    )
    _generic = _load(
        "tokenspeed.runtime.configs.paged_cache_spec", "paged_cache_spec.py"
    )
    _v4 = _load("deepseek_v4_cache_spec_flat_plan_test", "deepseek_v4_cache_spec.py")
    _plan = _load("flat_memory_plan_v4_test", "flat_memory_plan.py")

FlatComponentTensorPlan = _plan.FlatComponentTensorPlan
FlatGroupTablePlan = _plan.FlatGroupTablePlan
V4FlatMetadataAccounting = _plan.V4FlatMetadataAccounting
V4FlatPlanAgreementRecord = _plan.V4FlatPlanAgreementRecord
assert_v4_flat_plan_agreement = _plan.assert_v4_flat_plan_agreement
build_v4_cache_specs = _v4.build_v4_cache_specs
build_v4_flat_memory_plan = _plan.build_v4_flat_memory_plan
canonical_v4_flat_memory_plan = _plan.canonical_v4_flat_memory_plan
make_v4_flat_plan_agreement_record = _plan.make_v4_flat_plan_agreement_record
validate_v4_flat_graph_owner_allocation = _plan.validate_v4_flat_graph_owner_allocation


def _by_group(specs):
    return {spec.group_id: spec for spec in specs}


def _component(
    owner: str,
    group_id: str,
    component: str,
    bytes_per_block: int,
    *,
    dtype: str = "uint8",
):
    return FlatComponentTensorPlan(
        owner=owner,
        group_id=group_id,
        layer=0,
        component=component,
        dtype=dtype,
        shape_per_block=(bytes_per_block,),
        stride_bytes=(1,),
        alignment_bytes=16,
        bytes_per_block=bytes_per_block,
    )


class TestDeepSeekV4FlatCacheSpecs(unittest.TestCase):
    def test_six_storage_classes_have_exact_raw_spans_and_roles(self):
        ordered_specs = build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(1, 4, 128),
        )
        self.assertEqual(
            [spec.block_size_tokens for spec in ordered_specs],
            [64, 4, 256, 8, 256, 4],
        )
        specs = _by_group(ordered_specs)

        expected = {
            "v4.swa_kv": ("v4.swa", 64, "continuation_state", "bounded_window"),
            "v4.c4a.compressor_state": (
                "v4.c4.state",
                4,
                "continuation_state",
                "bounded_window",
            ),
            "v4.c4a.compressed_kv": (
                "v4.c4.history",
                256,
                "history_anchor",
                "absolute",
            ),
            "v4.c128a.compressor_state": (
                "v4.c128.state",
                8,
                "continuation_state",
                "bounded_window",
            ),
            "v4.c128a.compressed_kv": (
                "v4.c128.history",
                256,
                "history_anchor",
                "absolute",
            ),
            "v4.c4a.indexer_compressor_state": (
                "v4.index.state",
                4,
                "continuation_state",
                "bounded_window",
            ),
        }
        self.assertEqual(set(specs), set(expected))
        for group_id, expected_fields in expected.items():
            with self.subTest(group_id=group_id):
                pool_id, raw_span, prefix_role, table_layout = expected_fields
                spec = specs[group_id]
                self.assertEqual(spec.pool_id, pool_id)
                self.assertEqual(spec.block_size_tokens, raw_span)
                self.assertEqual(spec.prefix_role, prefix_role)
                self.assertEqual(spec.table_layout, table_layout)
                self.assertEqual(spec.owner_mask, _v4.CACHE_OWNER_TARGET)

    def test_c4_history_requires_main_and_indexer_producers(self):
        target = _by_group(
            build_v4_cache_specs(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4, 128),
            )
        )
        draft = _by_group(
            build_v4_cache_specs(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4, 128),
                owner_mask=_v4.CACHE_OWNER_DRAFT,
            )
        )

        self.assertEqual(
            target["v4.c4a.compressed_kv"].required_producer_domain_mask,
            _v4.V4_PRODUCER_TARGET_MAIN | _v4.V4_PRODUCER_TARGET_INDEXER,
        )
        self.assertEqual(
            target["v4.c4a.compressor_state"].required_producer_domain_mask,
            _v4.V4_PRODUCER_TARGET_MAIN,
        )
        self.assertEqual(
            target["v4.c4a.indexer_compressor_state"].required_producer_domain_mask,
            _v4.V4_PRODUCER_TARGET_INDEXER,
        )
        self.assertEqual(
            draft["v4.c4a.compressed_kv"].required_producer_domain_mask,
            _v4.V4_PRODUCER_DRAFT_MAIN | _v4.V4_PRODUCER_DRAFT_INDEXER,
        )


class TestV4FlatMemoryPlanUnion(unittest.TestCase):
    def setUp(self):
        target_all = _by_group(
            build_v4_cache_specs(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4),
            )
        )
        draft_all = _by_group(
            build_v4_cache_specs(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4),
                owner_mask=_v4.CACHE_OWNER_DRAFT,
            )
        )
        self.target_swa = target_all["v4.swa_kv"]
        self.draft_swa = draft_all["v4.swa_kv"]
        self.draft_only = draft_all["v4.c4a.compressor_state"]
        self.target_component = _component(
            "target", self.target_swa.group_id, "kv", 100
        )
        self.draft_component = _component("draft", self.draft_swa.group_id, "kv", 40)
        self.draft_only_component = _component(
            "draft", self.draft_only.group_id, "compressor_state", 30
        )

    def _build(self, *, reverse: bool = False, metadata_accounting=None):
        target_specs = [self.target_swa]
        draft_specs = [self.draft_swa, self.draft_only]
        target_components = [self.target_component]
        draft_components = [self.draft_component, self.draft_only_component]
        target_counts = {self.target_swa.group_id: 10}
        draft_counts = {
            self.draft_swa.group_id: 13,
            self.draft_only.group_id: 7,
        }
        if reverse:
            target_specs.reverse()
            draft_specs.reverse()
            target_components.reverse()
            draft_components.reverse()
            target_counts = dict(reversed(tuple(target_counts.items())))
            draft_counts = dict(reversed(tuple(draft_counts.items())))
        return build_v4_flat_memory_plan(
            target_group_specs=target_specs,
            target_group_page_counts=target_counts,
            target_components=target_components,
            draft_group_specs=draft_specs,
            draft_group_page_counts=draft_counts,
            draft_components=draft_components,
            metadata_accounting=metadata_accounting,
        )

    def _metadata_accounting(self):
        table_plans = (
            FlatGroupTablePlan(
                group_id=self.target_swa.group_id,
                target_capture_cols=3,
                draft_capture_cols=2,
                max_export_cols=5,
                max_live_descriptor_cols=5,
            ),
            FlatGroupTablePlan(
                group_id=self.draft_only.group_id,
                target_capture_cols=0,
                draft_capture_cols=4,
                max_export_cols=6,
                max_live_descriptor_cols=6,
            ),
        )
        target_bytes = 4 * 2 * (3 + 1)
        draft_bytes = 4 * 5 * ((2 + 1) + (4 + 1))
        return V4FlatMetadataAccounting(
            group_table_plans=table_plans,
            graph_metadata_bytes=target_bytes + draft_bytes,
            target_graph_batch_rows=2,
            draft_graph_batch_rows=5,
            max_scheduled_batch_rows=5,
        )

    def test_union_retains_draft_only_and_uses_max_capacity_plus_owner_bytes(self):
        plan = self._build()
        groups = _by_group(plan.scheduler_group_specs)
        pools = {pool.pool_id: pool for pool in plan.pools}

        self.assertEqual(
            set(groups),
            {self.target_swa.group_id, self.draft_only.group_id},
        )
        self.assertEqual(
            groups[self.target_swa.group_id].owner_mask,
            _v4.CACHE_OWNER_TARGET | _v4.CACHE_OWNER_DRAFT,
        )
        self.assertEqual(
            groups[self.target_swa.group_id].required_producer_domain_mask,
            _v4.V4_PRODUCER_TARGET_MAIN | _v4.V4_PRODUCER_DRAFT_MAIN,
        )
        self.assertEqual(
            groups[self.draft_only.group_id].owner_mask,
            _v4.CACHE_OWNER_DRAFT,
        )
        self.assertEqual(
            {spec.group_id for spec in plan.target_owner_group_specs},
            {self.target_swa.group_id},
        )
        self.assertEqual(
            {spec.group_id for spec in plan.draft_owner_group_specs},
            {self.draft_swa.group_id, self.draft_only.group_id},
        )

        shared_pool = pools["v4.swa"]
        self.assertEqual(shared_pool.total_blocks, 13)
        self.assertEqual(shared_pool.bytes_per_block, 140)
        self.assertEqual(len(shared_pool.tensors), 2)
        self.assertEqual(pools["v4.c4.state"].total_blocks, 7)
        self.assertEqual(pools["v4.c4.state"].bytes_per_block, 30)
        self.assertEqual(plan.payload_bytes, 13 * 140 + 7 * 30)
        self.assertEqual(plan.device_cache_total_bytes, plan.payload_bytes)
        self.assertEqual(plan.graph_metadata_bytes, 0)
        self.assertEqual(plan.forward_input_bytes, 0)
        self.assertEqual(plan.cpu_cache_metadata_total_bytes, 0)

    def test_same_group_window_mismatch_fails_fast(self):
        bad_draft = replace(
            self.draft_swa,
            sliding_window_tokens=self.draft_swa.sliding_window_tokens + 1,
        )
        with self.assertRaisesRegex(ValueError, "v4.swa_kv.*sliding_window_tokens"):
            build_v4_flat_memory_plan(
                target_group_specs=[self.target_swa],
                target_group_page_counts={self.target_swa.group_id: 10},
                target_components=[self.target_component],
                draft_group_specs=[bad_draft],
                draft_group_page_counts={bad_draft.group_id: 10},
                draft_components=[self.draft_component],
            )

    def test_graph_metadata_owner_shapes_and_runtime_bytes_are_exact(self):
        plan = self._build(metadata_accounting=self._metadata_accounting())

        self.assertEqual(
            plan.graph_capture_cols_by_group("target"),
            {self.target_swa.group_id: 3},
        )
        self.assertEqual(
            plan.graph_capture_cols_by_group("draft"),
            {self.target_swa.group_id: 2, self.draft_only.group_id: 4},
        )
        self.assertEqual(plan.graph_metadata_bytes_for_owner("target"), 32)
        self.assertEqual(plan.graph_metadata_bytes_for_owner("draft"), 160)
        plan.validate_graph_metadata_allocation(
            target_actual_bytes=32,
            draft_actual_bytes=160,
        )

        target_actual = validate_v4_flat_graph_owner_allocation(
            owner="target",
            capture_cols_by_group=plan.graph_capture_cols_by_group("target"),
            batch_rows=plan.graph_batch_rows("target"),
            table_shapes={self.target_swa.group_id: (2, 3)},
            base_shapes={self.target_swa.group_id: (2,)},
            table_nbytes={self.target_swa.group_id: 24},
            base_nbytes={self.target_swa.group_id: 8},
        )
        self.assertEqual(target_actual, 32)

    def test_graph_metadata_runtime_shape_and_total_drift_fail_closed(self):
        plan = self._build(metadata_accounting=self._metadata_accounting())

        def validate_owner(*, table_shape, table_nbytes):
            return validate_v4_flat_graph_owner_allocation(
                owner="target",
                capture_cols_by_group=plan.graph_capture_cols_by_group("target"),
                batch_rows=plan.graph_batch_rows("target"),
                table_shapes={self.target_swa.group_id: table_shape},
                base_shapes={self.target_swa.group_id: (2,)},
                table_nbytes={self.target_swa.group_id: table_nbytes},
                base_nbytes={self.target_swa.group_id: 8},
            )

        cases = (
            (
                "table shape drift",
                lambda: validate_owner(table_shape=(2, 2), table_nbytes=16),
                "shape disagrees",
            ),
            (
                "table byte drift",
                lambda: validate_owner(table_shape=(2, 3), table_nbytes=12),
                "bytes disagree",
            ),
            (
                "owner total drift",
                lambda: plan.validate_graph_metadata_allocation(
                    target_actual_bytes=28,
                    draft_actual_bytes=160,
                ),
                "target.*allocation",
            ),
        )
        for name, operation, error in cases:
            with self.subTest(name), self.assertRaisesRegex(RuntimeError, error):
                operation()

    def test_eager_only_zero_graph_rows_allocate_zero_flat_metadata(self):
        group_id = self.target_swa.group_id

        actual = validate_v4_flat_graph_owner_allocation(
            owner="target",
            capture_cols_by_group={group_id: 3},
            batch_rows=0,
            table_shapes={group_id: (0, 3)},
            base_shapes={group_id: (0,)},
            table_nbytes={group_id: 0},
            base_nbytes={group_id: 0},
        )
        self.assertEqual(actual, 0)

    def test_fingerprint_and_canonical_order_ignore_input_order(self):
        forward = self._build()
        reverse = self._build(reverse=True)

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.plan_fingerprint, reverse.plan_fingerprint)
        self.assertEqual(
            tuple(pool.pool_id for pool in forward.pools),
            tuple(sorted(pool.pool_id for pool in forward.pools)),
        )

    def test_canonical_agreement_record_is_covered_by_plan_fingerprint(self):
        plan = self._build()
        record = make_v4_flat_plan_agreement_record(plan, rank=2)

        self.assertEqual(
            _plan._canonical_hash(record.canonical_plan),
            plan.plan_fingerprint,
        )
        self.assertEqual(record.canonical_plan, canonical_v4_flat_memory_plan(plan))
        assert_v4_flat_plan_agreement(
            [record, make_v4_flat_plan_agreement_record(plan, rank=3)]
        )

    def test_agreement_reports_first_divergent_canonical_field_and_rank(self):
        plan = self._build()
        reference = make_v4_flat_plan_agreement_record(plan, rank=4)
        divergent_plan = copy.deepcopy(reference.canonical_plan)
        divergent_plan["pools"][0]["total_blocks"] += 1
        divergent = V4FlatPlanAgreementRecord(
            rank=7,
            plan_fingerprint=_plan._canonical_hash(divergent_plan),
            canonical_plan=divergent_plan,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"rank 7 diverges from rank 4.*pools\[0\]\.total_blocks",
        ):
            assert_v4_flat_plan_agreement([reference, divergent])

    def test_duplicate_component_identity_rejects_schema_drift(self):
        drift = replace(self.target_component, dtype="float8_e4m3fn")
        with self.assertRaisesRegex(ValueError, "duplicate component identity"):
            build_v4_flat_memory_plan(
                target_group_specs=[self.target_swa],
                target_group_page_counts={self.target_swa.group_id: 10},
                target_components=[self.target_component, drift],
            )

    def test_table_plan_owner_membership_mismatch_fails_fast(self):
        metadata = V4FlatMetadataAccounting(
            group_table_plans=(
                FlatGroupTablePlan(
                    group_id=self.target_swa.group_id,
                    target_capture_cols=0,
                    draft_capture_cols=2,
                    max_export_cols=2,
                    max_live_descriptor_cols=2,
                ),
                FlatGroupTablePlan(
                    group_id=self.draft_only.group_id,
                    target_capture_cols=1,
                    draft_capture_cols=1,
                    max_export_cols=2,
                    max_live_descriptor_cols=2,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "capture owner membership"):
            build_v4_flat_memory_plan(
                target_group_specs=[self.target_swa],
                target_group_page_counts={self.target_swa.group_id: 10},
                target_components=[self.target_component],
                draft_group_specs=[self.draft_swa, self.draft_only],
                draft_group_page_counts={
                    self.draft_swa.group_id: 13,
                    self.draft_only.group_id: 7,
                },
                draft_components=[
                    self.draft_component,
                    self.draft_only_component,
                ],
                metadata_accounting=metadata,
            )

    def test_graph_rows_cannot_exceed_scheduled_rows(self):
        with self.assertRaisesRegex(ValueError, "graph batch rows"):
            V4FlatMetadataAccounting(
                group_table_plans=(
                    FlatGroupTablePlan(
                        group_id=self.target_swa.group_id,
                        target_capture_cols=1,
                        draft_capture_cols=0,
                        max_export_cols=1,
                        max_live_descriptor_cols=1,
                    ),
                ),
                target_graph_batch_rows=2,
                max_scheduled_batch_rows=1,
            )


if __name__ == "__main__":
    unittest.main()
