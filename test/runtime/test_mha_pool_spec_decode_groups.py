"""KV pool paged-cache group publication vs ext build flavor.

Rule under test (kv_cache/mha.py, kv_cache/mla.py): a pool publishes
paged_cache_group_specs iff the tokenspeed_scheduler ext is flat-built
(TOKENSPEED_FLAT_KVCACHE); radix builds publish nothing. Speculative
decoding does not gate publication (flat+spec is supported); backend
capability is checked separately by validate_flat_scheduler_config.

Two pool families share that rule: MHA publishes one group per distinct
(retention, window); MLA and its DSA subclass are uniform full-history, so
always exactly ONE "full_attention" group -- which is what makes them
servable on a flat ext at all (its req_to_page fallback samples group 0).

The installed ext's real build flavor must not decide these tests, so the
scheduler_ext_flat_kvcache probe is patched per case; the probe's own
default-False behavior is covered separately.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

GPT_OSS_LAYER_TYPES = (
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
)

_FLAT_PROBE = "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"


class MHAPoolGroupPublicationTest(unittest.TestCase):
    """Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self, *, flat_ext: bool = True, **overrides):
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=16,
            rank=0,
            enable_alt_stream=False,
        )
        kwargs.update(overrides)
        # The pool resolves the probe lazily at construction time; patching
        # the module attribute pins the ext flavor regardless of the install.
        with mock.patch(_FLAT_PROBE, return_value=flat_ext):
            return self.MHATokenToKVPool(**kwargs)

    def test_plain_no_spec_publishes_single_full_group(self):
        # The flat scheduler allocates pages only through configured groups,
        # so plain models must keep the single full-history group published.
        pool = self._pool()
        self.assertEqual(len(pool.paged_cache_group_specs), 1)
        spec = pool.paged_cache_group_specs[0]
        self.assertEqual(spec.group_id, "full_attention")
        self.assertEqual(spec.retention, "full_history")
        self.assertIn("full_attention", pool.paged_cache_group_page_counts)

    def test_hybrid_no_spec_publishes_two_groups(self):
        # layer_num must match len(layer_types): the M12 slab layout's
        # pairing-completeness assert cross-checks them.
        pool = self._pool(
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
            layer_num=len(GPT_OSS_LAYER_TYPES),
        )
        self.assertEqual(
            {s.group_id for s in pool.paged_cache_group_specs},
            {"full_attention", "sliding_attention"},
        )
        self.assertEqual(
            set(pool.paged_cache_group_page_counts),
            {"full_attention", "sliding_attention"},
        )

    def test_radix_ext_plain_publishes_no_groups(self):
        # A radix scheduler never fills flat_block_tables, so publication
        # must stay off or graph capture binds buffers that never refresh.
        pool = self._pool(flat_ext=False)
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})

    def test_radix_ext_hybrid_publishes_no_groups(self):
        pool = self._pool(
            flat_ext=False,
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
        )
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})


class MLAPoolGroupPublicationTest(unittest.TestCase):
    """Constructs a real (tiny, CPU) MLA/DSA pool; skips without deps."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mla import (
                MLATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MLATokenToKVPool = MLATokenToKVPool

    def _pool(self, *, flat_ext: bool = True, cls=None, **overrides):
        kwargs = dict(
            size=64,
            dtype=self.torch.bfloat16,
            model_dtype=self.torch.bfloat16,
            quant_method="",
            kv_lora_rank=32,
            qk_rope_head_dim=8,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=128,
            page_size=16,
            rank=0,
            enable_alt_stream=False,
            max_scheduled_tokens=32,
        )
        kwargs.update(overrides)
        with mock.patch(_FLAT_PROBE, return_value=flat_ext):
            return (cls or self.MLATokenToKVPool)(**kwargs)

    def test_flat_ext_publishes_single_full_history_group(self):
        pool = self._pool(flat_ext=True)
        self.assertEqual(len(pool.paged_cache_group_specs), 1)
        spec = pool.paged_cache_group_specs[0]
        self.assertEqual(spec.group_id, "full_attention")
        self.assertEqual(spec.retention, "full_history")
        self.assertEqual(spec.family, "history")
        # rows_per_page must be the pool's page size: the scheduler derives
        # its block_size from it, and a mismatch would mis-address every read.
        self.assertEqual(spec.rows_per_page, 16)
        self.assertEqual(spec.entry_stride_tokens, 1)
        self.assertIsNone(spec.sliding_window_tokens)
        self.assertGreater(pool.paged_cache_group_page_counts["full_attention"], 0)

    def test_radix_ext_publishes_no_groups(self):
        pool = self._pool(flat_ext=False)
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})

    def test_single_group_admits_table_blind_backend(self):
        # The exact guard that rejected every MLA model before the pool
        # started publishing. test_flat_scheduler_config_guard covers this
        # rule with fakes; here it runs against the REAL pool's real specs,
        # which is the integration point that was broken.
        from tokenspeed.runtime.configs.paged_cache_spec import (
            validate_flat_scheduler_config,
        )

        pool = self._pool(flat_ext=True)

        class TableBlindBackend:
            uses_paged_cache_groups = False
            uses_flat_cache_groups = False
            flat_spec_capable = True

        validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=pool.paged_cache_group_specs,
            attn_backend=TableBlindBackend(),
            kv_pool=pool,
            speculative_algorithm="EAGLE3",
        )

    def test_dsa_subclass_inherits_publication(self):
        # GLM-5.2 rides the same single group; its packed index-K buffer is
        # addressed by the same page*page_size+slot locations.
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.dsa import (
                DSATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs DSA kernels: {exc}")
        pool = self._pool(flat_ext=True, cls=DSATokenToKVPool, index_head_dim=128)
        self.assertEqual(len(pool.paged_cache_group_specs), 1)
        self.assertEqual(pool.paged_cache_group_specs[0].group_id, "full_attention")


class FlatHostTierAutoDisableTest(unittest.TestCase):
    """Default ServerArgs + flat ext + MLA/DSA pool must still be servable.

    KVStore has no --enable flag, so the DEFAULT K2.5 / GLM-5.2 command line
    arrives with enable_kvstore=True and startup must downgrade L2 rather than
    refuse. Uses the real default args, since the regression is exactly
    "default command line fails".
    """

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.configs.paged_cache_spec import (
                flat_host_tier_unsupported_reason,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mla import (
                MLATokenToKVPool,
            )
            from tokenspeed.runtime.utils.server_args import prepare_server_args
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs the tokenspeed runtime deps: {exc}")
        self.torch = torch
        self.reason = flat_host_tier_unsupported_reason
        self.MLATokenToKVPool = MLATokenToKVPool
        self.prepare_server_args = prepare_server_args

    def _mla_pool(self, **overrides):
        kwargs = dict(
            size=64,
            dtype=self.torch.bfloat16,
            model_dtype=self.torch.bfloat16,
            quant_method="",
            kv_lora_rank=32,
            qk_rope_head_dim=8,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=128,
            page_size=16,
            rank=0,
            enable_alt_stream=False,
        )
        kwargs.update(overrides)
        with mock.patch(_FLAT_PROBE, return_value=True):
            return self.MLATokenToKVPool(**kwargs)

    def test_default_server_args_enable_kvstore(self):
        # The premise of the auto-disable: nobody has to opt in to L2.
        args = self.prepare_server_args(["some-model"])
        self.assertFalse(args.disable_kvstore)
        self.assertTrue(args.enable_kvstore)

    def test_mla_pool_reports_host_tier_gap(self):
        reason = self.reason(self._mla_pool())
        self.assertIsNotNone(reason)
        self.assertIn("MLATokenToKVPool", reason)

    def test_default_args_plus_mla_pool_downgrades_instead_of_failing(self):
        # The startup decision EventLoop.__init__ makes, composed here from the
        # real default args and the real pool: L2 off, serving still possible.
        args = self.prepare_server_args(["some-model"])
        pool = self._mla_pool()
        if self.reason(pool) is not None:
            args.enable_kvstore = False
        self.assertFalse(args.enable_kvstore)

    def test_mha_pool_keeps_the_host_tier(self):
        # The downgrade must be narrow: MHA pools do expose k_buffer/v_buffer
        # and must keep L2 on a flat build.
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        with mock.patch(_FLAT_PROBE, return_value=True):
            pool = MHATokenToKVPool(
                size=32,
                dtype=self.torch.bfloat16,
                head_num=1,
                head_dim=8,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                max_batch_size=2,
                max_context_len=64,
                page_size=16,
                rank=0,
                enable_alt_stream=False,
            )
        self.assertIsNone(self.reason(pool))

    def test_l3_verdict_ignores_enable_kvstore(self):
        # make_config derives enable_l3_storage from --kvstore-storage-backend
        # ALONE, so the rejection must not be coupled to the L2 flag. These are
        # the two ways enable_kvstore ends up False while a storage backend is
        # still configured -- both must still be rejected.
        from tokenspeed.runtime.configs.paged_cache_spec import (
            flat_l3_storage_unsupported_reason,
        )

        for argv in (
            [
                "some-model",
                "--disable-kvstore",
                "--kvstore-storage-backend",
                "mooncake",
            ],
            [
                "some-model",
                "--disaggregation-mode",
                "decode",
                "--kvstore-storage-backend",
                "mooncake",
            ],
        ):
            args = self.prepare_server_args(argv)
            self.assertFalse(args.enable_kvstore, argv)
            # The state the scheduler would otherwise be handed: L2 off, L3 on.
            self.assertTrue(args.kvstore_storage_backend is not None, argv)
            self.assertIsNotNone(flat_l3_storage_unsupported_reason(args), argv)

    def test_l3_verdict_is_none_without_a_storage_backend(self):
        # L3 is opt-in, so the default command line (and the plain
        # --disable-kvstore case) must stay servable.
        from tokenspeed.runtime.configs.paged_cache_spec import (
            flat_l3_storage_unsupported_reason,
        )

        for argv in (["some-model"], ["some-model", "--disable-kvstore"]):
            args = self.prepare_server_args(argv)
            self.assertIsNone(args.kvstore_storage_backend, argv)
            self.assertIsNone(flat_l3_storage_unsupported_reason(args), argv)

    def test_l3_guard_is_not_nested_under_enable_kvstore(self):
        # The bug this replaces a string-ordering check for: the guard used to
        # sit inside `if ... and server_args.enable_kvstore:`. Walk the AST and
        # assert no enclosing test of the L3 raise mentions enable_kvstore.
        import ast
        import inspect
        import textwrap

        from tokenspeed.runtime.engine import event_loop as _el

        tree = ast.parse(textwrap.dedent(inspect.getsource(_el.EventLoop.__init__)))

        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }

        def enclosing_if_tests(node):
            """Tests of every `if` this statement sits in the body/orelse of."""
            tests, prev, cur = [], node, parents.get(node)
            while cur is not None:
                if isinstance(cur, ast.If) and (prev in cur.body or prev in cur.orelse):
                    tests.append(cur.test)
                prev, cur = cur, parents.get(cur)
            return tests

        def gated_by_enable_kvstore(node) -> bool:
            return any(
                isinstance(n, ast.Attribute) and n.attr == "enable_kvstore"
                for test in enclosing_if_tests(node)
                for n in ast.walk(test)
            )

        # Anchor on the helper CALL, not the message text: matching wording let
        # a reworded -- or entirely deleted -- guard pass on an empty set.
        assigns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None)
            == "flat_l3_storage_unsupported_reason"
        ]
        self.assertTrue(
            assigns, "EventLoop must call flat_l3_storage_unsupported_reason"
        )
        verdicts = {
            t.id for node in assigns for t in node.targets if isinstance(t, ast.Name)
        }
        self.assertTrue(verdicts, "the helper's result must be bound to a name")

        def positive_branch(node: ast.If):
            """Branch taken when the verdict is truthy, or None if the test
            shape is unrecognized -- fail closed rather than stop checking."""
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id in verdicts
                and len(test.ops) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
            ):
                if isinstance(test.ops[0], ast.IsNot):
                    return node.body
                if isinstance(test.ops[0], ast.Is):
                    return node.orelse
            elif isinstance(test, ast.Name) and test.id in verdicts:
                return node.body
            elif (
                isinstance(test, ast.UnaryOp)
                and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Name)
                and test.operand.id in verdicts
            ):
                return node.orelse
            return None

        # The raise must sit DIRECTLY in the verdict's truthy branch: "some
        # Raise somewhere under the if" would also accept an inverted test or
        # an unrelated nested raise.
        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(isinstance(s, ast.Raise) for s in (positive_branch(node) or []))
        ]
        self.assertTrue(
            guards,
            "the L3 verdict's truthy branch must raise, not merely be computed",
        )

        # Both halves matter: gating the ASSIGNMENT on enable_kvstore leaves the
        # verdict None whenever L2 is off, which is the original bypass wearing
        # a different shape.
        offenders = [
            ast.dump(node) for node in assigns + guards if gated_by_enable_kvstore(node)
        ]
        self.assertEqual(
            offenders,
            [],
            "neither the L3 verdict nor its raise may be gated on enable_kvstore",
        )

    def test_contract_pool_is_left_to_its_own_guard(self):
        # Kimi-K3's FlatHybridCachePool has a dedicated message; this helper
        # must not shadow it by silently downgrading instead.
        class FakeContractPool:
            runtime_contract = object()

        self.assertIsNone(self.reason(FakeContractPool()))


class MLAConfigFieldOrderTest(unittest.TestCase):
    """DSAConfig must stay constructible after MLAConfig gained a default.

    MLAConfig is a positional dataclass (only BaseAttnConfig is kw_only), so
    max_scheduled_tokens must be keyword-only -- otherwise DSAConfig's
    required index_* fields sit behind a defaulted one and the whole configs
    package raises "non-default argument follows default argument" at import.
    """

    def test_dsa_config_required_fields_follow_kw_only_default(self):
        try:
            import dataclasses

            from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
            from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        field_map = {f.name: f for f in dataclasses.fields(MLAConfig)}
        self.assertTrue(field_map["max_scheduled_tokens"].kw_only)
        dsa_fields = {f.name: f for f in dataclasses.fields(DSAConfig)}
        for name in ("index_topk", "index_head_dim", "index_n_heads"):
            self.assertIs(dsa_fields[name].default, dataclasses.MISSING)


class SchedulerExtFlatKvcacheProbeTest(unittest.TestCase):
    """scheduler_ext_flat_kvcache reads the ext's FLAT_KVCACHE build flag with
    a radix-safe default: no package or no attribute -> False."""

    def setUp(self):
        try:
            # paged_cache_spec itself is torch-free, but the configs package
            # __init__ pulls transformers-backed model configs.
            from tokenspeed.runtime.configs.paged_cache_spec import (
                scheduler_ext_flat_kvcache,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs the tokenspeed runtime deps: {exc}")

        self.probe = scheduler_ext_flat_kvcache

    def test_flat_built_ext_reports_true(self):
        fake = types.ModuleType("tokenspeed_scheduler")
        fake.FLAT_KVCACHE = True
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": fake}):
            self.assertTrue(self.probe())

    def test_radix_built_ext_reports_false(self):
        fake = types.ModuleType("tokenspeed_scheduler")
        fake.FLAT_KVCACHE = False
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": fake}):
            self.assertFalse(self.probe())

    def test_older_ext_without_attribute_defaults_false(self):
        # Pre-FLAT_KVCACHE extensions lack the attribute entirely.
        fake = types.ModuleType("tokenspeed_scheduler")
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": fake}):
            self.assertFalse(self.probe())

    def test_missing_package_defaults_false(self):
        # sys.modules[name] = None makes `import name` raise ImportError.
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": None}):
            self.assertFalse(self.probe())


if __name__ == "__main__":
    unittest.main()
