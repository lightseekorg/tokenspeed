"""Flat-scheduler config guard (validate_flat_scheduler_config, called from
engine/event_loop before the C++ Scheduler ctor).

On a flat-built ext: a radix-populate-only backend (uses_paged_cache_groups
without uses_flat_cache_groups, DeepSeek V4/MLA-style) is rejected loudly;
zero published groups is rejected with the actual cause named; a
flat-capable backend with >=1 group passes. On a radix-built ext the guard
is a no-op. The validator is pure and torch-free, so this file runs on a
bare interpreter (loaded by file path when runtime deps are unavailable).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _load_paged_cache_spec():
    try:
        from tokenspeed.runtime.configs import paged_cache_spec

        return paged_cache_spec
    except (ImportError, ModuleNotFoundError):
        # The configs package __init__ pulls transformers-backed model
        # configs; the module itself is torch-free, so load it directly.
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        path = os.path.join(
            repo_root,
            "python",
            "tokenspeed",
            "runtime",
            "configs",
            "paged_cache_spec.py",
        )
        spec = importlib.util.spec_from_file_location("_paged_cache_spec_guard", path)
        module = importlib.util.module_from_spec(spec)
        # dataclass processing resolves cls.__module__ through sys.modules, so
        # the module must be registered before exec.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_pcs = _load_paged_cache_spec()


class FakeV4StyleBackend:
    """DeepSeek V4/MLA shape: radix-populate consumer, not flat-capable."""

    uses_paged_cache_groups = True
    uses_flat_cache_groups = False


class FakeFlatMHABackend:
    """backends/mha.py shape: flat-group capable, spec-verify wired."""

    uses_paged_cache_groups = False
    uses_flat_cache_groups = True


class FakeSpecIncapableBackend:
    """Flat-capable backend whose spec-verify path is not wired
    (flat_spec_capable=False); rejected at startup under spec decode."""

    uses_paged_cache_groups = False
    uses_flat_cache_groups = True
    flat_spec_capable = False


class FakeV4Pool:
    pass


class FakeMHAPool:
    pass


class FakeMambaPool:
    pass


class FakeGroup:
    group_id = "full_attention"


class ValidateFlatSchedulerConfigTest(unittest.TestCase):
    def test_flat_ext_v4_style_backend_raises(self):
        # V4 pool publishes specs unconditionally, so groups are non-empty;
        # the backend flags alone must trip the guard.
        with self.assertRaises(RuntimeError) as ctx:
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[FakeGroup()],
                attn_backend=FakeV4StyleBackend(),
                kv_pool=FakeV4Pool(),
                speculative_algorithm=None,
            )
        msg = str(ctx.exception)
        self.assertIn("does not support this model's cache layout", msg)
        self.assertIn("FakeV4StyleBackend", msg)
        self.assertIn("FakeV4Pool", msg)
        self.assertIn("radix-built", msg)

    def test_flat_ext_zero_groups_rejected_regardless_of_spec(self):
        # Spec decode no longer gates publication (flat+spec is supported);
        # a zero-group pool is rejected with the generic pool-shaped cause.
        with self.assertRaises(RuntimeError) as ctx:
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[],
                attn_backend=FakeFlatMHABackend(),
                kv_pool=FakeMHAPool(),
                speculative_algorithm="EAGLE3",
            )
        msg = str(ctx.exception)
        self.assertIn("at least one paged-cache group", msg)
        self.assertIn("publishes none", msg)

    def test_flat_ext_zero_groups_groupless_pool_names_the_pool(self):
        # Group-less pools (e.g. mamba) publish nothing without spec decode.
        with self.assertRaises(RuntimeError) as ctx:
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[],
                attn_backend=FakeFlatMHABackend(),
                kv_pool=FakeMambaPool(),
                speculative_algorithm=None,
            )
        msg = str(ctx.exception)
        self.assertIn("at least one paged-cache group", msg)
        self.assertIn("FakeMambaPool", msg)
        self.assertIn("mamba", msg)
        self.assertIn("radix-built", msg)

    def test_flat_ext_mha_groups_passes(self):
        _pcs.validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=[FakeGroup()],
            attn_backend=FakeFlatMHABackend(),
            kv_pool=FakeMHAPool(),
            speculative_algorithm=None,
        )

    def test_radix_ext_is_a_noop_regardless(self):
        # Every combination the flat arms reject must pass untouched.
        for backend, pool, groups, spec in (
            (FakeV4StyleBackend(), FakeV4Pool(), [FakeGroup()], None),
            (FakeFlatMHABackend(), FakeMHAPool(), [], "EAGLE3"),
            (FakeSpecIncapableBackend(), FakeMHAPool(), [FakeGroup()], "EAGLE3"),
            (FakeFlatMHABackend(), FakeMambaPool(), [], "DFLASH"),
        ):
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=False,
                paged_cache_groups=groups,
                attn_backend=backend,
                kv_pool=pool,
                speculative_algorithm=spec,
            )

    def test_flat_ext_spec_incapable_backend_rejected_under_spec(self):
        # Without this guard the server would crash on the backend's
        # capture/forward spec assert instead of failing at startup.
        with self.assertRaisesRegex(RuntimeError, "speculative decoding"):
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[FakeGroup()],
                attn_backend=FakeSpecIncapableBackend(),
                kv_pool=FakeMHAPool(),
                speculative_algorithm="EAGLE3",
            )

    def test_flat_ext_spec_incapable_backend_passes_without_spec(self):
        _pcs.validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=[FakeGroup()],
            attn_backend=FakeSpecIncapableBackend(),
            kv_pool=FakeMHAPool(),
            speculative_algorithm=None,
        )

    def test_flat_ext_spec_capable_backend_passes_under_spec(self):
        _pcs.validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=[FakeGroup()],
            attn_backend=FakeFlatMHABackend(),
            kv_pool=FakeMHAPool(),
            speculative_algorithm="EAGLE3",
        )

    def test_flat_ext_dflash_rejected_even_on_capable_backend(self):
        # DFLASH block decode stays asserted-out in the backends.
        with self.assertRaisesRegex(RuntimeError, "DFLASH"):
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[FakeGroup()],
                attn_backend=FakeFlatMHABackend(),
                kv_pool=FakeMHAPool(),
                speculative_algorithm="DFLASH",
            )

    def test_backend_without_flags_single_group_falls_back(self):
        # Backends predating the class flags (getattr defaults False) may
        # fall back to the C++ single table when exactly one group is
        # published: the first-group sample IS the only group.
        class LegacyBackend:
            pass

        _pcs.validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=[FakeGroup()],
            attn_backend=LegacyBackend(),
            kv_pool=FakeMHAPool(),
            speculative_algorithm=None,
        )

    def test_backend_without_flags_multi_group_rejected(self):
        # A table-blind backend on a >1-group pool would index every layer
        # through the first-group sample — silent KV corruption on slab
        # layouts. Must refuse at startup.
        class LegacyBackend:
            pass

        with self.assertRaisesRegex(RuntimeError, "single-table fallback"):
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[FakeGroup(), FakeGroup()],
                attn_backend=LegacyBackend(),
                kv_pool=FakeMHAPool(),
                speculative_algorithm=None,
            )

    def test_hybrid_with_flat_capable_full_sub_backend_passes(self):
        # The GDN-hybrid shape: hybrid wrapper over a flat-capable full
        # backend, pool publishing KV + state groups. Only the KV-table
        # consumer (full sub-backend) is checked; the linear side has its
        # own explicit flat path and is out of scope for this guard.
        class FlatFull:
            uses_flat_cache_groups = True

        class LinearWithoutFlag:
            pass

        class FlatWrapper:
            uses_flat_cache_groups = True

            def __init__(self):
                self.full_attn_backend = FlatFull()
                self.linear_attn_backend = LinearWithoutFlag()

        _pcs.validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=[FakeGroup(), FakeGroup()],
            attn_backend=FlatWrapper(),
            kv_pool=FakeMHAPool(),
            speculative_algorithm=None,
        )

    def test_hybrid_full_sub_backend_recursed(self):
        # The hybrid wrapper is flat-capable, but its user-selectable full
        # sub-backend must be checked on its own: a table-blind sub-backend
        # would silently drop the per-group tables from common_kw.
        class FlatWrapper:
            uses_flat_cache_groups = True

            def __init__(self, full):
                self.full_attn_backend = full
                self.linear_attn_backend = None

        class LegacyFull:
            pass

        with self.assertRaisesRegex(RuntimeError, "LegacyFull"):
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[FakeGroup(), FakeGroup()],
                attn_backend=FlatWrapper(LegacyFull()),
                kv_pool=FakeMHAPool(),
                speculative_algorithm=None,
            )


if __name__ == "__main__":
    unittest.main()
