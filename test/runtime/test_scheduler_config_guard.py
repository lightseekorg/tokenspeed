"""Cache-group scheduler configuration guards."""

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
        # The configs package imports transformer-backed model configs. This
        # module is torch-free, so load it directly in a bare environment.
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
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_pcs = _load_paged_cache_spec()


class FakeCacheGroupBackend:
    uses_cache_groups = True


class FakeSpecIncapableBackend:
    uses_cache_groups = True
    cache_group_spec_capable = False


class FakePool:
    pass


class FakeGroup:
    group_id = "full_attention"


class ValidateSchedulerConfigTest(unittest.TestCase):
    def test_no_build_capability_probe(self):
        self.assertNotIn("scheduler_ext_flat_kvcache", _pcs.__dict__)

    def test_zero_groups_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "at least one paged-cache group"):
            _pcs.validate_scheduler_config(
                paged_cache_groups=[],
                attn_backend=FakeCacheGroupBackend(),
                kv_pool=FakePool(),
            )

    def test_single_group_table_blind_backend_uses_compatibility_table(self):
        class SingleTableBackend:
            pass

        _pcs.validate_scheduler_config(
            paged_cache_groups=[FakeGroup()],
            attn_backend=SingleTableBackend(),
            kv_pool=FakePool(),
        )

    def test_multi_group_table_blind_backend_rejected(self):
        class SingleTableBackend:
            pass

        with self.assertRaisesRegex(RuntimeError, "single-table fallback"):
            _pcs.validate_scheduler_config(
                paged_cache_groups=[FakeGroup(), FakeGroup()],
                attn_backend=SingleTableBackend(),
                kv_pool=FakePool(),
            )

    def test_multi_group_consumer_passes(self):
        _pcs.validate_scheduler_config(
            paged_cache_groups=[FakeGroup(), FakeGroup()],
            attn_backend=FakeCacheGroupBackend(),
            kv_pool=FakePool(),
        )

    def test_spec_incapable_backend_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "speculative decoding"):
            _pcs.validate_scheduler_config(
                paged_cache_groups=[FakeGroup()],
                attn_backend=FakeSpecIncapableBackend(),
                kv_pool=FakePool(),
                speculative_algorithm="EAGLE3",
            )

    def test_dflash_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "DFLASH"):
            _pcs.validate_scheduler_config(
                paged_cache_groups=[FakeGroup()],
                attn_backend=FakeCacheGroupBackend(),
                kv_pool=FakePool(),
                speculative_algorithm="DFLASH",
            )

    def test_hybrid_checks_full_attention_backend(self):
        class Wrapper:
            uses_cache_groups = True

            def __init__(self):
                self.full_attn_backend = object()

        with self.assertRaisesRegex(RuntimeError, "object"):
            _pcs.validate_scheduler_config(
                paged_cache_groups=[FakeGroup(), FakeGroup()],
                attn_backend=Wrapper(),
                kv_pool=FakePool(),
            )


if __name__ == "__main__":
    unittest.main()
