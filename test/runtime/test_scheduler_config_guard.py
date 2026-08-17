"""Cache-group scheduler configuration guards."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _load_cache_spec():
    try:
        from tokenspeed.runtime.layers.attention.kv_cache.recipes import spec

        return spec
    except (ImportError, ModuleNotFoundError):
        # The package pulls torch-backed modules. spec.py itself is
        # torch-free, so load it directly in a bare environment.
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        path = os.path.join(
            repo_root,
            "python",
            "tokenspeed",
            "runtime",
            "layers",
            "attention",
            "kv_cache",
            "recipes",
            "spec.py",
        )
        spec = importlib.util.spec_from_file_location("_cache_spec_guard", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_pcs = _load_cache_spec()


class FakeGroupSpec:
    def __init__(self, family: str):
        self.family = family


class FakeContract:
    def __init__(self, *families: str):
        self.group_specs = tuple(FakeGroupSpec(f) for f in families)


class FakePool:
    def __init__(self, contract: FakeContract | None):
        # The arena publishes the contract; a view only names its arena.
        self.arena = SimpleNamespace(runtime_contract=contract)


class HistoryBackend:
    cache_consumer_families = frozenset({"history"})


class HybridBackend:
    cache_consumer_families = frozenset({"history", "state"})


class ValidateSchedulerConfigTest(unittest.TestCase):
    def test_no_build_capability_probe(self):
        self.assertNotIn("scheduler_ext_flat_kvcache", _pcs.__dict__)

    def test_contractless_pool_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CacheRuntimeContract"):
            _pcs.validate_scheduler_config(
                attn_backend=HistoryBackend(),
                kv_pool=FakePool(None),
            )

    def test_covered_families_pass(self):
        _pcs.validate_scheduler_config(
            attn_backend=HistoryBackend(),
            kv_pool=FakePool(FakeContract("history")),
        )
        _pcs.validate_scheduler_config(
            attn_backend=HybridBackend(),
            kv_pool=FakePool(FakeContract("history", "state")),
        )

    def test_missing_family_rejected(self):
        # A hybrid pool's state group has no consumer in a history-only
        # backend; its tables would go unread.
        with self.assertRaisesRegex(RuntimeError, r"missing \['state'\]"):
            _pcs.validate_scheduler_config(
                attn_backend=HistoryBackend(),
                kv_pool=FakePool(FakeContract("history", "state")),
            )

    def test_backend_without_declared_families_rejected(self):
        class UndeclaredBackend:
            pass

        with self.assertRaisesRegex(RuntimeError, "consumer families"):
            _pcs.validate_scheduler_config(
                attn_backend=UndeclaredBackend(),
                kv_pool=FakePool(FakeContract("history")),
            )


if __name__ == "__main__":
    unittest.main()
