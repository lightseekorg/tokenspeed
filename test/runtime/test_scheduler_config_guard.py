"""Cache-group scheduler configuration guards."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

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
    def __init__(self, attn_kind: str):
        self.attn_kind = attn_kind


class FakeContract:
    def __init__(self, *attn_kinds: str):
        self.group_specs = tuple(FakeGroupSpec(k) for k in attn_kinds)


class FakePool:
    def __init__(self, contract: FakeContract):
        # The arena publishes the contract; a view only names its arena.
        self.arena = SimpleNamespace(runtime_contract=contract)


class HistoryBackend:
    cache_consumer_attn_kinds = frozenset({"full_history", "sliding_window"})


class FullOnlyBackend:
    cache_consumer_attn_kinds = frozenset({"full_history"})


class HybridBackend:
    cache_consumer_attn_kinds = frozenset({"full_history", "sliding_window", "state"})


class TestValidateSchedulerConfigTest:
    def test_covered_kinds_pass(self):
        _pcs.validate_scheduler_config(
            attn_backend=HistoryBackend(),
            kv_pool=FakePool(FakeContract("full_history", "sliding_window")),
        )
        _pcs.validate_scheduler_config(
            attn_backend=HybridBackend(),
            kv_pool=FakePool(FakeContract("full_history", "state")),
        )

    def test_missing_kind_rejected(self):
        # A hybrid pool's state group has no consumer in a history-only
        # backend; its tables would go unread.
        with pytest.raises((RuntimeError), match=(r"missing \['state'\]")):
            _pcs.validate_scheduler_config(
                attn_backend=HistoryBackend(),
                kv_pool=FakePool(FakeContract("full_history", "state")),
            )
        # A backend that reads only full-history rows cannot serve a window.
        with pytest.raises((RuntimeError), match=(r"missing \['sliding_window'\]")):
            _pcs.validate_scheduler_config(
                attn_backend=FullOnlyBackend(),
                kv_pool=FakePool(FakeContract("full_history", "sliding_window")),
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
