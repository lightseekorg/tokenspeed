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


class TestBoundaryOnlySpecGuard:
    def test_boundary_only_requires_sinkless_replayable_sliding_window(self):
        spec = _pcs.CacheGroupSpec(
            group_id="tail",
            attn_kind="sliding_window",
            rows_per_page=2,
            entry_stride_tokens=1,
            sliding_window_tokens=2,
            replayable=True,
            boundary_only_in_prefill=True,
        )
        assert spec.boundary_only_in_prefill is True

        for bad_kwargs in (
            dict(attn_kind="full_history", replayable=False),
            dict(attn_kind="sliding_window", replayable=False),
            dict(attn_kind="sliding_window", replayable=False, sink_tokens=2),
        ):
            with pytest.raises(ValueError, match="boundary_only_in_prefill"):
                _pcs.CacheGroupSpec(
                    group_id="bad",
                    rows_per_page=2,
                    entry_stride_tokens=1,
                    sliding_window_tokens=(
                        2 if bad_kwargs["attn_kind"] == "sliding_window" else None
                    ),
                    boundary_only_in_prefill=True,
                    **bad_kwargs,
                )


class TestMaxRemoteAdmissionsArgs:
    def test_server_args_max_remote_admissions_per_round(self):
        from tokenspeed.runtime.utils.server_args import ServerArgs

        args = ServerArgs(model="m", disaggregation_mode="decode")
        assert args.disaggregation_max_remote_admissions_per_round == 1
        assert args.scheduler_max_remote_admissions_per_round == 1

        args_batched = ServerArgs(
            model="m",
            disaggregation_mode="decode",
            disaggregation_max_remote_admissions_per_round=8,
        )
        args_batched.validate_cache_options()
        assert args_batched.scheduler_max_remote_admissions_per_round == 8

        prefill_args = ServerArgs(
            model="m",
            disaggregation_mode="prefill",
            disaggregation_max_remote_admissions_per_round=8,
        )
        prefill_args.validate_cache_options()
        assert prefill_args.scheduler_max_remote_admissions_per_round == 1

        with pytest.raises(ValueError, match="disaggregation-max-remote-admissions"):
            ServerArgs(
                model="m",
                disaggregation_mode="decode",
                disaggregation_max_remote_admissions_per_round=0,
            ).validate_cache_options()

        with pytest.raises(ValueError, match="disaggregation-max-remote-admissions"):
            ServerArgs(
                model="m",
                disaggregation_mode="null",
                disaggregation_max_remote_admissions_per_round=4,
            ).validate_cache_options()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
