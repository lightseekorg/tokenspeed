"""Startup capability guards for DeepSeek V4 cache management."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch


def _load_capabilities_module():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path = os.path.join(
        repo_root,
        "python",
        "tokenspeed",
        "runtime",
        "deepseek_v4_cache_capabilities.py",
    )
    spec = importlib.util.spec_from_file_location("_deepseek_v4_capabilities", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_caps = _load_capabilities_module()


def _load_pd_factory():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path = os.path.join(
        repo_root, "python", "tokenspeed", "runtime", "pd", "factory.py"
    )

    def _stub(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    class _KVArgs:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    stubs = {
        "tokenspeed.runtime.deepseek_v4_cache_capabilities": _caps,
        "tokenspeed.runtime.pd.decode_executor": _stub(
            "tokenspeed.runtime.pd.decode_executor", DisaggDecodeExecutor=object
        ),
        "tokenspeed.runtime.pd.mooncake.entities": _stub(
            "tokenspeed.runtime.pd.mooncake.entities",
            KVArgs=_KVArgs,
            KVManagerArgs=object,
        ),
        "tokenspeed.runtime.pd.prefill_executor": _stub(
            "tokenspeed.runtime.pd.prefill_executor", DisaggPrefillExecutor=object
        ),
        "tokenspeed.runtime.pd.utils": _stub(
            "tokenspeed.runtime.pd.utils", TransferBackend=object
        ),
    }
    spec = importlib.util.spec_from_file_location("_pd_factory_capability_guard", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self, supports_pd_transfer: bool = True) -> None:
        self.supports_pd_transfer = supports_pd_transfer
        self.pointer_reads = 0

    def get_contiguous_buf_infos(self):
        self.pointer_reads += 1
        raise AssertionError("pointer enumeration must not run for rejected pools")


class DeepseekV4CacheCapabilitiesTest(unittest.TestCase):
    def test_active_pd_rejects_target_or_draft_v4(self):
        for mode in ("prefill", "decode"):
            for target_is_v4, draft_is_v4, role in (
                (True, False, "target"),
                (False, True, "draft"),
                (True, True, "target,draft"),
            ):
                with self.subTest(mode=mode, role=role):
                    with self.assertRaisesRegex(RuntimeError, role):
                        _caps.validate_deepseek_v4_cache_capabilities(
                            target_is_v4=target_is_v4,
                            draft_is_v4=draft_is_v4,
                            disaggregation_mode=mode,
                            flat_kvcache_ext=True,
                            enable_kvstore=False,
                        )

    def test_non_pd_modes_do_not_claim_pd_support(self):
        for mode in ("null", "encode"):
            with self.subTest(mode=mode):
                _caps.validate_deepseek_v4_cache_capabilities(
                    target_is_v4=True,
                    draft_is_v4=False,
                    disaggregation_mode=mode,
                    flat_kvcache_ext=True,
                    enable_kvstore=False,
                )

    def test_flat_v4_rejects_l2_kvstore(self):
        with self.assertRaisesRegex(RuntimeError, "device-side L1 only"):
            _caps.validate_deepseek_v4_cache_capabilities(
                target_is_v4=True,
                draft_is_v4=False,
                disaggregation_mode="null",
                flat_kvcache_ext=True,
                enable_kvstore=True,
            )

    def test_radix_or_non_v4_keeps_existing_kvstore_behavior(self):
        for target_is_v4, flat_kvcache_ext in ((False, True), (True, False)):
            with self.subTest(
                target_is_v4=target_is_v4, flat_kvcache_ext=flat_kvcache_ext
            ):
                _caps.validate_deepseek_v4_cache_capabilities(
                    target_is_v4=target_is_v4,
                    draft_is_v4=False,
                    disaggregation_mode="null",
                    flat_kvcache_ext=flat_kvcache_ext,
                    enable_kvstore=True,
                )

    def test_pd_pool_guard_runs_before_pointer_enumeration(self):
        target = _Pool(supports_pd_transfer=False)
        draft = _Pool(supports_pd_transfer=True)

        with self.assertRaisesRegex(RuntimeError, "target"):
            _caps.validate_pd_transfer_pools(target, draft)

        self.assertEqual(target.pointer_reads, 0)
        self.assertEqual(draft.pointer_reads, 0)

    def test_factory_rejects_draft_before_reading_target_pointers(self):
        factory = _load_pd_factory()
        target = _Pool(supports_pd_transfer=True)
        draft = _Pool(supports_pd_transfer=False)

        with self.assertRaisesRegex(RuntimeError, "draft"):
            factory.get_kv_args(0, 0, None, target, draft)

        self.assertEqual(target.pointer_reads, 0)
        self.assertEqual(draft.pointer_reads, 0)

    def test_legacy_pool_defaults_to_supported(self):
        class LegacyPool:
            pass

        _caps.validate_pd_transfer_pools(LegacyPool(), None)

    def test_event_loop_guard_precedes_distributed_initialization(self):
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        path = os.path.join(
            repo_root, "python", "tokenspeed", "runtime", "engine", "event_loop.py"
        )
        with open(path, encoding="utf-8") as source:
            module = ast.parse(source.read())

        event_loop = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "EventLoop"
        )
        init = next(
            node
            for node in event_loop.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        )

        def _called_name(node):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        yield child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        yield child.func.attr

        guard_line = min(
            node.lineno
            for node in init.body
            if "validate_deepseek_v4_cache_capabilities" in set(_called_name(node))
        )
        distributed_line = min(
            node.lineno
            for node in init.body
            if "_init_distributed" in set(_called_name(node))
        )
        self.assertLess(guard_line, distributed_line)


if __name__ == "__main__":
    unittest.main()
