import os
import sys
from types import ModuleType

from tokenspeed_kernel.ops.other.native.deep_ep import load_deep_ep


def test_load_deep_ep_uses_lazy_global_symbols_and_restores_flags(
    monkeypatch,
) -> None:
    original_flags = sys.getdlopenflags()
    imported = ModuleType("deep_ep")

    def import_module(name: str) -> ModuleType:
        assert name == "deep_ep"
        assert sys.getdlopenflags() == os.RTLD_LAZY | os.RTLD_GLOBAL
        return imported

    monkeypatch.setattr(
        "tokenspeed_kernel.ops.other.native.deep_ep.importlib.import_module",
        import_module,
    )

    assert load_deep_ep() is imported
    assert sys.getdlopenflags() == original_flags


def test_load_deep_ep_restores_flags_after_import_error(monkeypatch) -> None:
    original_flags = sys.getdlopenflags()

    def import_module(_name: str) -> ModuleType:
        raise ImportError("unavailable")

    monkeypatch.setattr(
        "tokenspeed_kernel.ops.other.native.deep_ep.importlib.import_module",
        import_module,
    )

    try:
        load_deep_ep()
    except ImportError:
        pass
    else:
        raise AssertionError("load_deep_ep should propagate ImportError")

    assert sys.getdlopenflags() == original_flags
