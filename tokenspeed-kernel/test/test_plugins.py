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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

import pytest
import torch
from tokenspeed_kernel import plugins as plugins_mod
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.plugins import (
    DISABLE_ENV_VAR,
    PluginInfo,
    disable_plugin,
    discover_plugins,
    list_plugins,
    reset_plugins,
)
from tokenspeed_kernel.plugins.cli import main as cli_main
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import format_signatures

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_plugins():
    """Reset registry and plugin state before/after each test."""
    KernelRegistry.reset()
    reset_plugins()
    yield
    KernelRegistry.reset()
    reset_plugins()


class _FakeMetadata(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeDist:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self.metadata = _FakeMetadata({"Name": name})


class _FakeEntryPoint:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(
        self,
        name: str,
        register_fn: Callable[[], None],
        *,
        package: str = "fake-plugin",
        version: str = "0.0.1",
    ) -> None:
        self.name = name
        self.group = plugins_mod.ENTRY_POINT_GROUP
        self.value = f"{package}:register"
        self._fn = register_fn
        self.dist = _FakeDist(package, version)

    def load(self) -> Callable[[], None]:
        return self._fn


@pytest.fixture
def patch_entry_points(monkeypatch):
    """Helper to replace the entry-point lookup with a fixed list."""

    def _apply(eps: list[_FakeEntryPoint]) -> None:
        monkeypatch.setattr(plugins_mod, "_entry_points", lambda group: list(eps))

    return _apply


def _make_register(
    name: str,
    *,
    family: str = "gemm",
    mode: str = "mm",
    solution: str | None = None,
    priority: int = 12,
    storage_dtypes=None,
    capability: CapabilityRequirement | None = None,
) -> Callable[[], None]:
    sol = solution or name

    def register():
        @register_kernel(
            family,
            mode,
            name=name,
            solution=sol,
            signatures=format_signatures(
                ("a", "b"), "dense", storage_dtypes or {torch.bfloat16}
            ),
            priority=priority,
            capability=capability,
        )
        def impl(*args, **kwargs):
            return name

        return impl

    return register


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPluginInfo:
    def test_frozen(self):
        info = PluginInfo(name="x", package="pkg", version="1.0", num_kernels=0)
        with pytest.raises(Exception):
            info.name = "y"

    def test_default_kernel_names(self):
        info = PluginInfo(name="x", package="pkg", version="1.0", num_kernels=0)
        assert info.kernel_names == ()


class TestExplicitRegistration:
    """The decorator path is what plugins use; test it works standalone."""

    def test_register_kernel_decorator(self, fresh_plugins):
        @register_kernel(
            "gemm",
            "mm",
            solution="manual",
            signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
        )
        def impl(a, b):
            return a @ b

        spec = KernelRegistry.get().get_by_name("manual_gemm_mm")
        assert spec is not None
        assert spec.priority == 10  # default
        assert KernelRegistry.get().get_impl("manual_gemm_mm") is impl


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discover_no_plugins(self, fresh_plugins, patch_entry_points):
        patch_entry_points([])
        loaded = discover_plugins()
        assert loaded == []
        assert list_plugins() == []

    def test_discover_loads_kernels(self, fresh_plugins, patch_entry_points):
        patch_entry_points(
            [
                _FakeEntryPoint(
                    "alpha",
                    _make_register("alpha_kernel"),
                    package="alpha-plugin",
                    version="1.2.3",
                ),
            ]
        )
        loaded = discover_plugins()
        assert len(loaded) == 1
        info = loaded[0]
        assert info.name == "alpha"
        assert info.package == "alpha-plugin"
        assert info.version == "1.2.3"
        assert info.num_kernels == 1
        assert info.kernel_names == ("alpha_kernel",)
        assert KernelRegistry.get().get_by_name("alpha_kernel") is not None

    def test_discover_alphabetical_order(self, fresh_plugins, patch_entry_points):
        order: list[str] = []

        def make(name):
            def register():
                order.append(name)

                @register_kernel(
                    "gemm",
                    "mm",
                    name=f"k_{name}",
                    solution=name,
                    signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
                )
                def impl():
                    return name

            return register

        # Provide entry points in non-alphabetical order; discovery should
        # sort them so loading is deterministic.
        patch_entry_points(
            [
                _FakeEntryPoint("zeta", make("zeta")),
                _FakeEntryPoint("alpha", make("alpha")),
                _FakeEntryPoint("mu", make("mu")),
            ]
        )
        discover_plugins()
        assert order == ["alpha", "mu", "zeta"]

    def test_discover_skips_already_loaded(self, fresh_plugins, patch_entry_points):
        calls = {"n": 0}

        def register():
            calls["n"] += 1

            @register_kernel(
                "gemm",
                "mm",
                name="once_kernel",
                solution="once",
                signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
            )
            def impl():
                return None

        patch_entry_points([_FakeEntryPoint("once", register)])
        discover_plugins()
        discover_plugins()
        assert calls["n"] == 1

    def test_discover_force_reloads(self, fresh_plugins, patch_entry_points):
        calls = {"n": 0}

        def register():
            calls["n"] += 1

            @register_kernel(
                "gemm",
                "mm",
                name="force_kernel",
                solution="force",
                signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
            )
            def impl():
                return None

        patch_entry_points([_FakeEntryPoint("force", register)])
        discover_plugins()
        KernelRegistry.reset()  # simulate registry reset
        discover_plugins(force=True)
        assert calls["n"] == 2
        assert KernelRegistry.get().get_by_name("force_kernel") is not None


class TestOverride:
    def test_higher_priority_plugin_wins(self, fresh_plugins, patch_entry_points):
        # Built-in style kernel registered before plugin discovery.
        @register_kernel(
            "gemm",
            "mm",
            name="builtin_gemm",
            solution="builtin",
            signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
            priority=10,
        )
        def builtin(a, b):
            return "builtin"

        patch_entry_points(
            [
                _FakeEntryPoint(
                    "vendor",
                    _make_register("vendor_gemm", priority=19),
                ),
            ]
        )
        discover_plugins()

        specs = KernelRegistry.get().get_for_operator("gemm", "mm")
        # Priority-sorted descending: vendor first.
        assert specs[0].name == "vendor_gemm"
        assert specs[0].priority == 19


# ---------------------------------------------------------------------------
# Isolation, disabling
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_failing_plugin_does_not_break_others(
        self, fresh_plugins, patch_entry_points
    ):
        def boom():
            raise RuntimeError("import failed")

        patch_entry_points(
            [
                _FakeEntryPoint("broken", boom),
                _FakeEntryPoint("good", _make_register("good_kernel")),
            ]
        )

        with pytest.warns(UserWarning, match="Failed to load kernel plugin 'broken'"):
            loaded = discover_plugins()

        assert {info.name for info in loaded} == {"good"}
        assert KernelRegistry.get().get_by_name("good_kernel") is not None
        assert KernelRegistry.get().get_by_name("broken") is None


class TestDisable:
    def test_disable_via_api(self, fresh_plugins, patch_entry_points):
        patch_entry_points(
            [
                _FakeEntryPoint("a", _make_register("a_kernel")),
                _FakeEntryPoint("b", _make_register("b_kernel")),
            ]
        )
        disable_plugin("a")
        loaded = discover_plugins()
        assert {info.name for info in loaded} == {"b"}
        assert KernelRegistry.get().get_by_name("a_kernel") is None
        assert KernelRegistry.get().get_by_name("b_kernel") is not None

    def test_disable_via_env(self, fresh_plugins, patch_entry_points, monkeypatch):
        monkeypatch.setenv(DISABLE_ENV_VAR, "a, c")  # whitespace, multi
        patch_entry_points(
            [
                _FakeEntryPoint("a", _make_register("a_kernel")),
                _FakeEntryPoint("b", _make_register("b_kernel")),
                _FakeEntryPoint("c", _make_register("c_kernel")),
            ]
        )
        loaded = discover_plugins()
        assert {info.name for info in loaded} == {"b"}


# ---------------------------------------------------------------------------
# Priority collision warning
# ---------------------------------------------------------------------------


class TestCollision:
    def test_equal_priority_emits_warning(self, fresh_plugins, patch_entry_points):
        # Pre-existing kernel (built-in style) at priority 14.
        @register_kernel(
            "gemm",
            "mm",
            name="builtin_eq",
            solution="builtin",
            signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
            priority=14,
        )
        def b(a, x):
            return "b"

        patch_entry_points(
            [
                _FakeEntryPoint(
                    "tied",
                    _make_register("tied_kernel", priority=14),
                ),
            ]
        )
        with pytest.warns(UserWarning, match="equal priority"):
            discover_plugins()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_list_empty(self, fresh_plugins, patch_entry_points):
        # Patch with [] so the CLI's discover call ignores any real plugins
        # that happen to be installed in the test environment.
        patch_entry_points([])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["list"])
        assert rc == 0
        assert "No kernel plugins" in buf.getvalue()

    def test_list_with_plugins(self, fresh_plugins, patch_entry_points):
        patch_entry_points(
            [
                _FakeEntryPoint(
                    "alpha",
                    _make_register("alpha_kernel"),
                    version="1.2.3",
                ),
            ]
        )
        discover_plugins()

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["list"])
        out = buf.getvalue()
        assert rc == 0
        assert "alpha" in out
        assert "1.2.3" in out
        assert "alpha_kernel" in out

    def test_info_unknown(self, fresh_plugins, patch_entry_points):
        patch_entry_points([])
        err = io.StringIO()
        with redirect_stderr(err):
            rc = cli_main(["info", "nope"])
        assert rc == 1
        assert "not loaded" in err.getvalue()

    def test_info_known(self, fresh_plugins, patch_entry_points):
        patch_entry_points(
            [
                _FakeEntryPoint(
                    "alpha",
                    _make_register("alpha_kernel", priority=15),
                    version="2.0",
                ),
            ]
        )
        discover_plugins()

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["info", "alpha"])
        out = buf.getvalue()
        assert rc == 0
        assert "alpha_kernel" in out
        assert "gemm.mm" in out
        assert "priority=15" in out

    def test_cli_discovers_plugins(self, fresh_plugins, patch_entry_points):
        # Loading is explicit, but `python -m tokenspeed_kernel.plugins`
        # should still see installed plugins without the caller invoking
        # discover_plugins() first.
        patch_entry_points([_FakeEntryPoint("auto", _make_register("auto_kernel"))])
        # No discover_plugins() here — the CLI itself must drive discovery.
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["list"])
        out = buf.getvalue()
        assert rc == 0
        assert "auto" in out
        assert "auto_kernel" in out
