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

import importlib.metadata
import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

import pytest
import tokenspeed_kernel.plugins as plugins_mod
import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
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
from tokenspeed_kernel.selection import KernelPlan, select_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_plugins(monkeypatch: pytest.MonkeyPatch):
    """Reset registry and plugin state before/after each test."""
    # Preserve the process-wide built-in registry for tests collected after
    # this module; monkeypatch restores it after this fixture tears down.
    monkeypatch.setattr(KernelRegistry, "_instance", None)
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
    traits=None,
    signatures=None,
) -> Callable[[], None]:
    sol = solution or name

    def register():
        @register_kernel(
            family,
            mode,
            name=name,
            solution=sol,
            signatures=(
                signatures
                if signatures is not None
                else format_signatures(
                    ("a", "b"), "dense", storage_dtypes or {torch.bfloat16}
                )
            ),
            priority=priority,
            capability=capability,
            traits=traits,
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
    @pytest.mark.parametrize("import_during_load", [True, False])
    def test_lazy_builtins_are_not_owned_by_plugin(
        self,
        fresh_plugins,
        patch_entry_points,
        monkeypatch,
        tmp_path,
        import_during_load,
    ):
        import tokenspeed_kernel

        core_name = "tokenspeed_kernel._test_lazy_core"
        plugin_name = "test_lazy_kernel_plugin"
        (tmp_path / "_test_lazy_core.py").write_text(
            "from tokenspeed_kernel.registry import register_kernel\n"
            "from tokenspeed_kernel.signature import format_signature\n"
            "@register_kernel('test', 'core', name='core_first', solution='core', "
            "signatures={format_signature()}, priority=12)\n"
            "def core_impl(): return 'core'\n"
            "register_kernel('test', 'core', name='core_second', solution='core', "
            "signatures={format_signature()}, priority=12)(core_impl)\n"
        )
        core_import = f"from {core_name} import core_impl\n"
        (tmp_path / f"{plugin_name}.py").write_text(
            "from tokenspeed_kernel.registry import register_kernel\n"
            "from tokenspeed_kernel.signature import format_signature\n"
            + (core_import if import_during_load else "")
            + "def register():\n"
            + ("" if import_during_load else f"    {core_import}")
            + "    register_kernel('test', 'plugin', name='plugin_reuse', "
            "solution='plugin', signatures={format_signature()}, priority=12)(core_impl)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(
            tokenspeed_kernel, "__path__", [*tokenspeed_kernel.__path__, str(tmp_path)]
        )
        patch_entry_points(
            [
                importlib.metadata.EntryPoint(
                    name="lazy",
                    value=f"{plugin_name}:register",
                    group=plugins_mod.ENTRY_POINT_GROUP,
                )
            ]
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = discover_plugins()
            assert not caught
            assert loaded[0].kernel_names == ("plugin_reuse",)
            assert loaded[0].num_kernels == 1
            registry = KernelRegistry.get()
            assert registry.get_impl("core_first")() == "core"
            assert registry.get_impl("core_second")() == "core"
            assert registry.get_impl("plugin_reuse") is registry.get_impl("core_first")
        finally:
            sys.modules.pop(core_name, None)
            sys.modules.pop(plugin_name, None)
            vars(tokenspeed_kernel).pop("_test_lazy_core", None)

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
    @pytest.mark.parametrize(
        "left,right",
        [
            (
                {"capability": CapabilityRequirement(vendors=frozenset({"nvidia"}))},
                {"capability": CapabilityRequirement(vendors=frozenset({"amd"}))},
            ),
            (
                {
                    "capability": CapabilityRequirement(
                        max_arch_version=ArchVersion(9, 0)
                    )
                },
                {
                    "capability": CapabilityRequirement(
                        min_arch_version=ArchVersion(10, 0)
                    )
                },
            ),
            (
                {
                    "capability": CapabilityRequirement(
                        vendors=frozenset({"nvidia", "amd"}),
                        vendor_min_arch_versions={
                            "nvidia": ArchVersion(10, 0),
                            "amd": ArchVersion(9, 5),
                        },
                    )
                },
                {
                    "capability": CapabilityRequirement(
                        vendors=frozenset({"nvidia"}),
                        max_arch_version=ArchVersion(9, 0),
                    )
                },
            ),
            ({"storage_dtypes": {torch.bfloat16}}, {"storage_dtypes": {torch.float16}}),
            (
                {"traits": {"m": frozenset({16, 32})}},
                {"traits": {"m": frozenset({64})}},
            ),
            (
                {"traits": {"tokens": range(1, 257)}},
                {"traits": {"tokens_min": frozenset({2560})}},
            ),
            (
                {"traits": {"m_max": frozenset({128})}},
                {"traits": {"m_min": frozenset({256})}},
            ),
            (
                {"traits": {"m": frozenset({3, 5})}},
                {"traits": {"m_align": frozenset({2})}},
            ),
            (
                {"traits": {"hidden": frozenset({96})}},
                {"traits": {"hidden_alignment": frozenset({128})}},
            ),
        ],
        ids=[
            "vendor",
            "arch",
            "vendor-arch",
            "dtype",
            "shape",
            "range-bound",
            "bounds",
            "alignment",
            "moe-alignment",
        ],
    )
    def test_disjoint_domains_are_silent(
        self, fresh_plugins, patch_entry_points, left, right
    ):
        _make_register("existing", **left)()
        patch_entry_points([_FakeEntryPoint("plugin", _make_register("new", **right))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = discover_plugins()
        assert not caught
        assert loaded[0].kernel_names == ("new",)
        assert len(KernelRegistry.get().get_for_operator("gemm", "mm")) == 2

    @pytest.mark.parametrize(
        "traits",
        [
            ({"m_max": frozenset({128})}, {"m_min": frozenset({128})}),
            ({"m_align": frozenset({32})}, {"m_align": frozenset({128})}),
            (
                {"hidden_alignment": frozenset({32})},
                {"hidden_alignment": frozenset({128})},
            ),
            ({"head_dim": frozenset({64})}, {"head_dim": frozenset({128})}),
        ],
    )
    def test_possible_overlap_still_warns(
        self, fresh_plugins, patch_entry_points, traits
    ):
        _make_register("existing", traits=traits[0])()
        patch_entry_points(
            [_FakeEntryPoint("plugin", _make_register("new", traits=traits[1]))]
        )
        with pytest.warns(UserWarning, match="equal priority"):
            discover_plugins()

    def test_predicates_are_not_executed(self, fresh_plugins, patch_entry_points):
        def predicate(*args):
            raise AssertionError("Discovery must not execute predicates")

        _make_register("existing", traits={"problem_filter": frozenset({predicate})})()
        patch_entry_points([_FakeEntryPoint("plugin", _make_register("new"))])
        with pytest.warns(UserWarning, match="equal priority"):
            discover_plugins()

    def test_positive_features_can_coexist(self, fresh_plugins, patch_entry_points):
        _make_register(
            "existing",
            capability=CapabilityRequirement(
                required_features=frozenset({"feature_a"})
            ),
        )()
        patch_entry_points(
            [
                _FakeEntryPoint(
                    "plugin",
                    _make_register(
                        "new",
                        capability=CapabilityRequirement(
                            required_features=frozenset({"feature_b"})
                        ),
                    ),
                )
            ]
        )
        with pytest.warns(UserWarning, match="equal priority"):
            discover_plugins()

    def test_layouts_can_be_ignored(self, fresh_plugins, patch_entry_points):
        def layout_a():
            pass

        def layout_b():
            pass

        _make_register("existing", signatures={format_signature(layouts={layout_a})})()
        patch_entry_points(
            [
                _FakeEntryPoint(
                    "plugin",
                    _make_register(
                        "new", signatures={format_signature(layouts={layout_b})}
                    ),
                )
            ]
        )
        with pytest.warns(UserWarning, match="equal priority"):
            discover_plugins()

    def test_pair_warned_once_without_changing_selection(
        self, fresh_plugins, patch_entry_points, h100_platform
    ):
        def register():
            _make_register("z_first")()
            _make_register("a_second")()

        patch_entry_points([_FakeEntryPoint("plugin", register)])
        with pytest.warns(UserWarning, match="equal priority") as caught:
            discover_plugins()
        assert len(caught) == 1
        signature = next(iter(format_signatures(("a", "b"), "dense", {torch.bfloat16})))
        selected = select_kernel("gemm", "mm", signature, platform=h100_platform)
        assert selected.name == "z_first"
        assert selected() == "z_first"
        planned = KernelPlan.select(
            "gemm",
            "mm",
            signature,
            features=frozenset(),
            static_traits={},
            variable_traits=frozenset(),
            declared_traits=frozenset(),
            solution=None,
            ignore_layout=False,
            platform=h100_platform,
        )
        assert planned.kernel.name == "a_second"
        assert planned.kernel() == "a_second"

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
