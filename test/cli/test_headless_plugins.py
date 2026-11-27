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


"""Exercise supervisor plugin preparation with real entry points on a CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("no-plugin", ["kernels", "ranks"]),
        ("legacy", ["kernels", "full", "trace", "metrics", "ranks"]),
        ("split", ["cli", "trace-cli", "ranks", "kernels", "full", "trace", "metrics"]),
        ("unknown", ["kernels", "full", "trace", "metrics", "other", "ranks"]),
        ("cli-error", ["cli", "kernels", "full", "trace", "metrics", "ranks"]),
        ("full-error", ["cli", "trace-cli", "ranks", "kernels", "full"]),
        (
            "second-cli-error",
            ["cli", "trace-cli", "kernels", "full", "trace", "metrics", "ranks"],
        ),
        (
            "disabled-legacy",
            ["cli", "trace-cli", "ranks", "kernels", "full", "trace", "metrics"],
        ),
    ],
)
def test_headless_plugin_registration(tmp_path, mode, expected):
    companion = tmp_path / "fixture_model_plugin"
    companion.mkdir()
    (companion / "__init__.py").write_text(dedent("""
            import os
            from dataclasses import dataclass
            from fixture_events import record
            from tokenspeed.runtime.configs.numerics import register_numerics_envelope
            from tokenspeed.runtime.plugins.registry import register_extra_server_args as register_args

            @dataclass(frozen=True)
            class Options:
                fixture_enabled: bool = False

            def add_numerics():
                register_args("fixture", Options, validate=None)
                register_numerics_envelope(
                    "fixture-profile", bitwise=False,
                    resolve=lambda args: None, validate=lambda **args: None,
                )

            def register_extra_server_args():
                record("cli")
                add_numerics()
                if os.environ["HEADLESS_MODE"] == "cli-error":
                    raise RuntimeError("invalid lightweight registration")

            def register():
                record("full")
                add_numerics()
                if os.environ["HEADLESS_MODE"] == "full-error":
                    raise RuntimeError("invalid full registration")

            if os.environ["HEADLESS_MODE"] == "legacy":
                del register_extra_server_args
            """))
    (companion / "tracing.py").write_text(dedent("""
            import os
            from dataclasses import dataclass
            from fixture_events import record
            from tokenspeed.runtime.plugins.registry import register_extra_server_args as register_args

            @dataclass(frozen=True)
            class Options:
                trace_enabled: bool = False

            def register_extra_server_args():
                record("trace-cli")
                register_args("tracing", Options, validate=None)
                if os.environ["HEADLESS_MODE"] == "second-cli-error":
                    raise RuntimeError("invalid second lightweight registration")

            def register():
                record("trace")
                register_args("tracing", Options, validate=None)
            """))
    (tmp_path / "fixture_metrics_plugin.py").write_text(
        'from fixture_events import record\ndef register():\n    record("metrics")\n'
        "def register_extra_server_args():\n    pass\n"
    )
    (tmp_path / "other_plugin.py").write_text(
        'from fixture_events import record\ndef register():\n    record("other")\n'
    )
    (tmp_path / "fixture_events.py").write_text(dedent("""
            events = []

            def record(value):
                events.append(value)
            """))

    if mode != "no-plugin":
        distribution = tmp_path / "headless_fixture-1.0.dist-info"
        distribution.mkdir()
        (distribution / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: headless-fixture\nVersion: 1.0\n"
        )
        entries = [
            "[tokenspeed.plugins]",
            "a_model = fixture_model_plugin:register",
            "b_tracing = fixture_model_plugin.tracing:register",
            "c_metrics = fixture_metrics_plugin:register",
        ]
        if mode in ("unknown", "disabled-legacy"):
            entries.append("z_other = other_plugin:register")
        entries.extend(
            [
                "[tokenspeed_kernel.plugins]",
                "model = fixture_model_plugin.kernels:register",
            ]
        )
        (distribution / "entry_points.txt").write_text("\n".join(entries) + "\n")

    python_source = Path(__file__).resolve().parents[2] / "python"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(python_source))),
        "HEADLESS_MODE": mode,
    }
    env.pop("TOKENSPEED_DISABLE_PLUGINS", None)
    if mode == "disabled-legacy":
        env["TOKENSPEED_DISABLE_PLUGINS"] = "z_other"
    driver = dedent("""
        import json
        import os
        from pathlib import Path
        import sys
        from types import ModuleType

        import tokenspeed
        from fixture_events import events, record

        # Supply the accelerator package boundary on a CPU.
        kernel = ModuleType("tokenspeed_kernel")
        kernel.__path__ = []
        platform = ModuleType("tokenspeed_kernel.platform")
        platform.current_platform = object()
        kernel_plugins = ModuleType("tokenspeed_kernel.plugins")
        kernel_plugins.discover_plugins = lambda: record("kernels")
        sys.modules.update({
            kernel.__name__: kernel, platform.__name__: platform,
            kernel_plugins.__name__: kernel_plugins,
        })
        # Import the real numerics module without loading the model catalog.
        configs = ModuleType("tokenspeed.runtime.configs")
        configs.__path__ = [
            str(Path(base) / "runtime" / "configs") for base in tokenspeed.__path__
        ]
        sys.modules[configs.__name__] = configs

        from tokenspeed.runtime import plugins
        from tokenspeed.runtime.configs.numerics import NUMERICS_ENVELOPES
        from tokenspeed.runtime.plugins.server_args import normalize_extra_server_args

        mode = os.environ["HEADLESS_MODE"]
        plugins.ensure_loaded(headless_config=True)
        # The native parser calls ensure_loaded() before reading these choices.
        plugins.ensure_loaded()
        assert ("fixture-profile" in NUMERICS_ENVELOPES) == (mode != "no-plugin")
        if mode != "no-plugin":
            assert normalize_extra_server_args({"fixture_enabled":False, "trace_enabled":True}) == {"fixture_enabled":False, "trace_enabled":True}
        record("ranks")
        if mode == "full-error":
            try:
                plugins.finish_headless_plugins()
            except RuntimeError as error:
                assert "Deferred plugin" in str(error)
            else:
                raise AssertionError("failed deferred plugin was accepted")
            assert "fixture-profile" not in NUMERICS_ENVELOPES
            for name in ("fixture_enabled", "trace_enabled"):
                try:
                    normalize_extra_server_args({name: False})
                except ValueError:
                    pass
                else:
                    raise AssertionError("failed deferred plugin retained its server arguments")
        else:
            plugins.finish_headless_plugins()
            assert NUMERICS_ENVELOPES.count("fixture-profile") == (mode != "no-plugin")
            if mode != "no-plugin":
                assert normalize_extra_server_args({"fixture_enabled":True}) == {"fixture_enabled":True}
        plugins.finish_headless_plugins()
        print(json.dumps(events))
        """)
    result = subprocess.run(
        [sys.executable, "-c", driver],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == expected
