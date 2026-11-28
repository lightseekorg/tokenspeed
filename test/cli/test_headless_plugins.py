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
        ("split", ["cli", "trace", "metrics", "ranks", "kernels", "full"]),
        ("unknown", ["kernels", "full", "trace", "metrics", "other", "ranks"]),
        ("cli-error", ["cli", "kernels", "full", "trace", "metrics", "ranks"]),
        ("full-error", ["cli", "trace", "metrics", "ranks", "kernels", "full"]),
    ],
)
def test_headless_plugin_registration(tmp_path, mode, expected):
    companion = tmp_path / "tokenspeed_longcat"
    companion.mkdir()
    (companion / "__init__.py").write_text(dedent("""
            import os
            from fixture_events import record
            from tokenspeed.runtime.configs.numerics import register_numerics_envelope

            def add_numerics():
                register_numerics_envelope(
                    "fixture-profile", bitwise=False,
                    resolve=lambda args: None, validate=lambda **args: None,
                )

            def register_cli_options():
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
                del register_cli_options
            """))
    (companion / "tracing.py").write_text(
        'from fixture_events import record\ndef register():\n    record("trace")\n'
    )
    (tmp_path / "tokenspeed_metrics_pycat.py").write_text(
        'from fixture_events import record\ndef register():\n    record("metrics")\n'
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
            "longcat = tokenspeed_longcat:register",
            "longcat-eval = tokenspeed_longcat.tracing:register",
            "pycat = tokenspeed_metrics_pycat:register",
        ]
        if mode == "unknown":
            entries.append("z_other = other_plugin:register")
        entries.extend(
            [
                "[tokenspeed_kernel.plugins]",
                "longcat = tokenspeed_longcat.kernels:register",
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

        mode = os.environ["HEADLESS_MODE"]
        plugins.ensure_loaded(headless_config=True)
        # The native parser calls ensure_loaded() before reading these choices.
        plugins.ensure_loaded()
        assert ("fixture-profile" in NUMERICS_ENVELOPES) == (mode != "no-plugin")
        record("ranks")
        if mode == "full-error":
            try:
                plugins.finish_headless_plugins()
            except RuntimeError as error:
                assert "Deferred plugin" in str(error)
            else:
                raise AssertionError("failed deferred plugin was accepted")
            assert "fixture-profile" not in NUMERICS_ENVELOPES
        else:
            plugins.finish_headless_plugins()
            assert NUMERICS_ENVELOPES.count("fixture-profile") == (mode != "no-plugin")
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
