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

"""Builds the ``tokenspeed-engine`` wheel from the Cargo workspace.

The wheel carries ``tokenspeed_engine._native`` (``crates/ts-py``) and the
``ts-host`` rank binary (``crates/ts-host``) as ``tokenspeed_engine/bin``,
both built for the interpreter that runs the build: ``ts-host`` embeds it,
so it links that interpreter's libpython, which the extension module leaves
to whoever loads it. Building needs a Rust toolchain (``cargo``) besides
the build requirements in ``pyproject.toml``, and runs from this checkout:
``ts-host`` compiles ``../tokenspeed-scheduler`` into itself.
"""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


def _cargo_build(package: str, extension_module: bool) -> Path:
    """Build ``package`` in release mode; returns the directory it lands in."""
    env = dict(os.environ, PYO3_PYTHON=sys.executable)
    if extension_module:
        env["PYO3_BUILD_EXTENSION_MODULE"] = "1"
    else:
        env.pop("PYO3_BUILD_EXTENSION_MODULE", None)
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "-p", package],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return Path(env.get("CARGO_TARGET_DIR", ROOT / "target")) / "release"


class BuildPyWithRust(build_py):
    """``build_py``, then the two Rust artifacts into the built package."""

    def run(self) -> None:
        super().run()
        # An editable install imports from the source tree, not build_lib.
        root = ROOT / "python" if self.editable_mode else Path(self.build_lib)
        package = root / "tokenspeed_engine"
        release = _cargo_build("tokenspeed-py", extension_module=True)
        shutil.copy2(
            release / "lib_native.so",
            package / f"_native{sysconfig.get_config_var('EXT_SUFFIX')}",
        )
        release = _cargo_build("tokenspeed-host", extension_module=False)
        (package / "bin").mkdir(exist_ok=True)
        shutil.copy2(release / "ts-host", package / "bin" / "ts-host")


class BinaryDistribution(Distribution):
    """A platform wheel: the package carries native code."""

    def has_ext_modules(self) -> bool:
        return True


setup(cmdclass={"build_py": BuildPyWithRust}, distclass=BinaryDistribution)
