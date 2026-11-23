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

"""Build the frontend and host extensions with one CPython ABI contract.

Python 3.11 uses its native ABI; Python 3.12+ builds target the 3.12 stable
ABI. The host uses the interpreter importing it, so no libpython is bundled.
"""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
ABI3 = sys.version_info >= (3, 12)
if sysconfig.get_config_var("Py_GIL_DISABLED"):
    raise RuntimeError("tokenspeed-engine requires a standard GIL-enabled CPython")


def _cargo_build(package: str) -> Path:
    """Build one extension with the same ABI mode as the wheel."""
    env = dict(os.environ, PYO3_PYTHON=sys.executable, PYO3_BUILD_EXTENSION_MODULE="1")
    command = ["cargo", "build", "--release", "--locked", "-p", package]
    if ABI3:
        command += ["--features", "abi3"]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return Path(env.get("CARGO_TARGET_DIR", ROOT / "target")) / "release"


class BuildPyWithRust(build_py):
    """``build_py``, then the two Rust artifacts into the built package."""

    def run(self) -> None:
        super().run()
        # An editable install imports from the source tree, not build_lib.
        root = ROOT / "python" if self.editable_mode else Path(self.build_lib)
        package = root / "tokenspeed_engine"
        suffix = ".abi3.so" if ABI3 else sysconfig.get_config_var("EXT_SUFFIX")
        for crate, module in (
            ("tokenspeed-py", "_native"),
            ("tokenspeed-host", "_host"),
        ):
            release = _cargo_build(crate)
            shutil.copy2(release / f"lib{module}.so", package / f"{module}{suffix}")
        (package / "bin").mkdir(exist_ok=True)
        shutil.copy2(release / "ts-host", package / "bin" / "ts-host")


class EngineWheel(bdist_wheel):
    """Advertise abi3 only when both extensions were compiled for it."""

    def finalize_options(self) -> None:
        super().finalize_options()
        if ABI3:
            self.py_limited_api = "cp312"


class BinaryDistribution(Distribution):
    """A platform wheel: the package carries native code."""

    def has_ext_modules(self) -> bool:
        return True


setup(
    cmdclass={"build_py": BuildPyWithRust, "bdist_wheel": EngineWheel},
    distclass=BinaryDistribution,
)
