# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU-only contracts for the shared AMD extern intrinsic libraries."""

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

_AMD = Path(__file__).resolve().parents[3] / "tokenspeed-kernel-amd"
_PACKAGE = "tokenspeed_kernel_amd"
_DIRECTORY = _AMD / "python" / _PACKAGE


@pytest.fixture
def extern():
    pytest.importorskip(
        "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
    )
    spec = importlib.util.spec_from_file_location(
        "amd_extern_policy_test", _DIRECTORY / "_extern.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_intrinsic_library_is_package_data(extern):
    libraries = extern.extern_compile_options()["extern_libs"]
    paths = [Path(path) for path in libraries.values()]
    assert sorted(paths) == sorted((_DIRECTORY / "_intrinsics").glob("*.ll"))
    config = tomllib.loads((_AMD / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"][_PACKAGE]
    for path in paths:
        assert any(path.match(pattern) for pattern in patterns), path


def test_symbols_carry_their_library_name(extern):
    # Triton links a library only for calls whose symbol contains its name.
    for name, path in extern.extern_compile_options()["extern_libs"].items():
        text = Path(path).read_text()
        symbols = set(re.findall(r"^define \S+ @(\w+)", text, re.M))
        assert symbols, path
        for symbol in symbols:
            assert symbol.startswith(f"__{name}_"), (path, symbol)


def test_changed_library_changes_compile_key(extern, tmp_path):
    compiler = pytest.importorskip("tokenspeed_triton.backends.amd.compiler")
    libraries = {}
    for name, source in extern.extern_compile_options()["extern_libs"].items():
        path = tmp_path / Path(source).name
        path.write_bytes(Path(source).read_bytes())
        libraries[name] = path
    options = compiler.HIPOptions(
        arch="gfx950",
        extern_libs={name: str(path) for name, path in libraries.items()},
    )
    try:
        for path in libraries.values():
            before = options.hash()
            path.write_bytes(path.read_bytes() + b"; changed contents, same path\n")
            # Triton memoizes file hashes per process; start cold as a new process.
            compiler.file_hash.cache_clear()
            assert options.hash() != before, path
    finally:
        compiler.file_hash.cache_clear()


def test_compile_options_are_not_shared_mutable_state(extern):
    options = extern.extern_compile_options()
    expected = {"extern_libs": dict(options["extern_libs"])}
    options["extern_libs"].clear()
    assert extern.extern_compile_options() == expected


def test_sched_group_mask_ors_named_classes(extern):
    assert extern._sched_group_mask("mfma") == 0x8
    assert extern._sched_group_mask(("valu", "trans")) == 0x402
    with pytest.raises(ValueError, match="unknown sched_group class"):
        extern._sched_group_mask(("mfma", "lds"))
