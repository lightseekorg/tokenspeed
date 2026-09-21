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

"""CPU-only regression tests for release artifact validation."""

import zipfile

import pytest
from scripts.check_release import check_release, read_version


@pytest.fixture
def release(tmp_path):
    package_dir = tmp_path / "package"
    source_dir = package_dir / "python" / "tokenspeed_mla"
    source_dir.mkdir(parents=True)
    (package_dir / "pyproject.toml").write_text(
        '[project]\nname = "tokenspeed-mla"\nversion = "1.2.3"\n'
    )
    entries = {}
    for name in ("__init__.py", "fmha.py", "mla_prefill.py", "mla_decode.py"):
        path = source_dir / name
        path.write_text(f'"""{name}"""\n')
        entries[f"tokenspeed_mla/{name}"] = path.read_bytes()
    for name in ("LICENSE", "THIRDPARTYNOTICES"):
        (package_dir / name).write_text(f"Public {name}\n")
        entries[f"tokenspeed_mla-1.2.3.dist-info/licenses/{name}"] = (
            package_dir / name
        ).read_bytes()
    entries["tokenspeed_mla-1.2.3.dist-info/METADATA"] = (
        b"Metadata-Version: 2.4\nName: tokenspeed-mla\nVersion: 1.2.3\n"
    )
    entries["tokenspeed_mla-1.2.3.dist-info/WHEEL"] = (
        b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    return package_dir, dist_dir, entries


def write_wheel(dist_dir, entries):
    wheel_path = dist_dir / "tokenspeed_mla-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for name, content in entries.items():
            wheel.writestr(name, content)
    return wheel_path


def test_accepts_source_only_wheel(release):
    package_dir, dist_dir, entries = release
    wheel_path = write_wheel(dist_dir, entries)
    assert check_release(package_dir, dist_dir) == wheel_path


@pytest.mark.parametrize(
    "entry",
    [
        "tokenspeed_mla/mla_decode.py",
        "tokenspeed_mla-1.2.3.dist-info/licenses/LICENSE",
        "tokenspeed_mla-1.2.3.dist-info/licenses/THIRDPARTYNOTICES",
    ],
)
def test_rejects_missing_source_or_notice(release, entry):
    package_dir, dist_dir, entries = release
    del entries[entry]
    write_wheel(dist_dir, entries)
    with pytest.raises(ValueError):
        check_release(package_dir, dist_dir)


@pytest.mark.parametrize(
    "entry, content",
    [
        ("tokenspeed_mla/mla_decode.py", b"# stale source\n"),
        ("tokenspeed_mla/fmha_binary.py", b"# obsolete backend\n"),
        ("tokenspeed_mla/objs/kernel.so", b"\x7fELF"),
        ("kernel.cubin", b"\x7fELF"),
        (
            "tokenspeed_mla-1.2.3.dist-info/METADATA",
            b"Name: other-package\nVersion: 1.2.3\n",
        ),
        (
            "tokenspeed_mla-1.2.3.dist-info/METADATA",
            b"Name: tokenspeed-mla\nVersion: 1.2.2\n",
        ),
        (
            "tokenspeed_mla-1.2.3.dist-info/WHEEL",
            b"Root-Is-Purelib: false\nTag: cp312-none-any\n",
        ),
        ("tokenspeed_mla-1.2.3.dist-info/licenses/LICENSE", b"wrong notice\n"),
    ],
)
def test_rejects_invalid_wheel_contents(release, entry, content):
    package_dir, dist_dir, entries = release
    entries[entry] = content
    write_wheel(dist_dir, entries)
    with pytest.raises(ValueError):
        check_release(package_dir, dist_dir)


def test_rejects_platform_wheel(release):
    package_dir, dist_dir, entries = release
    write_wheel(dist_dir, entries).rename(
        dist_dir / "tokenspeed_mla-1.2.3-cp312-cp312-linux_x86_64.whl"
    )
    with pytest.raises(ValueError, match="exactly one artifact"):
        check_release(package_dir, dist_dir)


def test_rejects_multiple_artifacts(release):
    package_dir, dist_dir, entries = release
    write_wheel(dist_dir, entries)
    (dist_dir / "stale.whl").write_bytes(b"")
    with pytest.raises(ValueError, match="exactly one artifact"):
        check_release(package_dir, dist_dir)


def test_rejects_local_release_version(release):
    package_dir, _, _ = release
    (package_dir / "pyproject.toml").write_text(
        '[project]\nname = "tokenspeed-mla"\nversion = "1.2.3+local"\n'
    )
    with pytest.raises(ValueError, match="public release version"):
        read_version(package_dir)
