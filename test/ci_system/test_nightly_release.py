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

import runpy
import shutil
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PREPARE = runpy.run_path(str(ROOT / ".github/scripts/prepare-tokenspeed-nightly.py"))[
    "prepare_nightly"
]
UPDATE = runpy.run_path(str(ROOT / ".github/scripts/update-nightly-index.py"))[
    "update_index"
]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "python/tokenspeed").mkdir(parents=True)
    for name in ("python/pyproject.toml", "python/tokenspeed/version.py"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def test_nightly_versions_and_kernel_dependency_share_date(project):
    base_version = tomllib.loads((project / "python/pyproject.toml").read_text())[
        "project"
    ]["version"]
    version = PREPARE(project, "20260930", "0.1.3.post20260930")
    metadata = tomllib.loads((project / "python/pyproject.toml").read_text())["project"]
    assert metadata["version"] == version == f"{base_version}.post20260930"
    kernel = [d for d in metadata["dependencies"] if d.startswith("tokenspeed-kernel")]
    assert kernel == ["tokenspeed-kernel==0.1.3.post20260930"]
    assert (
        runpy.run_path(str(project / "python/tokenspeed/version.py"))["__version__"]
        == version
    )


def test_release_candidate_kernel_nightly_makes_a_tokenspeed_release_candidate(
    project,
):
    base_version = tomllib.loads((project / "python/pyproject.toml").read_text())[
        "project"
    ]["version"]
    version = PREPARE(project, "20261108", "0.1.4rc1.post20261108")
    metadata = tomllib.loads((project / "python/pyproject.toml").read_text())["project"]
    assert metadata["version"] == version == f"{base_version}rc1.post20261108"
    pins = [d for d in metadata["dependencies"] if d.startswith("tokenspeed-kernel")]
    assert pins == ["tokenspeed-kernel==0.1.4rc1.post20261108"]
    assert (
        runpy.run_path(str(project / "python/tokenspeed/version.py"))["__version__"]
        == version
    )


@pytest.mark.parametrize(
    "date,kernel",
    [("20260230", "0.1.3.post20260230"), ("20260930", "0.1.3.post20260929")],
)
def test_invalid_or_mismatched_date_does_not_change_metadata(project, date, kernel):
    path = project / "python/pyproject.toml"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        PREPARE(project, date, kernel)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "kernel",
    [
        "0.1.4.post20261108.dev1",
        "0.1.4.post20261108+exp1",
        "0.1.4a1.post20261108",
        "0.1.4rc1.post20261108.dev1",
    ],
)
def test_kernel_with_anything_but_a_release_candidate_is_refused(project, kernel):
    path = project / "python/pyproject.toml"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        PREPARE(project, "20261108", kernel)
    assert path.read_bytes() == original


@pytest.mark.parametrize("variant", ["cu130", "rocm72"])
def test_tokenspeed_index_preserves_kernel_and_history(tmp_path, variant):
    wheelhouse = tmp_path / "wheelhouse"
    nightly = wheelhouse / "nightly"
    if variant == "rocm72":
        nightly /= "rocm7.2"
    (nightly / "tokenspeed-kernel").mkdir(parents=True)
    kernel = nightly / "tokenspeed-kernel/index.html"
    kernel.write_text("existing kernel links\n")
    (nightly / "index.html").write_text(
        '<a href="tokenspeed-kernel/">tokenspeed-kernel</a><br>\n'
    )
    (nightly / "tokenspeed").mkdir()
    index = nightly / "tokenspeed/index.html"
    index.write_text("older tokenspeed nightly\n")
    dist = tmp_path / "dist/tokenspeed-dist"
    dist.mkdir(parents=True)
    name = "tokenspeed-0.1.0.post20260930-py3-none-any.whl"
    (dist / name).write_bytes(b"wheel")
    url = f"https://github.com/lightseekorg/whl/releases/download/tokenspeed-nightly-v0.1.0.post20260930/{name}"
    release = {
        "assets": [
            {"name": name, "browser_download_url": url, "digest": f"sha256:{'a' * 64}"}
        ]
    }
    UPDATE(wheelhouse, release, dist.parent, "tokenspeed", variant)
    contents = index.read_text()
    assert contents.startswith("older tokenspeed nightly\n")
    assert f'{url}#sha256={"a" * 64}' in contents
    assert kernel.read_text() == "existing kernel links\n"
    assert 'href="tokenspeed-kernel/"' in (nightly / "index.html").read_text()
    assert 'href="tokenspeed/"' in (nightly / "index.html").read_text()
    UPDATE(wheelhouse, release, dist.parent, "tokenspeed", variant)
    assert index.read_text() == contents
    if variant == "rocm72":
        assert 'href="rocm7.2/"' in (wheelhouse / "nightly/index.html").read_text()
        assert not (wheelhouse / "rocm7.2/nightly").exists()

    release["assets"][0]["digest"] = f"sha256:{'b' * 64}"
    UPDATE(wheelhouse, release, dist.parent, "tokenspeed", variant, replace=True)
    updated = index.read_text()
    assert updated.startswith("older tokenspeed nightly\n")
    assert f'{url}?sha256={"b" * 64}#sha256={"b" * 64}' in updated
    assert f'#sha256={"a" * 64}' not in updated
    assert updated.count(f">{name}</a>") == 1
    assert kernel.read_text() == "existing kernel links\n"
    UPDATE(wheelhouse, release, dist.parent, "tokenspeed", variant, replace=True)
    assert index.read_text() == updated


@pytest.mark.parametrize(
    "variant,version,roots",
    [
        ("engine-release", "0.1.0", [""]),
        ("engine-nightly", "0.1.0rc20261112", ["nightly", "nightly/rocm7.2"]),
    ],
)
def test_engine_indexes_preserve_history_and_reject_replacement(
    tmp_path, variant, version, roots
):
    wheelhouse = tmp_path / "wheelhouse"
    dist = tmp_path / "dist"
    dist.mkdir()
    name = f"tokenspeed_engine-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
    (dist / name).write_bytes(b"wheel")
    url = f"https://github.com/lightseekorg/whl/releases/download/tokenspeed-engine-v{version}/{name}"
    release = {
        "assets": [
            {"name": name, "browser_download_url": url, "digest": f"sha256:{'a' * 64}"}
        ]
    }
    for directory in roots:
        root = wheelhouse / directory
        (root / "tokenspeed-engine").mkdir(parents=True)
        (root / "tokenspeed-engine/index.html").write_text("older engine links\n")
        (root / "index.html").write_text('<a href="existing/">existing</a><br>\n')

    paths = UPDATE(wheelhouse, release, dist, "tokenspeed-engine", variant)
    snapshot = {path: (wheelhouse / path).read_text() for path in paths}
    for directory in roots:
        root = wheelhouse / directory
        contents = (root / "tokenspeed-engine/index.html").read_text()
        assert contents.startswith("older engine links\n")
        assert f'{url}#sha256={"a" * 64}' in contents
        assert 'href="tokenspeed-engine/"' in (root / "index.html").read_text()
        assert 'href="existing/"' in (root / "index.html").read_text()
    if variant == "engine-nightly":
        assert 'href="rocm7.2/"' in (wheelhouse / "nightly/index.html").read_text()
        assert not (wheelhouse / "tokenspeed-engine").exists()
    else:
        assert not (wheelhouse / "nightly").exists()

    UPDATE(wheelhouse, release, dist, "tokenspeed-engine", variant)
    assert snapshot == {path: (wheelhouse / path).read_text() for path in paths}
    release["assets"][0]["digest"] = f"sha256:{'b' * 64}"
    with pytest.raises(ValueError, match="digest changed"):
        UPDATE(wheelhouse, release, dist, "tokenspeed-engine", variant)
    with pytest.raises(ValueError, match="must not be replaced"):
        UPDATE(wheelhouse, release, dist, "tokenspeed-engine", variant, replace=True)
    assert snapshot == {path: (wheelhouse / path).read_text() for path in paths}
