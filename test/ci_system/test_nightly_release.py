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
    version = PREPARE(project, "20260930", "0.1.3.post20260930")
    metadata = tomllib.loads((project / "python/pyproject.toml").read_text())["project"]
    assert metadata["version"] == version == "0.1.0.post20260930"
    kernel = [d for d in metadata["dependencies"] if d.startswith("tokenspeed-kernel")]
    assert kernel == ["tokenspeed-kernel==0.1.3.post20260930"]
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
