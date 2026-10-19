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
_NIGHTLY = runpy.run_path(str(ROOT / ".github/scripts/prepare-tokenspeed-nightly.py"))
PREPARE = _NIGHTLY["prepare_nightly"]
NEWEST_SMG = _NIGHTLY["newest_smg_nightly"]
SMG_0930 = {
    "tokenspeed-smg": "1.13.0.post20260930",
    "tokenspeed-smg-grpc-proto": "0.4.27.post20260930",
    "tokenspeed-smg-grpc-servicer": "0.14.0.post20260930",
}
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
    version = PREPARE(project, "20260930", "0.1.3.post20260930", SMG_0930)
    metadata = tomllib.loads((project / "python/pyproject.toml").read_text())["project"]
    assert metadata["version"] == version == f"{base_version}.post20260930"
    kernel = [d for d in metadata["dependencies"] if d.startswith("tokenspeed-kernel")]
    assert kernel == ["tokenspeed-kernel==0.1.3.post20260930"]
    smg = sorted(d for d in metadata["dependencies"] if d.startswith("tokenspeed-smg"))
    assert smg == sorted(f"{name}=={v}" for name, v in SMG_0930.items())
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
        PREPARE(project, date, kernel, SMG_0930)
    assert path.read_bytes() == original


def test_newest_smg_nightly_comes_from_one_build_on_or_before_the_date():
    releases = {
        # 20261001 lacks the servicer; 20261003 is after the nightly date.
        "tokenspeed-smg": [
            "1.12.0",
            "1.12.0.post20260929",
            "1.13.0.post20260930",
            "1.13.0.post20261001",
            "1.13.0.post20261003",
        ],
        "tokenspeed-smg-grpc-proto": [
            "0.4.26.post20260929",
            "0.4.27.post20260930",
            "0.4.27.post20261001",
            "0.4.27.post20261003",
        ],
        "tokenspeed-smg-grpc-servicer": [
            "0.13.0.post1",
            "0.14.1.post20260930.dev0",
            "0.13.6.post20260929",
            "0.14.0.post20260930",
            "0.14.0.post20261003",
        ],
    }
    assert NEWEST_SMG(releases, "20261002") == SMG_0930
    assert NEWEST_SMG(releases, "20260929") == {
        "tokenspeed-smg": "1.12.0.post20260929",
        "tokenspeed-smg-grpc-proto": "0.4.26.post20260929",
        "tokenspeed-smg-grpc-servicer": "0.13.6.post20260929",
    }
    with pytest.raises(ValueError):
        NEWEST_SMG(releases, "20260928")


@pytest.mark.parametrize(
    "smg",
    [
        {**SMG_0930, "tokenspeed-smg": "1.13.0.post20261001"},
        {**SMG_0930, "tokenspeed-smg-grpc-proto": "0.4.27.post20260929"},
        {**SMG_0930, "tokenspeed-smg-grpc-servicer": "0.14.0"},
        {name: v.replace("20260930", "20261001") for name, v in SMG_0930.items()},
    ],
)
def test_future_or_mixed_smg_nightly_does_not_change_metadata(project, smg):
    path = project / "python/pyproject.toml"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        PREPARE(project, "20260930", "0.1.3.post20260930", smg)
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
