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
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/scripts/update-nightly-index.py"


@pytest.mark.parametrize("variant", ["cu130", "rocm72"])
def test_nightly_index_preserves_history_and_uses_published_digest(
    tmp_path, variant
) -> None:
    update_index = runpy.run_path(str(SCRIPT))["update_index"]
    wheelhouse = tmp_path / "wheelhouse"
    nightly = wheelhouse / "nightly"
    if variant == "rocm72":
        nightly /= "rocm7.2"
    nightly.mkdir(parents=True)
    root = nightly / "index.html"
    root.write_text('<!DOCTYPE html>\n<a href="cu130/">cu130</a><br>\n')
    project = nightly / "tokenspeed-kernel" / "index.html"
    project.parent.mkdir()
    history = '<!DOCTYPE html>\n<a href="older.whl">older.whl</a><br>\n'
    project.write_text(history)
    artifact = (
        "tokenspeed-kernel-wheel-cu130-x64-py3.12"
        if variant == "cu130"
        else "tokenspeed-kernel-rocm72-wheel-x64-py3.12"
    )
    wheels = tmp_path / "dist" / artifact
    wheels.mkdir(parents=True)
    name = "tokenspeed_kernel-0.1.3.post20260929-cp312-cp312-manylinux_2_28_x86_64.whl"
    (wheels / name).write_bytes(b"rebuilt contents differ from the published wheel")
    url = f"https://github.com/lightseekorg/whl/releases/download/nightly/{name}"
    release = {
        "assets": [
            {"name": name, "browser_download_url": url, "digest": f"sha256:{'a' * 64}"}
        ]
    }
    update_index(wheelhouse, release, wheels.parent, "tokenspeed-kernel", variant)
    contents = project.read_text()
    assert contents.startswith(history)
    assert f'{url}#sha256={"a" * 64}' in contents
    assert 'href="tokenspeed-kernel/"' in root.read_text()
    assert 'href="cu130/"' in root.read_text()
    update_index(wheelhouse, release, wheels.parent, "tokenspeed-kernel", variant)
    assert project.read_text() == contents
    if variant == "rocm72":
        assert 'href="rocm7.2/"' in (wheelhouse / "nightly/index.html").read_text()
        assert not (wheelhouse / "rocm7.2/nightly").exists()

    with pytest.raises(ValueError, match="missing expected nightly wheels"):
        update_index(
            wheelhouse, {"assets": []}, wheels.parent, "tokenspeed-kernel", variant
        )
    assert project.read_text() == contents


def test_explicit_replacement_refreshes_hash_and_download_url(tmp_path) -> None:
    update_index = runpy.run_path(str(SCRIPT))["update_index"]
    wheelhouse = tmp_path / "wheelhouse"
    wheels = tmp_path / "dist" / "tokenspeed-kernel-wheel-cu130-x64-py3.12"
    wheels.mkdir(parents=True)
    name = "tokenspeed_kernel-0.1.3.post20260930-cp312-cp312-manylinux_2_28_x86_64.whl"
    (wheels / name).write_bytes(b"corrected wheel")
    url = f"https://github.com/lightseekorg/whl/releases/download/nightly/{name}"
    release = {
        "assets": [
            {"name": name, "browser_download_url": url, "digest": f"sha256:{'a' * 64}"}
        ]
    }
    update_index(wheelhouse, release, wheels.parent, "tokenspeed-kernel", "cu130")
    index = wheelhouse / "nightly/tokenspeed-kernel/index.html"
    original = index.read_text()
    release["assets"][0]["digest"] = f"sha256:{'b' * 64}"
    with pytest.raises(ValueError, match="replacement required"):
        update_index(wheelhouse, release, wheels.parent, "tokenspeed-kernel", "cu130")
    assert index.read_text() == original
    update_index(
        wheelhouse, release, wheels.parent, "tokenspeed-kernel", "cu130", replace=True
    )
    updated = index.read_text()
    assert f'?sha256={"b" * 64}#sha256={"b" * 64}' in updated
    assert f'#sha256={"a" * 64}' not in updated
    assert updated.count(f">{name}</a>") == 1
    update_index(wheelhouse, release, wheels.parent, "tokenspeed-kernel", "cu130")
    assert index.read_text() == updated
