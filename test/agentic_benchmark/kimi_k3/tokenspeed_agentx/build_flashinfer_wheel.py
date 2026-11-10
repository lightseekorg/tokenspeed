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

"""Build the FlashInfer 0.7.2 wheel used for the Kimi K3 AgentX results.

Repacks the official flashinfer-python 0.7.2 wheel with the Kimi K3 JIT
sources of upstream FlashInfer: AttnRes and router sources from BASE, and the
latent MoE front/tail backend from FIX (upstream PR #6109). The version stays
0.7.2 for cubin and JIT-cache compatibility; the wheel gets build tag 1k3gb300.

Usage:
    python3 build_flashinfer_wheel.py --source-wheel flashinfer_python-0.7.2-py3-none-any.whl \
        --repo FLASHINFER_CLONE --work NEW_EMPTY_DIR --output OUTPUT_DIR

--source-wheel: the official wheel (checked by SHA256).
--repo: a FlashInfer git clone containing BASE and FIX.
--work: a scratch directory that must not exist yet.
--output: where the wheel and backport-manifest.json are written.
"""

import argparse
import ast
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

BASE = "50a180ed0ae1195305a3c17ab1508875796a2ba5"
FIX = "5dc3cd06c57465c48f6c6f7317f642b6eb683083"
SHA256 = "770550a0aa92b2f10e6c9a1f048dd43713f0f3bb6a12710186fcf8e4e9c99bd8"
PACKAGES = ("kimi_k3_latent_moe", "cake_kimi_k3_attn_res", "kimi_k3_fused_router")
p = argparse.ArgumentParser()
p.add_argument("--source-wheel", type=Path, required=True)
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--work", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert hashlib.sha256(a.source_wheel.read_bytes()).hexdigest() == SHA256
assert not a.work.exists(), a.work
root = a.work / "flashinfer_python-0.7.2"
root.mkdir(parents=True)
a.output.mkdir(parents=True, exist_ok=True)


def git(*args):
    return subprocess.check_output(["git", "-C", str(a.repo), *args])


with zipfile.ZipFile(a.source_wheel) as z:
    for package in PACKAGES:
        path = f"flashinfer/experimental/{package}/cake_jit.py"
        assert z.read(path) == git("show", f"{BASE}:{path}"), path
    path = "flashinfer/experimental/kimi_k3_latent_moe/cake_backend.py"
    assert z.read(path) == git("show", f"{BASE}:{path}")
    for path in (
        "flashinfer/kimi_k3_latent_moe.py",
        "flashinfer/kimi_k3_attn_res.py",
        "flashinfer/fused_moe/kimi_k3_fused_router.py",
    ):
        assert z.read(path) == git("show", f"{FIX}:{path}"), path
    z.extractall(root)

for package in PACKAGES:
    path = f"flashinfer/experimental/{package}"
    revision = FIX if package == "kimi_k3_latent_moe" else BASE
    if package != "kimi_k3_latent_moe":
        path += "/csrc"
    blob = git("archive", revision, path)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        archive.extractall(root, filter="data")

for package in PACKAGES:
    package_dir = root / "flashinfer/experimental" / package
    module_tree = ast.parse((package_dir / "cake_jit.py").read_text())
    assignment = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "MODULES"
    )
    registry = ast.literal_eval(assignment.value)
    for record in registry.values():
        for source in record["sources"]:
            assert (package_dir / "csrc" / source).is_file(), source

subprocess.run(
    [
        sys.executable,
        "-m",
        "wheel",
        "pack",
        "--build-number",
        "1k3gb300",
        "--dest-dir",
        str(a.output),
        str(root),
    ],
    check=True,
)
wheel = a.output / "flashinfer_python-0.7.2-1k3gb300-py3-none-any.whl"
with zipfile.ZipFile(a.source_wheel) as before, zipfile.ZipFile(wheel) as after:
    original = {
        name: before.read(name) for name in before.namelist() if not name.endswith("/")
    }
    updated = {
        name: after.read(name) for name in after.namelist() if not name.endswith("/")
    }
    deleted = sorted(original.keys() - updated.keys())
    assert not deleted, deleted
    changed = sorted(name for name in original if original[name] != updated[name])
    added = sorted(updated.keys() - original.keys())
    allowed = "flashinfer/experimental/kimi_k3_latent_moe/"
    metadata = {
        "flashinfer_python-0.7.2.dist-info/RECORD",
        "flashinfer_python-0.7.2.dist-info/WHEEL",
    }
    assert all(
        name.startswith(allowed) or name in metadata for name in changed
    ), changed
    assert all(
        any(
            name.startswith(f"flashinfer/experimental/{package}/")
            for package in PACKAGES
        )
        for name in added
    ), added
manifest = dict(
    base_wheel=str(a.source_wheel),
    base_sha256=SHA256,
    matching_registry_revision=BASE,
    moe_backport_revision=FIX,
    wheel=str(wheel),
    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    modified_files=changed,
    added_files=added,
    unrelated_files_changed=False,
)
(a.output / "backport-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            k: manifest[k]
            for k in (
                "wheel",
                "wheel_sha256",
                "modified_files",
                "unrelated_files_changed",
            )
        },
        indent=2,
    )
)
