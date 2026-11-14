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

import importlib.metadata
from test.ci_system.install_deps_cu129 import validate_cuda_distributions

import pytest


def installed_distributions(tmp_path, packages):
    for index, (name, version) in enumerate(packages):
        metadata = tmp_path / f"package{index}.dist-info" / "METADATA"
        metadata.parent.mkdir()
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
    return importlib.metadata.distributions(path=[str(tmp_path)])


@pytest.mark.parametrize(
    "name", ["nvidia-cutlass-dsl-libs-cu13", "NVIDIA_cutlass.dsl-libs_cu13"]
)
def test_accepts_cutlass_companion_with_cuda12(tmp_path, name):
    packages = [
        (name, "4.8.0"),
        ("nvidia-cutlass-dsl-libs-cu12", "4.8.0"),
        ("nvidia-cuda-runtime-cu12", "12.6.77"),
        ("cuda-bindings", "12.9.7"),
        ("torch", "2.14.0+cu126"),
    ]
    validate_cuda_distributions(installed_distributions(tmp_path, packages))


@pytest.mark.parametrize(
    "name,version",
    [
        ("nvidia-cutlass-dsl-libs-cu13", "4.7.0"),
        ("nvidia-cutlass-dsl-libs-cu13", "4.9.0"),
        ("nvidia-cuda-runtime-cu13", "13.0.0"),
        ("nvidia-cudnn-cu13", "9.20.0"),
        ("nvidia-nccl-cu13", "2.29.7"),
        ("nvidia-nvshmem-cu13", "3.4.0"),
        ("nvidia-cublas", "13.0.0"),
        ("nvidia-cuda-runtime", "13.0.0"),
        ("nvidia-cuda-nvrtc", "13.0.0"),
        ("tokenspeed-cutedsl-kda", "0.1.0"),
        ("cuda-toolkit", "13.0.0"),
        ("cuda-python", "13.0.0"),
        ("cuda-bindings", "13.0.0"),
        ("torch", "2.14.0+cu130"),
        ("torchvision", "0.29.0+cu130"),
    ],
)
def test_rejects_cuda13_conflicts(tmp_path, name, version):
    packages = [("nvidia-cutlass-dsl-libs-cu13", "4.8.0"), (name, version)]
    with pytest.raises(RuntimeError) as error:
        validate_cuda_distributions(installed_distributions(tmp_path, packages))
    assert f"{name}=={version}" in str(error.value)
    assert "nvidia-cutlass-dsl-libs-cu13==4.8.0" not in str(error.value)
