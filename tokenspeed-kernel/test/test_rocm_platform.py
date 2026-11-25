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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import tokenspeed_kernel.platform as platform_module
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement


@pytest.fixture
def detect_amd():
    def detect(arch):
        props = SimpleNamespace(
            gcnArchName=arch,
            name="AMD Radeon AI PRO R9700",
            total_memory=32 * 1024**3,
            multi_processor_count=32,
            max_shared_memory_per_block=64 * 1024,
        )
        with (
            patch("torch.cuda.get_device_properties", return_value=props),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.device_count", return_value=1),
            patch.object(
                platform_module, "_get_rocm_runtime_features", return_value=frozenset()
            ),
        ):
            return platform_module._detect_rocm_platform()

    return detect


@pytest.mark.parametrize("arch", ["gfx1201", "gfx1201:sramecc-:xnack-"])
def test_gfx1201_detection(detect_amd, arch):
    platform = detect_amd(arch)
    assert platform.is_amd
    assert platform.arch_version == ArchVersion(12, 0)
    assert platform.generation_name == "RDNA4"
    assert platform.sm_features == frozenset()
    assert platform.max_shared_memory_per_sm == 64 * 1024
    assert platform.interconnect.topology == "single_gpu"
    assert not platform.is_cdna4
    assert not platform.is_cdna5
    assert not platform.is_cdna4_plus
    assert not platform.is_cdna5_plus


def test_gfx1201_uses_portable_capabilities(detect_amd):
    platform = detect_amd("gfx1201")
    assert CapabilityRequirement(vendors=frozenset({"amd"})).satisfied_by(platform)
    for feature in (
        "tensor_core:f16",
        "tensor_core:f8",
        "tensor_core:f4",
        "memory:async_copy",
    ):
        assert not CapabilityRequirement(
            required_features=frozenset({feature})
        ).satisfied_by(platform)
    for arch in (ArchVersion(9, 5), ArchVersion(12, 5)):
        assert not CapabilityRequirement(
            min_arch_version=arch, max_arch_version=arch
        ).satisfied_by(platform)


@pytest.mark.parametrize(
    "arch,version,generation,cdna5",
    [
        ("gfx950", ArchVersion(9, 5), "CDNA4", False),
        ("gfx1250", ArchVersion(12, 5), "CDNA5", True),
    ],
)
def test_cdna_detection_is_preserved(detect_amd, arch, version, generation, cdna5):
    platform = detect_amd(arch)
    assert platform.arch_version == version
    assert platform.generation_name == generation
    assert platform.is_cdna4_plus
    assert platform.is_cdna5_plus == cdna5
    assert platform.sm_features == frozenset(
        {"tensor_core:f16", "tensor_core:f8", "tensor_core:f4", "memory:async_copy"}
    )


def test_unknown_amd_architecture_is_rejected(detect_amd):
    with pytest.raises(
        RuntimeError, match="unsupported AMD GPU architecture 'gfx9999'"
    ):
        detect_amd("gfx9999")


def test_loaded_hip_runtime_path_reads_the_mapped_library(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f0000000000-7f0000001000 r--p 00000000 00:2a 11 /usr/lib/libc.so.6\n"
        "7f0000002000-7f0000003000 r-xp 00000000 00:00 0 [heap]\n"
        "7f0000004000-7f0000005000 r--p 00000000 00:2a 12 "
        "/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7\n"
    )
    assert (
        platform_module._loaded_hip_runtime_path(str(maps))
        == "/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7"
    )


def test_loaded_hip_runtime_path_without_hip_mapping(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f0000000000-7f0000001000 r--p 00000000 00:2a 11 /usr/lib/libc.so.6\n"
    )
    assert platform_module._loaded_hip_runtime_path(str(maps)) is None
    assert platform_module._loaded_hip_runtime_path(str(tmp_path / "missing")) is None


def test_hip_runtime_prefers_the_mapped_library():
    opened = []

    def cdll(name):
        opened.append(name)
        return SimpleNamespace(hipHostGetDevicePointer=SimpleNamespace())

    platform_module._get_hip_runtime.cache_clear()
    try:
        with (
            patch.object(
                platform_module,
                "_loaded_hip_runtime_path",
                return_value="/rocm/lib/libamdhip64.so.7",
            ),
            patch.object(platform_module.ctypes, "CDLL", side_effect=cdll),
        ):
            platform_module._get_hip_runtime()
    finally:
        platform_module._get_hip_runtime.cache_clear()
    assert opened == ["/rocm/lib/libamdhip64.so.7"]
