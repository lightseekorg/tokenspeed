"""Validate native TRT-LLM MoE wheel and source-distribution contents."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZipFile

_PACKAGE_PREFIX = "tokenspeed_kernel/thirdparty/trtllm_native_moe/"
_RUNTIME_FILES = {
    "__init__.py",
    "LICENSE.CUTLASS",
    "LICENSE.NVIDIA",
    "LICENSE.PYBIND11",
    "NOTICE",
    "objs/libtokenspeed_trtllm_native_moe.so",
}
_SOURCE_FILES = {
    "CMakeLists.txt",
    "LICENSE.CUTLASS",
    "LICENSE.NVIDIA",
    "LICENSE.PYBIND11",
    "NOTICE",
    "__init__.py",
    "tactic_abi.cpp",
}
_SOURCE_SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".json", ".zst"}
_BATCHED_GEMM_EXPORT = (
    "csrc/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/" "trtllmGen_bmm_export"
)
_CUBIN_ARCHIVE_DIR = f"{_BATCHED_GEMM_EXPORT}/cubins"
_KERNEL_META_INFO = f"{_BATCHED_GEMM_EXPORT}/KernelMetaInfo.h"
_CUBIN_ARCHIVE_SUFFIX = ".cubin.tar.zst"
_EXPECTED_CUBIN_COUNT = 85
_EXPECTED_CUBIN_FAMILIES = (
    "Bmm_MxE4m3_MxE2m1MxE4m3_Fp32_",
    "Bmm_Bfloat16_MxE2m1MxE4m3_Fp32_",
)
_KERNEL_META_DATA_SYMBOL = re.compile(
    r"^extern unsigned char const ([A-Za-z0-9_]+)_cubin\[\];$", re.MULTILINE
)
_KERNEL_META_LENGTH_SYMBOL = re.compile(
    r"^extern unsigned int const ([A-Za-z0-9_]+)_cubin_len;$", re.MULTILINE
)
_KERNEL_META_CONFIG_ENTRY = re.compile(
    r"^\s*\{(Bmm_[A-Za-z0-9_]+)_cubin,", re.MULTILINE
)


def _native_relative_path(archive_path: str) -> str | None:
    marker = f"/{_PACKAGE_PREFIX}"
    if marker in archive_path:
        return archive_path.split(marker, 1)[1]
    if archive_path.startswith(_PACKAGE_PREFIX):
        return archive_path[len(_PACKAGE_PREFIX) :]
    return None


def _validate_wheel(path: Path) -> None:
    with ZipFile(path) as archive:
        infos = {
            relative: info
            for info in archive.infolist()
            if (relative := _native_relative_path(info.filename)) is not None
            and not info.is_dir()
        }

    actual = set(infos)
    if actual != _RUNTIME_FILES:
        missing = sorted(_RUNTIME_FILES - actual)
        unexpected = sorted(actual - _RUNTIME_FILES)
        raise RuntimeError(
            f"invalid native MoE wheel contents; missing={missing}, "
            f"unexpected={unexpected[:20]}"
        )

    library = infos["objs/libtokenspeed_trtllm_native_moe.so"]
    if library.file_size <= 1024 * 1024:
        raise RuntimeError(f"native MoE DSO is unexpectedly small: {library.file_size}")


def _expected_sdist_files(source_root: Path) -> set[str]:
    native_root = source_root / _PACKAGE_PREFIX
    expected = set(_SOURCE_FILES)
    expected.update(
        path.relative_to(native_root).as_posix()
        for path in (native_root / "cmake").rglob("*.cmake")
        if path.is_file()
    )
    expected.update(
        path.relative_to(native_root).as_posix()
        for path in (native_root / "csrc").rglob("*")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES
    )
    return expected


def _cubin_archive_stem(filename: str) -> str:
    if not filename.endswith(_CUBIN_ARCHIVE_SUFFIX):
        raise RuntimeError(f"invalid native MoE cubin archive name: {filename}")
    return filename[: -len(_CUBIN_ARCHIVE_SUFFIX)]


def _kernel_meta_symbols(contents: str) -> tuple[set[str], set[str], set[str]]:
    inventories = []
    for label, pattern in (
        ("data symbols", _KERNEL_META_DATA_SYMBOL),
        ("length symbols", _KERNEL_META_LENGTH_SYMBOL),
        ("config entries", _KERNEL_META_CONFIG_ENTRY),
    ):
        symbols = pattern.findall(contents)
        unique_symbols = set(symbols)
        if len(symbols) != len(unique_symbols):
            raise RuntimeError(
                f"duplicate cubin {label} in native MoE KernelMetaInfo.h"
            )
        inventories.append(unique_symbols)
    return inventories[0], inventories[1], inventories[2]


def _validate_cubin_inventory(
    archive_stems: set[str],
    data_symbols: set[str],
    length_symbols: set[str],
    config_entries: set[str],
    *,
    origin: str,
) -> None:
    problems = []

    for label, symbols in (
        ("data symbols", data_symbols),
        ("length symbols", length_symbols),
        ("config entries", config_entries),
    ):
        missing_metadata = sorted(archive_stems - symbols)
        missing_archives = sorted(symbols - archive_stems)
        if missing_metadata or missing_archives:
            problems.append(
                f"archive/{label} mismatch: "
                f"archives_without_{label.replace(' ', '_')}={missing_metadata[:10]}, "
                f"{label.replace(' ', '_')}_without_archives={missing_archives[:10]}"
            )

    unexpected_families = sorted(
        stem for stem in archive_stems if not stem.startswith(_EXPECTED_CUBIN_FAMILIES)
    )
    if unexpected_families:
        problems.append(f"unexpected cubin families={unexpected_families[:10]}")

    if len(archive_stems) != _EXPECTED_CUBIN_COUNT:
        problems.append(
            f"expected {_EXPECTED_CUBIN_COUNT} cubin archives, "
            f"found {len(archive_stems)}"
        )

    if problems:
        details = "; ".join(problems)
        raise RuntimeError(f"invalid native MoE cubin inventory in {origin}; {details}")


def _validate_source_cubin_inventory(native_root: Path) -> None:
    cubin_root = native_root / _CUBIN_ARCHIVE_DIR
    archive_stems = {
        _cubin_archive_stem(path.name)
        for path in cubin_root.glob(f"*{_CUBIN_ARCHIVE_SUFFIX}")
        if path.is_file()
    }
    metadata_inventories = _kernel_meta_symbols(
        (native_root / _KERNEL_META_INFO).read_text()
    )
    _validate_cubin_inventory(
        archive_stems,
        *metadata_inventories,
        origin=f"source tree {native_root}",
    )


def _validate_sdist(path: Path, source_root: Path) -> None:
    native_root = source_root / _PACKAGE_PREFIX
    _validate_source_cubin_inventory(native_root)

    with tarfile.open(path, "r:gz") as archive:
        members = {
            relative: member
            for member in archive.getmembers()
            if member.isfile()
            and (relative := _native_relative_path(member.name)) is not None
        }
        actual = set(members)

        metadata_member = members.get(_KERNEL_META_INFO)
        if metadata_member is None:
            raise RuntimeError(
                f"native MoE metadata missing from sdist: {_KERNEL_META_INFO}"
            )
        metadata_file = archive.extractfile(metadata_member)
        if metadata_file is None:
            raise RuntimeError(
                f"cannot read native MoE metadata from sdist: {_KERNEL_META_INFO}"
            )
        metadata_inventories = _kernel_meta_symbols(metadata_file.read().decode())

    archive_stems = {
        _cubin_archive_stem(Path(relative).name)
        for relative in actual
        if relative.startswith(f"{_CUBIN_ARCHIVE_DIR}/")
        and relative.endswith(_CUBIN_ARCHIVE_SUFFIX)
    }
    _validate_cubin_inventory(
        archive_stems,
        *metadata_inventories,
        origin=f"source distribution {path}",
    )

    expected = _expected_sdist_files(source_root)
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"native MoE files missing from sdist: {missing[:20]}")

    forbidden = sorted(
        relative
        for relative in actual
        if relative.startswith("objs/")
        or "/__pycache__/" in f"/{relative}"
        or Path(relative).suffix in {".o", ".so", ".pyc"}
    )
    if forbidden:
        raise RuntimeError(f"native MoE build outputs leaked into sdist: {forbidden}")


def _preload_cuda_stub() -> None:
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    candidates = [
        cuda_home / "lib64/stubs/libcuda.so",
        *cuda_home.glob("targets/*/lib/stubs/libcuda.so"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            ctypes.CDLL(candidate, mode=os.RTLD_GLOBAL | os.RTLD_NOW)
            return
    raise RuntimeError(f"CUDA driver stub not found below {cuda_home}")


def _load_wheel_runner(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="tokenspeed-native-moe-wheel-") as tmp:
        root = Path(tmp)
        with ZipFile(path) as archive:
            archive.extractall(root)

        _preload_cuda_stub()
        module_path = root / _PACKAGE_PREFIX / "__init__.py"
        module_name = "_tokenspeed_native_moe_wheel_smoke"
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
            submodule_search_locations=[str(module_path.parent)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load native MoE module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not module.has_native_mxfp4_moe():
            raise RuntimeError(f"native MoE DSO failed to load: {module._LOAD_ERROR!r}")
        if not module._OFFLINE_TACTICS_COMPATIBLE:
            raise RuntimeError("native MoE DSO tactic ABI does not match")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="artifact_type", required=True)

    wheel = subparsers.add_parser("wheel")
    wheel.add_argument("artifact", type=Path)
    wheel.add_argument("--load", action="store_true")

    sdist = subparsers.add_parser("sdist")
    sdist.add_argument("artifact", type=Path)
    sdist.add_argument("--source-root", type=Path, required=True)

    args = parser.parse_args()
    if args.artifact_type == "wheel":
        _validate_wheel(args.artifact)
        if args.load:
            _load_wheel_runner(args.artifact)
    else:
        _validate_sdist(args.artifact, args.source_root.resolve())
    print(f"validated native MoE {args.artifact_type}: {args.artifact}")


if __name__ == "__main__":
    main()
