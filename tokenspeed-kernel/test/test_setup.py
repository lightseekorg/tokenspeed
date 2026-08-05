from __future__ import annotations

import runpy
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import setuptools
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools import build_meta

SETUP_PY = Path(__file__).parents[1] / "python" / "setup.py"
REQUIREMENTS_DIR = SETUP_PY.parent / "requirements"
CUDA_CSRC_DIR = SETUP_PY.parent / "csrc" / "cuda"
OPS_TEST_DIR = Path(__file__).parent / "ops"


def _capture_install_requires(monkeypatch, backend: str) -> list[str]:
    setup_kwargs = {}
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", backend)
    monkeypatch.setattr(
        setuptools, "setup", lambda **kwargs: setup_kwargs.update(kwargs)
    )
    runpy.run_path(str(SETUP_PY))
    return setup_kwargs["install_requires"]


def _direct_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        requirements.append(line)
    return requirements


def _expected_install_requires(backend: str) -> list[str]:
    requirements = _direct_requirements(REQUIREMENTS_DIR / "common.txt")
    requirements.extend(_direct_requirements(REQUIREMENTS_DIR / f"{backend}.txt"))
    requirements.extend(
        _direct_requirements(REQUIREMENTS_DIR / f"{backend}-thirdparty.txt")
    )
    return list(dict.fromkeys(requirements))


def _requirements_by_name(requirements: list[str]) -> dict[str, Requirement]:
    assert all(not requirement.startswith("-") for requirement in requirements)
    parsed = [Requirement(requirement) for requirement in requirements]
    names = [canonicalize_name(requirement.name) for requirement in parsed]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"duplicate dependency names: {duplicates}"
    return dict(zip(names, parsed, strict=True))


def test_cuda_install_requires_include_runtime_dependencies(monkeypatch) -> None:
    install_requires = _capture_install_requires(monkeypatch, "cuda")

    assert install_requires == _expected_install_requires("cuda")
    requirements = _requirements_by_name(install_requires)
    assert {
        "tokenspeed-proton",
        "tokenspeed-triton",
        "flashinfer-python",
        "nvidia-ml-py",
        "nvtx",
        "torch",
        "tokenspeed-deepgemm",
    } <= requirements.keys()
    assert {"tokenspeed-kernel-amd", "tokenspeed-iris"}.isdisjoint(requirements)
    assert requirements["nvidia-cutlass-dsl"].extras == {"cu13"}


def test_rocm_install_requires_exclude_cuda_dependencies(monkeypatch) -> None:
    install_requires = _capture_install_requires(monkeypatch, "rocm")

    assert install_requires == _expected_install_requires("rocm")
    requirements = _requirements_by_name(install_requires)
    assert {
        "tokenspeed-proton",
        "tokenspeed-triton",
        "tokenspeed-kernel-amd",
        "tokenspeed-iris",
        "torch",
    } <= requirements.keys()
    assert {
        specifier.operator
        for specifier in requirements["tokenspeed-kernel-amd"].specifier
    } == {">="}
    assert {
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-cu13",
        "nvidia-ml-py",
        "nvtx",
        "quack-kernels",
        "tokenspeed-deepep",
        "tokenspeed-deepgemm",
        "tokenspeed-fa3",
        "tokenspeed-fa4",
        "tokenspeed-fast-hadamard-transform",
        "tokenspeed-flashmla",
        "tokenspeed-mla",
        "tokenspeed-trtllm-kernel",
    }.isdisjoint(requirements)


def test_read_requirements_skips_installer_options_and_cycles(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "-rsecond.txt\n--extra-index-url https://example.invalid/simple\n"
        "first-package==1\n",
        encoding="utf-8",
    )
    second.write_text(
        "--requirement=first.txt\n--find-links https://example.invalid/wheels\n"
        "second-package>=2\n",
        encoding="utf-8",
    )

    assert setup_namespace["_read_requirements"](first) == [
        "second-package>=2",
        "first-package==1",
    ]


def test_sdist_includes_requirement_files(tmp_path, monkeypatch) -> None:
    source = tmp_path / "python"
    dist_dir = tmp_path / "dist"
    shutil.copytree(SETUP_PY.parent, source)
    dist_dir.mkdir()
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setenv("TOKENSPEED_KERNEL_GIT_SHA", "test")
    monkeypatch.chdir(source)

    archive_name = build_meta.build_sdist(str(dist_dir))

    with tarfile.open(dist_dir / archive_name) as archive:
        archived_files = {
            name.split("/", maxsplit=1)[1] for name in archive.getnames() if "/" in name
        }
    expected_files = {
        f"requirements/{path.name}" for path in REQUIREMENTS_DIR.glob("*.txt")
    }
    assert expected_files <= archived_files
    expected_cuda_assets = {
        f"csrc/cuda/{path.relative_to(CUDA_CSRC_DIR)}"
        for path in CUDA_CSRC_DIR.rglob("*")
        if path.is_file()
    }
    assert expected_cuda_assets <= archived_files


def test_cuda_sources_and_thirdparty_layout(monkeypatch) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    assert setup_namespace["CUDA_CSRC_DIR"] == CUDA_CSRC_DIR
    assert "MSA_CSRC_DIR" not in setup_namespace
    assert "MSA_INSTALLED_CSRC" not in setup_namespace
    assert not (SETUP_PY.parent / "tokenspeed_kernel" / "thirdparty").exists()

    native_suffixes = {".cu", ".cuh", ".h", ".hpp", ".jinja"}
    package_native_assets = [
        path
        for path in (SETUP_PY.parent / "tokenspeed_kernel").rglob("*")
        if path.is_file() and path.suffix in native_suffixes
    ]
    assert package_native_assets == []

    non_msa_assets = [
        path
        for path in CUDA_CSRC_DIR.rglob("*")
        if path.is_file() and "msa" not in path.relative_to(CUDA_CSRC_DIR).parts
    ]
    assert len(non_msa_assets) == 73


def test_msa_cuda_extensions_are_aot_only() -> None:
    msa_ops = SETUP_PY.parent / "tokenspeed_kernel" / "ops" / "attention" / "msa"
    assert not (msa_ops / "cute_dsl/src/sm100/build_k2q_csr").exists()
    assert not (
        msa_ops / "cute_dsl/src/sm100/fwd_decode/build_decode_schedule"
    ).exists()

    for loader in (
        msa_ops / "cuda/k2q_csr/__init__.py",
        msa_ops / "cuda/decode_schedule/__init__.py",
    ):
        source = loader.read_text(encoding="utf-8")
        assert "torch.utils.cpp_extension" not in source
        assert "load_extension" in source


def test_solution_siblings_use_consistent_packages() -> None:
    ops = SETUP_PY.parent / "tokenspeed_kernel" / "ops"
    layout = {
        "activation": {"cuda", "flashinfer", "triton"},
        "attention/dsa": {
            "cuda",
            "cute_dsl",
            "deep_gemm",
            "flash_mla",
            "flashinfer",
            "gluon",
            "triton",
        },
        "attention/gdn": {"flashinfer", "triton"},
        "attention/kda": {"cute_dsl", "flash_kda", "gluon", "triton"},
        "attention/mha": {"flash_attn", "flashinfer", "gluon", "triton"},
        "attention/mla": {
            "flash_mla",
            "flashinfer",
            "gluon",
            "tokenspeed_mla",
            "triton",
        },
        "attention/msa": {"cuda", "cute_dsl", "triton"},
        "attention/rmha": {"cute_dsl", "flash_attn", "gluon", "triton"},
        "communication": {
            "cuda",
            "deep_ep",
            "flashinfer",
            "iris",
            "nccl",
            "triton",
            "trtllm",
        },
        "embedding": {"cuda", "flashinfer", "triton"},
        "gemm/fp16": {"cuda", "flashinfer", "gluon", "triton"},
        "gemm/nvfp4": {"cute_dsl", "flashinfer", "trtllm"},
        "kvcache": {"cuda", "triton"},
        "layernorm": {"cuda", "flashinfer", "triton"},
        "model/deepseek_v4": {"cuda", "deep_gemm", "triton"},
        "model/kimi_k3/attn_res": {"cuda", "gluon", "torch", "triton"},
        "moe/fp8": {"deep_gemm", "flashinfer_cutlass", "flashinfer_trtllm", "triton"},
        "other/native": {"deep_ep", "deep_gemm", "trtllm"},
        "other/merge_state": {"cuda", "triton"},
        "other/moe_finalize": {"cuda"},
        "other/moe_routing": {"cuda", "triton"},
        "quantization": {"cuda", "flashinfer", "triton", "trtllm"},
    }

    for scope, solutions in layout.items():
        scope_dir = ops / scope
        for solution in solutions:
            assert (scope_dir / solution / "__init__.py").is_file()
            assert not (scope_dir / f"{solution}.py").exists()

    attention_ops = ops / "attention"
    for old_bundle in ("flash_attn", "flash_mla", "flashinfer", "gluon"):
        assert not (attention_ops / old_bundle).exists()


def test_moe_family_contains_only_weight_datatypes() -> None:
    moe_ops = SETUP_PY.parent / "tokenspeed_kernel" / "ops" / "moe"
    old_non_weight_dirs = {
        "finalize",
        "grouped_routing",
        "routing",
        "sigmoid_topk",
        "unfused",
    }
    entries = {
        path.relative_to(moe_ops).parts[0]
        for path in moe_ops.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path != moe_ops / "__init__.py"
    }
    assert entries.isdisjoint(old_non_weight_dirs)
    assert entries == {"fp16", "fp8", "int4", "mxfp4", "nvfp4"}


def test_operator_tests_are_grouped_by_family() -> None:
    assert not list(OPS_TEST_DIR.glob("test_*.py"))


def test_family_initializers_do_not_reexport_solutions() -> None:
    ops = SETUP_PY.parent / "tokenspeed_kernel" / "ops"

    for relative_path in (
        "activation/__init__.py",
        "other/metadata/__init__.py",
        "other/moe_grouped_routing/__init__.py",
        "other/moe_sigmoid_topk/__init__.py",
        "other/moe_unfused/__init__.py",
    ):
        assert not (ops / relative_path).read_text(encoding="utf-8")

    attention = (ops / "attention/__init__.py").read_text(encoding="utf-8")
    assert "kda_chunk_prefill" not in attention

    communication = (ops / "communication/__init__.py").read_text(encoding="utf-8")
    assert '"allgather_dual_rmsnorm"' not in communication
    assert '"allreduce_residual_rmsnorm"' not in communication
    assert '"reducescatter_residual_rmsnorm"' not in communication

    moe = (ops / "moe/__init__.py").read_text(encoding="utf-8")
    assert "moe_grouped_routing" not in moe
    assert "moe_sigmoid_bias_topk" not in moe
    assert "moe_unfused_apply" not in moe

    msa = (ops / "attention/msa/__init__.py").read_text(encoding="utf-8")
    assert " import *" not in msa


def test_cuda_kernel_groups_and_output_packages(monkeypatch) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))
    groups = setup_namespace["KERNEL_GROUPS"]

    expected_packages = {
        "rope": "tokenspeed_kernel.ops.embedding.cuda",
        "deepseek_v4_attention": "tokenspeed_kernel.ops.model.deepseek_v4.cuda",
        "minimax_m3_fused": "tokenspeed_kernel.ops.model.minimax_m3.cuda",
        "dsv3_gemm": "tokenspeed_kernel.ops.gemm.fp16.cuda",
        "marlin": "tokenspeed_kernel.ops.quantization.cuda",
        "routing": "tokenspeed_kernel.ops.other.moe_routing.cuda",
        "sampling_chain": "tokenspeed_kernel.ops.sampling.cuda",
        "fused_topk_topp": "tokenspeed_kernel.ops.sampling.cuda",
        "rmsnorm_fused_parallel": "tokenspeed_kernel.ops.layernorm.cuda",
        "merge_state": "tokenspeed_kernel.ops.other.merge_state.cuda",
        "flashinfer_softmax": "tokenspeed_kernel.ops.sampling.flashinfer",
        "silu_fuse_block_quant": "tokenspeed_kernel.ops.activation.cuda",
        "silu_fuse_nvfp4_quant": "tokenspeed_kernel.ops.activation.cuda",
        "moe_finalize_fuse_shared": "tokenspeed_kernel.ops.other.moe_finalize.cuda",
        "kvcacheio": "tokenspeed_kernel.ops.kvcache.cuda",
        "lm_head_gemm": "tokenspeed_kernel.ops.gemm.fp16.cuda",
        "trtllm_comm": "tokenspeed_kernel.ops.communication.trtllm",
        "attn_res": "tokenspeed_kernel.ops.model.kimi_k3.attn_res.cuda",
    }
    assert len(groups) == 18
    assert sum(len(sources) for _, sources, _, _, _ in groups) == 36
    assert {name: package for name, _, package, _, _ in groups} == expected_packages
    setup_namespace["_validate_kernel_groups"](groups)

    for name, sources, package, _, _ in groups:
        assert all(source.is_relative_to(CUDA_CSRC_DIR) for source in sources)
        assert setup_namespace["_kernel_output_path"](name, package) == (
            SETUP_PY.parent.joinpath(*package.split(".")) / "objs" / name / f"{name}.so"
        )

    extension_groups = setup_namespace["PYTORCH_CUDA_EXTENSION_GROUPS"]
    assert extension_groups == [
        (
            "sparse_build_k2q_csr_ext",
            CUDA_CSRC_DIR / "msa" / "build_k2q_csr.cu",
            "tokenspeed_kernel.ops.attention.msa.cuda",
        ),
        (
            "sparse_decode_schedule_ext",
            CUDA_CSRC_DIR / "msa" / "build_decode_schedule.cu",
            "tokenspeed_kernel.ops.attention.msa.cuda",
        ),
    ]
    setup_namespace["_validate_pytorch_extension_groups"](extension_groups)
    for name, source, package in extension_groups:
        assert source.is_relative_to(CUDA_CSRC_DIR / "msa")
        output = setup_namespace["_python_extension_output_path"](name, package)
        assert output.parent == (
            SETUP_PY.parent.joinpath(*package.split(".")) / "objs" / name
        )
        assert output.name.startswith(name)
        assert output.suffix == ".so"


def test_cuda_package_data(monkeypatch) -> None:
    setup_kwargs = {}
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(
        setuptools, "setup", lambda **kwargs: setup_kwargs.update(kwargs)
    )
    setup_namespace = runpy.run_path(str(SETUP_PY))
    package_data = setup_kwargs["package_data"]

    output_packages = {entry[2] for entry in setup_namespace["KERNEL_GROUPS"]}
    output_packages.update(
        entry[2] for entry in setup_namespace["PYTORCH_CUDA_EXTENSION_GROUPS"]
    )
    assert setup_namespace["CUDA_OUTPUT_PACKAGES"] == sorted(output_packages)
    assert all(package_data[package] == ["objs/**/*.so"] for package in output_packages)
    assert all("thirdparty" not in package for package in package_data)
    assert package_data["tokenspeed_kernel.ops.attention.msa"] == ["README.md"]
    assert package_data["tokenspeed_kernel.ops.attention.msa.cute_dsl"] == [
        "README.md",
        "requirements.txt",
    ]
    assert setup_kwargs["distclass"]().has_ext_modules()


def test_rocm_package_data_excludes_cuda_libraries(monkeypatch) -> None:
    setup_kwargs = {}
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "rocm")
    monkeypatch.setattr(
        setuptools, "setup", lambda **kwargs: setup_kwargs.update(kwargs)
    )
    setup_namespace = runpy.run_path(str(SETUP_PY))

    package_data = setup_kwargs["package_data"]
    output_packages = {entry[2] for entry in setup_namespace["KERNEL_GROUPS"]}
    output_packages.update(
        entry[2] for entry in setup_namespace["PYTORCH_CUDA_EXTENSION_GROUPS"]
    )
    assert output_packages.isdisjoint(package_data)
    assert not setup_kwargs["distclass"]().has_ext_modules()


def test_cuda_include_dirs_prefer_complete_toolkit_headers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "cuda_runtime.h").touch()
    (cuda_include / "cublas_v2.h").touch()

    site_packages = tmp_path / "site-packages"
    wheel_include = site_packages / "nvidia" / "cu13" / "include"
    wheel_include.mkdir(parents=True)
    (wheel_include / "cuda_runtime.h").touch()
    (wheel_include / "cublas_v2.h").touch()

    builder = setup_namespace["CudaKernelBuilder"]([], verbose=False)
    monkeypatch.setattr(builder, "_cuda_toolkit_roots", lambda: iter([cuda_root]))
    monkeypatch.setattr(builder, "_site_paths", lambda: iter([site_packages]))

    include_dirs = builder._resolve_include_dirs()

    assert str(cuda_include) in include_dirs
    assert str(wheel_include) not in include_dirs


def test_cuda_include_dirs_fall_back_from_partial_toolkit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "cuda_runtime.h").touch()

    site_packages = tmp_path / "site-packages"
    wheel_include = site_packages / "nvidia" / "cu13" / "include"
    wheel_include.mkdir(parents=True)
    (wheel_include / "cuda_runtime.h").touch()
    (wheel_include / "cublas_v2.h").touch()

    builder = setup_namespace["CudaKernelBuilder"]([], verbose=False)
    monkeypatch.setattr(builder, "_cuda_toolkit_roots", lambda: iter([cuda_root]))
    monkeypatch.setattr(builder, "_site_paths", lambda: iter([site_packages]))

    include_dirs = builder._resolve_include_dirs()

    assert str(cuda_include) not in include_dirs
    assert str(wheel_include) in include_dirs
