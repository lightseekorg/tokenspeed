import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR_VERSION = "3.8.10.post20260828"


def read_vendor_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("tokenspeed-proton", "tokenspeed-triton")):
            continue
        assert "==" in line, f"{path}: vendor requirements must be exact-pinned"
        name, version = line.split("==", maxsplit=1)
        pins[name] = version
    return pins


def test_vendor_dependency_pins_are_consistent():
    expected = {
        "tokenspeed-proton": VENDOR_VERSION,
        "tokenspeed-triton": VENDOR_VERSION,
    }
    requirements = ROOT / "tokenspeed-kernel/python/requirements"

    assert read_vendor_pins(requirements / "cuda.txt") == expected
    assert read_vendor_pins(requirements / "rocm.txt") == expected

    triton_pin = f"tokenspeed-triton=={VENDOR_VERSION}"
    for project in ("tokenspeed-kernel-amd", "tokenspeed-mla"):
        pyproject = (ROOT / project / "pyproject.toml").read_text()
        assert f'"{triton_pin}",' in pyproject


def test_ci_installers_preinstall_vendor_pins_from_testpypi():
    scripts = {
        "install_deps.sh": "CUDA_REQ",
        "install_deps_rocm.sh": "ROCM_REQ",
    }

    for script_name, requirements_var in scripts.items():
        script = (ROOT / "test/ci_system" / script_name).read_text()
        assert (
            "TOKENSPEED_TESTPYPI_INDEX=${TOKENSPEED_TESTPYPI_INDEX:-"
            "https://test.pypi.org/simple}"
        ) in script


def test_kernel_sources_use_tokenspeed_triton_adapter():
    source_root = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel"
    adapter = source_root / "_triton.py"
    # These helpers intentionally interact with framework-owned stock Triton;
    # neither defines TokenSpeed Triton kernels.
    stock_triton_interop = {
        source_root / "thirdparty/fla.py",
        source_root / "thirdparty/msa/cute/src/common/cute_dsl_utils.py",
    }
    direct_imports = {}

    for path in source_root.rglob("*.py"):
        if path == adapter or path in stock_triton_interop:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            else:
                continue
            if any(
                name == "triton"
                or name.startswith("triton.")
                or name == "tokenspeed_triton"
                or name.startswith("tokenspeed_triton.")
                for name in names
            ):
                lines.append(node.lineno)
        if lines:
            direct_imports[str(path.relative_to(ROOT))] = lines

    assert direct_imports == {}, (
        "TokenSpeed kernel sources must import Triton through "
        f"tokenspeed_kernel._triton: {direct_imports}"
    )
