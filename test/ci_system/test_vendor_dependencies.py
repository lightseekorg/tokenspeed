from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR_VERSION = "3.8.10.post20260920"


def _read_vendor_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("tokenspeed-proton", "tokenspeed-triton")):
            continue
        name, separator, version = line.partition("==")
        assert separator, f"{path}: vendor requirements must be exact-pinned"
        pins[name] = version
    return pins


def test_vendor_dependency_pins_are_consistent() -> None:
    expected = {
        "tokenspeed-proton": VENDOR_VERSION,
        "tokenspeed-triton": VENDOR_VERSION,
    }
    requirements = ROOT / "tokenspeed-kernel/python/requirements"

    assert _read_vendor_pins(requirements / "cuda.txt") == expected
    assert _read_vendor_pins(requirements / "rocm.txt") == expected

    triton_pin = f"tokenspeed-triton=={VENDOR_VERSION}"
    for project in ("tokenspeed-kernel-amd", "tokenspeed-mla"):
        pyproject = (ROOT / project / "pyproject.toml").read_text(encoding="utf-8")
        assert f'"{triton_pin}",' in pyproject


def test_ci_installers_preinstall_vendor_pins_from_testpypi() -> None:
    scripts = {
        "install_deps.sh": "CUDA_REQ",
        "install_deps_rocm.sh": "ROCM_REQ",
    }

    for script_name, requirements_var in scripts.items():
        script = (ROOT / "test/ci_system" / script_name).read_text(encoding="utf-8")
        expected_grep = (
            "grep -E '^tokenspeed-(triton|proton)==' " f'"${{{requirements_var}}}"'
        )

        assert (
            "TOKENSPEED_TESTPYPI_INDEX=${TOKENSPEED_TESTPYPI_INDEX:-"
            "https://test.pypi.org/simple}"
        ) in script
        assert expected_grep in script
        assert '--index-url "${TOKENSPEED_TESTPYPI_INDEX}"' in script
        assert script.count("preinstall_tokenspeed_testpypi_packages") == 2
