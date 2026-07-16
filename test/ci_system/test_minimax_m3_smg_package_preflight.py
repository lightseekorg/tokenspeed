import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

import pytest
from minimax_m3_smg_package_preflight import (
    PipCheckResult,
    PipInstallResult,
    audit_release_packages,
    main,
)

VERSIONS = {
    "tokenspeed-smg": "1.7.0.post20260716",
    "tokenspeed-smg-grpc-proto": "0.4.14.post20260716",
    "tokenspeed-smg-grpc-servicer": "0.6.0.post20260716",
}


def _write_distribution(
    site_packages: Path,
    name: str,
    version: str,
    files: dict[str, str | bytes],
) -> None:
    for relative, contents in files.items():
        path = site_packages / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, bytes):
            path.write_bytes(contents)
        else:
            path.write_text(contents)

    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    metadata_path = dist_info / "METADATA"
    wheel_path = dist_info / "WHEEL"
    metadata_path.write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    wheel_path.write_text("Wheel-Version: 1.0\nRoot-Is-Purelib: true\n")
    recorded = [
        *files,
        str(metadata_path.relative_to(site_packages)),
        str(wheel_path.relative_to(site_packages)),
    ]
    (dist_info / "RECORD").write_text("".join(f"{path},,\n" for path in recorded))


def _write_pyproject(
    path: Path,
    *,
    smg_specifier: str = "==",
    smg_version: str = VERSIONS["tokenspeed-smg"],
) -> None:
    path.write_text(
        "[project]\n"
        "dependencies = [\n"
        f'  "tokenspeed-smg{smg_specifier}{smg_version}",\n'
        f'  "tokenspeed-smg-grpc-proto=={VERSIONS["tokenspeed-smg-grpc-proto"]}",\n'
        f'  "tokenspeed-smg-grpc-servicer=={VERSIONS["tokenspeed-smg-grpc-servicer"]}",\n'
        "]\n"
    )


def _write_pip_report(path: Path, *, url_scheme: str = "https") -> None:
    path.write_text(
        json.dumps(
            {
                "version": "1",
                "install": [
                    {
                        "download_info": {
                            "url": (
                                f"{url_scheme}://files.pythonhosted.org/packages/"
                                f"{name}-{version}-py3-none-any.whl"
                            ),
                            "archive_info": {"hashes": {"sha256": "a" * 64}},
                        },
                        "is_direct": False,
                        "requested": True,
                        "metadata": {"name": name, "version": version},
                    }
                    for name, version in VERSIONS.items()
                ],
            }
        )
    )


@pytest.fixture
def release_environment(tmp_path: Path, monkeypatch):
    prefix = tmp_path / "venv"
    site_packages = prefix / "lib/python3.12/site-packages"
    site_packages.mkdir(parents=True)
    _write_distribution(
        site_packages,
        "tokenspeed-smg",
        VERSIONS["tokenspeed-smg"],
        {
            "smg/__init__.py": "",
            "smg/smg_rs.abi3.so": b"binary-prefix-minimax_m3_vl-binary-suffix",
        },
    )
    _write_distribution(
        site_packages,
        "tokenspeed-smg-grpc-proto",
        VERSIONS["tokenspeed-smg-grpc-proto"],
        {"smg_grpc_proto/__init__.py": ""},
    )
    _write_distribution(
        site_packages,
        "tokenspeed-smg-grpc-servicer",
        VERSIONS["tokenspeed-smg-grpc-servicer"],
        {
            "smg_grpc_servicer/__init__.py": "",
            "smg_grpc_servicer/mm_rdma.py": "CONFIG = None\n",
            # These adapters are outside TokenSpeed's release surface.  Their
            # dynamic/product environment access must not make this preflight
            # report a false positive.
            "smg_grpc_servicer/mm_shm.py": (
                'import os\nunlink = os.getenv("TOKENSPEED_UNLINK_MM_SHM_AFTER_READ")\n'
            ),
            "smg_grpc_servicer/sglang/utils.py": (
                'import os\n_TOKEN_ID_ARRAY_ENV = "TOKEN_ID_ARRAY"\n'
                "token_ids = os.environ.get(_TOKEN_ID_ARRAY_ENV)\n"
            ),
            "smg_grpc_servicer/tokenspeed/__init__.py": "",
            "smg_grpc_servicer/tokenspeed/encoder_servicer.py": (
                "pixel = server_args.epd_pixel_shm\n"
                "offloop = server_args.epd_ingest_offloop\n"
                "unlink = server_args.unlink_mm_shm_after_read\n"
            ),
            "smg_grpc_servicer/tokenspeed/rdma_config.py": "class RdmaConfig: pass\n",
            "smg_grpc_servicer/tokenspeed/scheduler_launcher.py": (
                "async_llm.auto_create_handle_loop(manage_signals=False)\n"
            ),
            "smg_grpc_servicer/tokenspeed/server.py": (
                "limit = _grpc_max_message_bytes(server_args)\n"
            ),
            "smg_grpc_servicer/tokenspeed/servicer.py": (
                "async def close(self):\n    await self.async_llm.close()\n"
            ),
        },
    )
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject)

    for import_name in ("smg", "smg_grpc_proto", "smg_grpc_servicer"):
        sys.modules.pop(import_name, None)
    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()
    distributions = list(metadata.distributions(path=[str(site_packages)]))
    pip_report = tmp_path / "pip-install.json"
    _write_pip_report(pip_report)
    python_environment = {
        "executable": str(prefix / "bin/python"),
        "prefix": str(prefix),
        "base_prefix": str(tmp_path / "base-python"),
        "venv_config": str(prefix / "pyvenv.cfg"),
        "venv_config_error": "",
        "include_system_site_packages": "false",
        "user_site_enabled": False,
        "pythonpath": "",
        "pip_override_keys": [],
        "sys_path": [str(site_packages)],
    }
    return site_packages, pyproject, distributions, pip_report, python_environment


def _audit(
    release_environment,
    *,
    distributions=None,
    python_environment=None,
    pip_report=None,
):
    site_packages, pyproject, installed, report, environment = release_environment
    del site_packages
    return audit_release_packages(
        pyproject,
        distributions=installed if distributions is None else distributions,
        pip_check=_passing_pip_check,
        pip_install_report=report if pip_report is None else pip_report,
        python_environment=(
            environment if python_environment is None else python_environment
        ),
    )


def _passing_pip_check() -> PipCheckResult:
    return PipCheckResult(0, "No broken requirements found.\n", "")


def test_release_package_audit_passes_for_exact_private_wheels(
    release_environment,
) -> None:
    result = _audit(release_environment)

    assert result["ok"] is True
    assert result["failures"] == []
    assert result["pip_check"]["ok"] is True
    assert result["python_environment"]["pythonpath_present"] is False
    assert "pythonpath" not in result["python_environment"]
    assert result["servicer_product_environment_scan"]["ok"] is True
    assert result["servicer_product_environment_scan"]["violation_count"] == 0
    scanned_sources = {
        details["file"]
        for details in result["servicer_product_environment_scan"]["files"]
    }
    assert scanned_sources == {
        "smg_grpc_servicer/mm_rdma.py",
        "smg_grpc_servicer/tokenspeed/__init__.py",
        "smg_grpc_servicer/tokenspeed/encoder_servicer.py",
        "smg_grpc_servicer/tokenspeed/rdma_config.py",
        "smg_grpc_servicer/tokenspeed/scheduler_launcher.py",
        "smg_grpc_servicer/tokenspeed/server.py",
        "smg_grpc_servicer/tokenspeed/servicer.py",
    }
    assert result["servicer_product_environment_scan"]["source_file_count"] == len(
        scanned_sources
    )
    assert result["binding_product_environment_scan"] == {
        "file": "smg/smg_rs.abi3.so",
        "forbidden_keys": [],
        "ok": True,
    }
    assert len(result["marker_checks"]) == 8
    assert all(check["ok"] for check in result["marker_checks"])
    assert all(
        package["import_owned_by_distribution"]
        for package in result["packages"].values()
    )


@pytest.mark.parametrize(
    ("environment_read", "expected_api", "expected_key"),
    [
        (
            'import os\nhidden = os.getenv("TOKENSPEED_MODEL")\n',
            "os.getenv",
            "TOKENSPEED_MODEL",
        ),
        (
            'import os\nhidden = os.environ["SMG_ENGINE"]\n',
            "os.environ[]",
            "SMG_ENGINE",
        ),
        (
            'from os import environ\nhidden = environ.get("EPD_TRANSPORT")\n',
            "os.environ.get",
            "EPD_TRANSPORT",
        ),
        (
            'import os\nhidden = os.getenv("TS_DEBUG_CACHE")\n',
            "os.getenv",
            "TS_DEBUG_CACHE",
        ),
        (
            'from os import getenv\nkey = "HOME"\nhidden = getenv(key)\n',
            "os.getenv",
            "<dynamic>",
        ),
    ],
)
def test_audit_rejects_servicer_product_or_dynamic_environment_reads(
    release_environment,
    environment_read: str,
    expected_api: str,
    expected_key: str,
) -> None:
    site_packages, _, _, _, _ = release_environment
    server_path = site_packages / "smg_grpc_servicer/tokenspeed/server.py"
    server_path.write_text(
        environment_read + "limit = _grpc_max_message_bytes(server_args)\n"
    )

    result = _audit(release_environment)

    scan = result["servicer_product_environment_scan"]
    violations = [
        violation
        for file_details in scan["files"]
        for violation in file_details["violations"]
    ]
    assert result["ok"] is False
    assert scan["ok"] is False
    assert scan["violation_count"] == 1
    assert len(violations) == 1
    assert violations[0]["api"] == expected_api
    assert violations[0]["key"] == expected_key
    assert violations[0]["line"] == (3 if expected_key == "<dynamic>" else 2)
    assert any(expected_key in failure for failure in result["failures"])


@pytest.mark.parametrize(
    ("environment_access", "expected_api", "expected_key"),
    [
        (
            'import os\nos.environ["SMG_ENGINE"] = "hidden"\n',
            "os.environ[]=",
            "SMG_ENGINE",
        ),
        (
            'import os\nos.environ.update({"EPD_TRANSPORT": "hidden"})\n',
            "os.environ.update",
            "<dynamic>",
        ),
        (
            'import os\nos.putenv("TOKENSPEED_MODEL", "hidden")\n',
            "os.putenv",
            "TOKENSPEED_MODEL",
        ),
        (
            'from os import putenv as set_env\nset_env("TS_DEBUG", "1")\n',
            "os.putenv",
            "TS_DEBUG",
        ),
        (
            'import os\nset_env = os.putenv\nset_env("SMG_ALIAS", "1")\n',
            "os.putenv",
            "SMG_ALIAS",
        ),
        (
            'import os\nunset_env = os.unsetenv\nunset_env("EPD_ALIAS")\n',
            "os.unsetenv",
            "EPD_ALIAS",
        ),
    ],
)
def test_audit_rejects_servicer_product_environment_writes(
    release_environment,
    environment_access: str,
    expected_api: str,
    expected_key: str,
) -> None:
    site_packages, _, _, _, _ = release_environment
    root_module = site_packages / "smg_grpc_servicer/mm_rdma.py"
    root_module.write_text(environment_access)

    result = _audit(release_environment)

    scan = result["servicer_product_environment_scan"]
    violations = [
        violation
        for file_details in scan["files"]
        if file_details["file"] == "smg_grpc_servicer/mm_rdma.py"
        for violation in file_details["violations"]
    ]
    assert result["ok"] is False
    assert scan["ok"] is False
    assert len(violations) == 1
    assert violations[0]["api"] == expected_api
    assert violations[0]["key"] == expected_key


def test_audit_rejects_missing_required_mm_rdma_source(release_environment) -> None:
    site_packages, _, _, _, _ = release_environment
    (site_packages / "smg_grpc_servicer/mm_rdma.py").unlink()

    result = _audit(release_environment)

    scan = result["servicer_product_environment_scan"]
    mm_rdma = next(
        details
        for details in scan["files"]
        if details["file"] == "smg_grpc_servicer/mm_rdma.py"
    )
    assert result["ok"] is False
    assert scan["ok"] is False
    assert mm_rdma["ok"] is False
    assert "distribution RECORD does not contain" in mm_rdma["error"]
    assert "smg_grpc_servicer/mm_rdma.py" in mm_rdma["error"]


def test_audit_rejects_forbidden_environment_key_in_compiled_binding(
    release_environment,
) -> None:
    site_packages, _, _, _, _ = release_environment
    binding = site_packages / "smg/smg_rs.abi3.so"
    binding.write_bytes(
        b"binary-prefix-minimax_m3_vl-SMG_RDMA_SLOT_BYTES-binary-suffix"
    )

    result = _audit(release_environment)

    scan = result["binding_product_environment_scan"]
    assert result["ok"] is False
    assert scan == {
        "file": "smg/smg_rs.abi3.so",
        "forbidden_keys": ["SMG_RDMA_SLOT_BYTES"],
        "ok": False,
    }
    assert any("SMG_RDMA_SLOT_BYTES" in failure for failure in result["failures"])


def test_audit_rejects_public_distribution_and_source_overlay(
    release_environment, tmp_path: Path, monkeypatch
) -> None:
    site_packages, _, _, _, _ = release_environment
    _write_distribution(site_packages, "smg", "1.7.0", {})
    overlay = tmp_path / "smg-source-checkout"
    (overlay / "smg").mkdir(parents=True)
    (overlay / "smg/__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(overlay))
    importlib.invalidate_caches()
    distributions = list(metadata.distributions(path=[str(site_packages)]))

    result = _audit(release_environment, distributions=distributions)

    assert result["ok"] is False
    assert any(
        "conflicting public distribution smg" in item for item in result["failures"]
    )
    assert any("resolves outside" in item for item in result["failures"])


def test_audit_rejects_non_exact_pin_and_missing_release_markers(
    release_environment,
) -> None:
    site_packages, pyproject, _, _, _ = release_environment
    _write_pyproject(pyproject, smg_specifier=">=")
    (site_packages / "smg/smg_rs.abi3.so").write_bytes(b"old-binding")
    (site_packages / "smg_grpc_servicer/tokenspeed/rdma_config.py").unlink()
    (site_packages / "smg_grpc_servicer/tokenspeed/scheduler_launcher.py").write_text(
        "async_llm.auto_create_handle_loop()\n"
    )

    distributions = list(metadata.distributions(path=[str(site_packages)]))
    result = _audit(release_environment, distributions=distributions)

    assert result["ok"] is False
    assert any(
        "must use one unconditional, exact == pin" in item
        for item in result["failures"]
    )
    assert any("lacks minimax_m3_vl" in item for item in result["failures"])
    assert any("rdma_config.py" in item for item in result["failures"])
    assert any(
        "launcher_disables_nested_signal_management" in item
        for item in result["failures"]
    )


def test_audit_rejects_installed_version_that_differs_from_exact_pin(
    release_environment,
) -> None:
    _, pyproject, _, _, _ = release_environment
    _write_pyproject(pyproject, smg_version="1.7.0.post20990101")

    result = _audit(release_environment)

    assert result["ok"] is False
    assert result["packages"]["tokenspeed-smg"]["version_matches_pin"] is False
    assert any("does not match exact pin" in item for item in result["failures"])


def test_audit_rejects_private_distribution_installed_from_source(
    release_environment,
) -> None:
    site_packages, _, _, _, _ = release_environment
    dist_info = site_packages / f"tokenspeed_smg-{VERSIONS['tokenspeed-smg']}.dist-info"
    direct_url = dist_info / "direct_url.json"
    direct_url.write_text(
        json.dumps(
            {
                "url": "file:///workspace/private-smg-source",
                "dir_info": {"editable": True},
            }
        )
    )
    with (dist_info / "RECORD").open("a") as record:
        record.write(f"{direct_url.relative_to(site_packages)},,\n")
    distributions = list(metadata.distributions(path=[str(site_packages)]))

    result = _audit(release_environment, distributions=distributions)

    assert result["ok"] is False
    assert result["packages"]["tokenspeed-smg"]["source_install"] is True
    assert any(
        "installed from a source directory" in item for item in result["failures"]
    )


def test_audit_rejects_non_isolated_python_and_pythonpath(
    release_environment,
) -> None:
    _, _, _, _, environment = release_environment
    invalid_environment = {
        **environment,
        "base_prefix": environment["prefix"],
        "include_system_site_packages": "true",
        "user_site_enabled": True,
        "pythonpath": "/workspace/source-overlay",
        "pip_override_keys": ["PIP_INDEX_URL"],
    }

    result = _audit(
        release_environment,
        python_environment=invalid_environment,
    )

    assert result["ok"] is False
    assert any(
        "not running inside a virtual environment" in failure
        for failure in result["failures"]
    )
    assert any(
        "disable system site-packages" in failure for failure in result["failures"]
    )
    assert any("PYTHONPATH must be unset" in failure for failure in result["failures"])
    assert any("PIP_INDEX_URL" in failure for failure in result["failures"])


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("http", "official PyPI wheel"),
        ("direct", "direct install"),
        ("missing_hash", "valid SHA-256"),
        ("sdist", "official PyPI wheel"),
    ],
)
def test_audit_rejects_untrusted_pip_report(
    release_environment, tmp_path: Path, mutation: str, expected_failure: str
) -> None:
    _, _, _, report, _ = release_environment
    document = json.loads(report.read_text())
    entry = document["install"][0]
    if mutation == "http":
        entry["download_info"]["url"] = entry["download_info"]["url"].replace(
            "https://", "http://"
        )
    elif mutation == "direct":
        entry["is_direct"] = True
    elif mutation == "missing_hash":
        entry["download_info"]["archive_info"]["hashes"] = {}
    else:
        entry["download_info"]["url"] = entry["download_info"]["url"].replace(
            ".whl", ".tar.gz"
        )
    invalid_report = tmp_path / f"{mutation}.json"
    invalid_report.write_text(json.dumps(document))

    result = _audit(release_environment, pip_report=invalid_report)

    assert result["ok"] is False
    assert any(expected_failure in failure for failure in result["failures"])


def test_main_installs_exact_published_wheels_before_audit(
    release_environment, tmp_path: Path
) -> None:
    _, pyproject, distributions, _, python_environment = release_environment
    output = tmp_path / "artifacts/smg_packages.json"
    report = tmp_path / "artifacts/install.json"
    calls = []

    def install(pins, report_path):
        calls.append((dict(pins), report_path))
        _write_pip_report(report_path)
        return PipInstallResult(0, "installed\n", "")

    returncode = main(
        [
            "--pyproject",
            str(pyproject),
            "--install-published",
            "--pip-install-report",
            str(report),
            "--output",
            str(output),
        ],
        distributions=distributions,
        pip_check=_passing_pip_check,
        pip_install=install,
        python_environment=python_environment,
    )

    artifact = json.loads(output.read_text())
    assert returncode == 0
    assert calls == [(VERSIONS, report)]
    assert artifact["published_install"]["ok"] is True
    assert artifact["pip_install_report"]["ok"] is True


def test_main_refuses_to_install_before_strict_python_is_proven(
    release_environment, tmp_path: Path
) -> None:
    _, pyproject, distributions, _, python_environment = release_environment
    invalid_environment = {
        **python_environment,
        "base_prefix": python_environment["prefix"],
    }
    output = tmp_path / "artifacts/smg_packages.json"
    report = tmp_path / "artifacts/install.json"
    called = False

    def install(_pins, _report_path):
        nonlocal called
        called = True
        return PipInstallResult(0)

    returncode = main(
        [
            "--pyproject",
            str(pyproject),
            "--install-published",
            "--pip-install-report",
            str(report),
            "--output",
            str(output),
        ],
        distributions=distributions,
        pip_check=_passing_pip_check,
        pip_install=install,
        python_environment=invalid_environment,
    )

    artifact = json.loads(output.read_text())
    assert returncode == 1
    assert called is False
    assert "refusing to install outside" in artifact["published_install"]["stderr"]


def test_main_writes_durable_json_when_pip_check_fails(
    release_environment, tmp_path: Path, capsys
) -> None:
    _, pyproject, distributions, pip_report, python_environment = release_environment
    output = tmp_path / "artifacts/smg_packages.json"

    returncode = main(
        [
            "--pyproject",
            str(pyproject),
            "--pip-install-report",
            str(pip_report),
            "--output",
            str(output),
        ],
        distributions=distributions,
        pip_check=lambda: PipCheckResult(1, "broken dependency\n", ""),
        python_environment=python_environment,
    )

    artifact = json.loads(output.read_text())
    assert returncode == 1
    assert artifact["ok"] is False
    assert artifact["pip_check"]["stdout"] == "broken dependency\n"
    assert "python -m pip check" in capsys.readouterr().out
