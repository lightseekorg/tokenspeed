import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

import pytest
from minimax_m3_smg_package_preflight import (
    PipCheckResult,
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


@pytest.fixture
def release_environment(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
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
    return site_packages, pyproject, distributions


def _passing_pip_check() -> PipCheckResult:
    return PipCheckResult(0, "No broken requirements found.\n", "")


def test_release_package_audit_passes_for_exact_private_wheels(
    release_environment,
) -> None:
    _, pyproject, distributions = release_environment

    result = audit_release_packages(
        pyproject,
        distributions=distributions,
        pip_check=_passing_pip_check,
    )

    assert result["ok"] is True
    assert result["failures"] == []
    assert result["pip_check"]["ok"] is True
    assert len(result["marker_checks"]) == 8
    assert all(check["ok"] for check in result["marker_checks"])
    assert all(
        package["import_owned_by_distribution"]
        for package in result["packages"].values()
    )


def test_audit_rejects_public_distribution_and_source_overlay(
    release_environment, tmp_path: Path, monkeypatch
) -> None:
    site_packages, pyproject, _ = release_environment
    _write_distribution(site_packages, "smg", "1.7.0", {})
    overlay = tmp_path / "smg-source-checkout"
    (overlay / "smg").mkdir(parents=True)
    (overlay / "smg/__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(overlay))
    importlib.invalidate_caches()
    distributions = list(metadata.distributions(path=[str(site_packages)]))

    result = audit_release_packages(
        pyproject,
        distributions=distributions,
        pip_check=_passing_pip_check,
    )

    assert result["ok"] is False
    assert any(
        "conflicting public distribution smg" in item for item in result["failures"]
    )
    assert any("resolves outside" in item for item in result["failures"])


def test_audit_rejects_non_exact_pin_and_missing_release_markers(
    release_environment,
) -> None:
    site_packages, pyproject, distributions = release_environment
    _write_pyproject(pyproject, smg_specifier=">=")
    (site_packages / "smg/smg_rs.abi3.so").write_bytes(b"old-binding")
    (site_packages / "smg_grpc_servicer/tokenspeed/rdma_config.py").unlink()
    (site_packages / "smg_grpc_servicer/tokenspeed/scheduler_launcher.py").write_text(
        "async_llm.auto_create_handle_loop()\n"
    )

    result = audit_release_packages(
        pyproject,
        distributions=distributions,
        pip_check=_passing_pip_check,
    )

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
    _, pyproject, distributions = release_environment
    _write_pyproject(pyproject, smg_version="1.7.0.post20990101")

    result = audit_release_packages(
        pyproject,
        distributions=distributions,
        pip_check=_passing_pip_check,
    )

    assert result["ok"] is False
    assert result["packages"]["tokenspeed-smg"]["version_matches_pin"] is False
    assert any("does not match exact pin" in item for item in result["failures"])


def test_audit_rejects_private_distribution_installed_from_source(
    release_environment,
) -> None:
    site_packages, pyproject, _ = release_environment
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

    result = audit_release_packages(
        pyproject,
        distributions=distributions,
        pip_check=_passing_pip_check,
    )

    assert result["ok"] is False
    assert result["packages"]["tokenspeed-smg"]["source_install"] is True
    assert any(
        "installed from a source directory" in item for item in result["failures"]
    )


def test_main_writes_durable_json_when_pip_check_fails(
    release_environment, tmp_path: Path, capsys
) -> None:
    _, pyproject, distributions = release_environment
    output = tmp_path / "artifacts/smg_packages.json"

    returncode = main(
        ["--pyproject", str(pyproject), "--output", str(output)],
        distributions=distributions,
        pip_check=lambda: PipCheckResult(1, "broken dependency\n", ""),
    )

    artifact = json.loads(output.read_text())
    assert returncode == 1
    assert artifact["ok"] is False
    assert artifact["pip_check"]["stdout"] == "broken dependency\n"
    assert "python -m pip check" in capsys.readouterr().out
