import zipfile
from pathlib import Path

import flashinfer_jit_cache_installer as installer
import pytest
from flashinfer_jit_cache_installer import (
    expected_jit_cache_version,
    index_wheel_url,
    install_url_if_needed,
    jit_cache_wheel_url,
    main,
    read_exact_pin,
    simple_index_url,
    unsatisfied_providers,
)


def test_read_exact_pin_ignores_other_requirements(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("torch==2.11.0\n" "flashinfer-python==0.6.18\n")

    assert read_exact_pin(requirements, "flashinfer-python") == "0.6.18"


def test_read_exact_pin_requires_exact_pin(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python>=0.6\n")

    with pytest.raises(ValueError, match="flashinfer-python exact pin not found"):
        read_exact_pin(requirements, "flashinfer-python")


@pytest.mark.parametrize("version", ["0.6.18", "0.7.2.post1"])
def test_jit_cache_url_tracks_flashinfer_and_cuda_versions(version):
    assert expected_jit_cache_version(version, "130") == f"{version}+cu130"
    assert jit_cache_wheel_url(
        version, "130", platform_tag="manylinux_2_28_aarch64"
    ) == (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        f"v{version}/flashinfer_jit_cache-{version}+cu130-cp39-abi3-"
        "manylinux_2_28_aarch64.whl"
    )


def test_install_url_if_needed_skips_matching_version(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(installer.metadata, "requires", lambda _: [])
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.18\n")

    url, expected, installed = install_url_if_needed(
        requirements,
        "130",
        installed_version="0.6.18+cu130",
    )

    assert url is None
    assert expected == "0.6.18+cu130"
    assert installed == "0.6.18+cu130"


def test_install_url_if_needed_reinstalls_missing_or_stale_version(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.18\n")

    missing_url, _, missing_installed = install_url_if_needed(
        requirements,
        "130",
        installed_version=None,
    )
    stale_url, _, stale_installed = install_url_if_needed(
        requirements,
        "130",
        installed_version="0.6.11.post3+cu130",
    )

    expected_url = jit_cache_wheel_url("0.6.18", "130")
    assert missing_url == expected_url
    assert missing_installed is None
    assert stale_url == expected_url
    assert stale_installed == "0.6.11.post3+cu130"


@pytest.mark.parametrize(
    "provider_version", [None, "0.6.18+cu130", "0.7.0+cu129", "0.7.0+cu130"]
)
def test_matching_shim_requires_matching_providers(
    tmp_path, monkeypatch, provider_version
):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.7.0\n")
    monkeypatch.setattr(
        installer.metadata,
        "requires",
        lambda _: ["flashinfer-jit-cache-sm100a==0.7.0+cu130"],
    )
    monkeypatch.setattr(
        installer, "installed_distribution_version", lambda _: provider_version
    )

    url, expected, installed = install_url_if_needed(
        requirements, "130", installed_version="0.7.0+cu130"
    )

    assert expected == installed == "0.7.0+cu130"
    assert url == (
        None
        if provider_version == "0.7.0+cu130"
        else jit_cache_wheel_url("0.7.0", "130")
    )


def test_matching_shim_checks_every_required_provider(tmp_path, monkeypatch):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.7.0\n")
    monkeypatch.setattr(
        installer.metadata,
        "requires",
        lambda _: [
            "flashinfer-jit-cache-sm100a==0.7.0+cu130",
            "flashinfer-jit-cache-sm103a==0.7.0+cu130",
        ],
    )
    versions = {"flashinfer-jit-cache-sm100a": "0.7.0+cu130"}
    monkeypatch.setattr(installer, "installed_distribution_version", versions.get)

    url, _, _ = install_url_if_needed(
        requirements, "130", installed_version="0.7.0+cu130"
    )

    assert url == jit_cache_wheel_url("0.7.0", "130")


def test_jit_cache_ignores_inactive_dependency_markers(monkeypatch):
    monkeypatch.setattr(
        installer.metadata,
        "requires",
        lambda _: ['flashinfer-jit-cache-sm100a==0.7.0+cu130; python_version < "3.0"'],
    )
    monkeypatch.setattr(installer, "installed_distribution_version", lambda _: None)

    assert installer.jit_cache_dependencies_satisfied()


class FakeIndexResponse:
    def __init__(self, page: bytes):
        self.page = page

    def read(self) -> bytes:
        return self.page

    def __enter__(self) -> "FakeIndexResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


def index_page(*links: str) -> bytes:
    anchors = "".join(f'<a href="{link}">wheel</a><br/>' for link in links)
    return f"<html><body>{anchors}</body></html>".encode()


def write_metapackage_wheel(tmp_path: Path, *requires: str) -> Path:
    wheel = (
        tmp_path
        / "flashinfer_jit_cache-0.7.2.post1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"
    )
    metadata_text = (
        "Metadata-Version: 2.1\n"
        "Name: flashinfer-jit-cache\n"
        "Version: 0.7.2.post1+cu130\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "flashinfer_jit_cache-0.7.2.post1+cu130.dist-info/METADATA",
            metadata_text,
        )
    return wheel


SM100A_SHA256 = "ab" * 32
SM103A_SHA256 = "cd" * 32
METAPACKAGE_SHA256 = "ef" * 32
SM100A_WHEEL = (
    "flashinfer_jit_cache_sm100a-0.7.2.post1+cu130-cp39-abi3-"
    "manylinux_2_28_aarch64.whl"
)
RELEASE_BASE = (
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.2.post1"
)


def test_simple_index_url_tracks_release_and_nightly():
    assert simple_index_url("130", "0.7.2.post1") == "https://flashinfer.ai/whl/cu130"
    assert (
        simple_index_url("129", "0.8.0.dev20261101")
        == "https://flashinfer.ai/whl/nightly/cu129"
    )


def test_index_wheel_url_resolves_hash_and_decodes_escaped_plus(monkeypatch):
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        return FakeIndexResponse(
            index_page(
                f"{RELEASE_BASE}/{SM100A_WHEEL.replace('+', '%2B')}"
                f"#sha256={SM100A_SHA256}",
                f"{RELEASE_BASE}/other-wheel.whl#sha256={'00' * 32}",
            )
        )

    monkeypatch.setattr(installer.urllib.request, "urlopen", fake_urlopen)

    url, sha256 = index_wheel_url(
        "https://flashinfer.ai/whl/cu130",
        "flashinfer-jit-cache-sm100a",
        "0.7.2.post1+cu130",
        platform_tag="manylinux_2_28_aarch64",
    )

    assert requested == ["https://flashinfer.ai/whl/cu130/flashinfer-jit-cache-sm100a/"]
    assert url == f"{RELEASE_BASE}/{SM100A_WHEEL}"
    assert sha256 == SM100A_SHA256


def test_index_wheel_url_requires_wheel_with_hash(monkeypatch):
    monkeypatch.setattr(
        installer.urllib.request,
        "urlopen",
        lambda url, timeout: FakeIndexResponse(
            index_page(
                f"{RELEASE_BASE}/{SM100A_WHEEL}",  # no #sha256 fragment
                f"{RELEASE_BASE}/other-wheel.whl#sha256={'00' * 32}",
            )
        ),
    )

    with pytest.raises(ValueError, match="not found in"):
        index_wheel_url(
            "https://flashinfer.ai/whl/cu130",
            "flashinfer-jit-cache-sm100a",
            "0.7.2.post1+cu130",
            platform_tag="manylinux_2_28_aarch64",
        )


def test_unsatisfied_providers_skips_satisfied_and_inactive(
    tmp_path: Path, monkeypatch
):
    wheel = write_metapackage_wheel(
        tmp_path,
        "flashinfer-jit-cache-sm100a==0.7.2.post1+cu130",
        "flashinfer-jit-cache-sm103a==0.7.2.post1+cu130",
        'flashinfer-jit-cache-sm110a==0.7.2.post1+cu130; python_version < "3.0"',
    )
    versions = {"flashinfer-jit-cache-sm100a": "0.7.2.post1+cu130"}
    monkeypatch.setattr(installer, "installed_distribution_version", versions.get)

    unsatisfied = unsatisfied_providers(wheel)

    assert [requirement.name for requirement in unsatisfied] == [
        "flashinfer-jit-cache-sm103a"
    ]


def test_unsatisfied_providers_requires_exact_pins(tmp_path: Path, monkeypatch):
    wheel = write_metapackage_wheel(tmp_path, "flashinfer-jit-cache-sm100a>=0.7.2")
    monkeypatch.setattr(installer, "installed_distribution_version", lambda *args: None)

    (requirement,) = unsatisfied_providers(wheel)

    with pytest.raises(ValueError, match="does not pin an exact version"):
        installer.exact_requirement_version(requirement)


def test_main_resolve_metapackage_prints_url_and_hash(
    tmp_path: Path, monkeypatch, capsys
):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.7.2.post1\n")
    monkeypatch.setattr(installer, "installed_distribution_version", lambda *args: None)
    monkeypatch.setattr(
        installer, "default_platform_tag", lambda: "manylinux_2_28_aarch64"
    )
    monkeypatch.setattr(
        installer.urllib.request,
        "urlopen",
        lambda url, timeout: FakeIndexResponse(
            index_page(
                f"{RELEASE_BASE}/flashinfer_jit_cache-0.7.2.post1+cu130-cp39-abi3-"
                f"manylinux_2_28_aarch64.whl#sha256={METAPACKAGE_SHA256}"
            )
        ),
    )

    assert (
        main(
            [
                "--requirements",
                str(requirements),
                "--cuda-index",
                "130",
                "--resolve-metapackage",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out.split() == [
        f"{RELEASE_BASE}/flashinfer_jit_cache-0.7.2.post1+cu130-cp39-abi3-"
        "manylinux_2_28_aarch64.whl",
        METAPACKAGE_SHA256,
    ]
    assert "do not match" in captured.err


def test_main_list_providers_prints_only_unsatisfied(
    tmp_path: Path, monkeypatch, capsys
):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.7.2.post1\n")
    wheel = write_metapackage_wheel(
        tmp_path,
        "flashinfer-jit-cache-sm100a==0.7.2.post1+cu130",
        "flashinfer-jit-cache-sm103a==0.7.2.post1+cu130",
    )
    versions = {"flashinfer-jit-cache-sm100a": "0.7.2.post1+cu130"}
    monkeypatch.setattr(installer, "installed_distribution_version", versions.get)
    monkeypatch.setattr(
        installer, "default_platform_tag", lambda: "manylinux_2_28_aarch64"
    )
    pages = {
        "https://flashinfer.ai/whl/cu130/flashinfer-jit-cache-sm103a/": index_page(
            f"{RELEASE_BASE}/flashinfer_jit_cache_sm103a-0.7.2.post1+cu130-cp39-abi3-"
            f"manylinux_2_28_aarch64.whl#sha256={SM103A_SHA256}"
        ),
    }
    monkeypatch.setattr(
        installer.urllib.request,
        "urlopen",
        lambda url, timeout: FakeIndexResponse(pages[url]),
    )

    assert (
        main(
            [
                "--requirements",
                str(requirements),
                "--cuda-index",
                "130",
                "--list-providers",
                str(wheel),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out.split() == [
        f"{RELEASE_BASE}/flashinfer_jit_cache_sm103a-0.7.2.post1+cu130-cp39-abi3-"
        "manylinux_2_28_aarch64.whl",
        SM103A_SHA256,
    ]
