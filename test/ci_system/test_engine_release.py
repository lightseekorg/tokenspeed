"""Check release metadata without compiling the native engine."""

import runpy
import shutil
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PREPARE = runpy.run_path(str(ROOT / ".github/scripts/prepare-engine-release.py"))[
    "prepare_release"
]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "tokenspeed-engine").mkdir()
    shutil.copyfile(
        ROOT / "tokenspeed-engine/pyproject.toml",
        tmp_path / "tokenspeed-engine/pyproject.toml",
    )
    return tmp_path


def test_stable_version_is_unchanged(project):
    path = project / "tokenspeed-engine/pyproject.toml"
    original = path.read_bytes()
    base = tomllib.loads(original.decode())["project"]["version"]
    assert PREPARE(project, "release", "", "engine", set()) == base
    assert path.read_bytes() == original


def test_post_skips_occupied_dates_and_preserves_base_until_build(project):
    path = project / "tokenspeed-engine/pyproject.toml"
    original = path.read_bytes()
    base = tomllib.loads(original.decode())["project"]["version"]
    tags = {
        f"tokenspeed-engine-nightly-v{base}.post20260131",
        f"tokenspeed-engine-v{base}.post20260201",
        f"tokenspeed-engine-v{base}rc20260202",
    }
    version = PREPARE(project, "post", "20260131", "version", tags)
    assert version == f"{base}.post20260202"
    assert path.read_bytes() == original
    assert PREPARE(project, "post", "20260202", "engine", set()) == version
    assert tomllib.loads(path.read_text())["project"]["version"] == version


def test_explicit_rc_uses_requested_date(project):
    path = project / "tokenspeed-engine/pyproject.toml"
    base = tomllib.loads(path.read_text())["project"]["version"]
    version = PREPARE(project, "rc", "20261112", "version", set())
    assert version == f"{base}rc20261112"
    assert PREPARE(project, "rc", "20261112", "engine", set()) == version
    assert tomllib.loads(path.read_text())["project"]["version"] == version


@pytest.mark.parametrize(
    "kind,date",
    [
        ("rc", "20260230"),
        ("release", "20261111"),
    ],
)
def test_invalid_inputs_leave_metadata_unchanged(project, kind, date):
    path = project / "tokenspeed-engine/pyproject.toml"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        PREPARE(project, kind, date, "engine", set())
    assert path.read_bytes() == original


def test_coordinated_nightly_keeps_its_date_for_resume_or_replacement(project):
    base = tomllib.loads((project / "tokenspeed-engine/pyproject.toml").read_text())[
        "project"
    ]["version"]
    occupied = {f"tokenspeed-engine-v{base}.post20261122"}
    version = PREPARE(project, "nightly", "20261122", "version", occupied)
    assert version == f"{base}.post20261122"
    assert PREPARE(project, "nightly", "20261122", "engine", set()) == version


def test_candidate_nightly_does_not_consume_normal_or_future_engine_dates(project):
    base = tomllib.loads((project / "tokenspeed-engine/pyproject.toml").read_text())[
        "project"
    ]["version"]
    tags = {f"tokenspeed-engine-v{base}.post20261122"}
    candidate = PREPARE(project, "nightly", "20261122", "version", tags, "rc2")
    assert candidate == f"{base}rc2.post20261122"
    tags.add(f"tokenspeed-engine-v{candidate}")
    assert PREPARE(project, "nightly", "20261122", "version", tags, "rc2") == candidate
    assert (
        PREPARE(project, "nightly", "20261123", "version", tags)
        == f"{base}.post20261123"
    )
