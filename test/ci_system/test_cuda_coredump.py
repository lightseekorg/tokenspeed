import sys
from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))
from ci_system import ci_utils, cuda_coredump


def test_configure_projects_explicit_cuda_driver_contract(tmp_path: Path):
    env = {}

    cuda_coredump.configure(str(tmp_path), env=env)

    assert env == {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": (
            "skip_nonrelocated_elf_images,skip_global_memory,"
            "skip_shared_memory,skip_local_memory,skip_constbank_memory"
        ),
        "CUDA_COREDUMP_FILE": f"{tmp_path}/cuda_coredump_%h.%p.%t",
    }


def test_configure_rejects_inherited_conflict(tmp_path: Path):
    env = {"CUDA_COREDUMP_FILE": "/inherited/dump"}
    dump_dir = tmp_path / "dumps"

    with pytest.raises(ValueError, match="conflicts with an existing value"):
        cuda_coredump.configure(str(dump_dir), env=env)

    assert env == {"CUDA_COREDUMP_FILE": "/inherited/dump"}
    assert not dump_dir.exists()


def test_disable_removes_inherited_driver_configuration():
    env = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": "inherited",
        "CUDA_COREDUMP_FILE": "/inherited/dump",
        "KEEP": "value",
    }

    cuda_coredump.disable(env)

    assert env == {"KEEP": "value"}


def test_cleanup_and_report_use_explicit_directory(tmp_path: Path, capsys):
    stale = tmp_path / "cuda_coredump_host.1.2"
    stale.write_bytes(b"old")
    cuda_coredump.cleanup_dump_dir(str(tmp_path))
    assert not stale.exists()

    dump = tmp_path / "cuda_coredump_host.3.4"
    dump.write_bytes(b"new")
    cuda_coredump.report(
        str(tmp_path),
        env={"GITHUB_RUN_ID": "123", "GITHUB_REPOSITORY": "org/repo"},
    )
    output = capsys.readouterr().out
    assert str(dump) in output
    assert "gh run download 123 --repo org/repo" in output


def test_unit_runner_enables_coredump_only_from_explicit_argument(
    monkeypatch, tmp_path: Path
):
    configured = []
    cleaned = []
    monkeypatch.setattr(
        cuda_coredump,
        "configure",
        lambda path, *, env: configured.append((path, env)),
    )
    monkeypatch.setattr(cuda_coredump, "cleanup_dump_dir", cleaned.append)

    result = ci_utils.run_unittest_files(
        [],
        timeout_per_file=1,
        cuda_coredump_dir=str(tmp_path),
    )

    assert result == 0
    assert len(configured) == 1
    assert configured[0][0] == str(tmp_path)
    assert isinstance(configured[0][1], dict)
    assert cleaned == [str(tmp_path)]


def test_unit_runner_leaves_coredump_disabled_by_default(monkeypatch):
    monkeypatch.setattr(
        cuda_coredump,
        "configure",
        lambda *_, **__: pytest.fail("coredump must remain disabled"),
    )
    inherited = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_FILE": "/inherited/dump",
    }
    with monkeypatch.context() as context:
        context.setattr(ci_utils.os, "environ", inherited)
        disabled = []
        context.setattr(
            cuda_coredump,
            "disable",
            lambda env: disabled.append(dict(env)),
        )
        assert ci_utils.run_unittest_files([], timeout_per_file=1) == 0

    assert disabled == [inherited]
