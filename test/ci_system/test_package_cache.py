import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("package_cache.sh")


def run_bash(command: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {command}'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_other_clusters_do_not_enable_package_cache(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "CI_RUNNER_LABEL": "gb200-4gpu",
            "FLASHINFER_CACHE_DIR": str(tmp_path / "flashinfer"),
        }
    )
    env.pop("PIP_CACHE_DIR", None)
    env.pop("CI_WHEEL_CACHE_DIR", None)
    env.pop("CI_CCACHE_DIR", None)
    result = run_bash(
        'configure_package_cache; printf "%s|%s|%s" "${PIP_CACHE_DIR:-}" "${CI_WHEEL_CACHE_DIR:-}" "${CI_CCACHE_DIR:-}"',
        env,
    )
    assert result.stdout == "||"


def test_b200v2_uses_persistent_cache_next_to_flashinfer(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "CI_RUNNER_LABEL": "b200v2-8gpu",
            "FLASHINFER_CACHE_DIR": str(tmp_path / "flashinfer"),
        }
    )
    env.pop("PIP_CACHE_DIR", None)
    env.pop("CI_WHEEL_CACHE_DIR", None)
    env.pop("CI_CCACHE_DIR", None)
    result = run_bash(
        'configure_package_cache >/dev/null; printf "%s|%s|%s" "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}" "${CI_CCACHE_DIR}"',
        env,
    )
    assert result.stdout == (
        f"{tmp_path / 'pip'}|{tmp_path / 'wheelhouse'}|{tmp_path / 'ccache'}"
    )
    assert (tmp_path / "pip").is_dir()
    assert (tmp_path / "wheelhouse").is_dir()


def test_slurm_uses_mounted_persistent_cache(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "CI_RUNNER_LABEL": "slurm-gb300-4gpu",
            "XDG_CACHE_HOME": str(tmp_path),
        }
    )
    env.pop("PIP_CACHE_DIR", None)
    env.pop("CI_WHEEL_CACHE_DIR", None)
    env.pop("CI_CCACHE_DIR", None)
    result = run_bash(
        'configure_package_cache >/dev/null; printf "%s|%s|%s" "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}" "${CI_CCACHE_DIR}"',
        env,
    )
    assert result.stdout == (
        f"{tmp_path / 'pip'}|{tmp_path / 'wheelhouse'}|{tmp_path / 'ccache'}"
    )


@pytest.mark.parametrize(("fork_pr", "read_only"), [("false", ""), ("true", "1")])
def test_nvcc_cache_normalizes_checkout_and_isolates_fork_writes(
    tmp_path: Path, fork_pr: str, read_only: str
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_ccache = bin_dir / "ccache"
    fake_ccache.write_text("#!/bin/bash\nexit 0\n")
    fake_ccache.chmod(0o755)
    workspace = tmp_path / "checkout"
    cache_dir = tmp_path / "ccache"
    env = os.environ.copy()
    env.update(
        {
            "CI_CCACHE_DIR": str(cache_dir),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "TOKENSPEED_CI_FORK_PR": fork_pr,
            "WORKSPACE": str(workspace),
        }
    )
    for name in (
        "CCACHE_BASEDIR",
        "CCACHE_COMPILERCHECK",
        "CCACHE_COMPILERTYPE",
        "CCACHE_DIR",
        "CCACHE_MAXSIZE",
        "CCACHE_NOHASHDIR",
        "CCACHE_READONLY",
        "CCACHE_SLOPPINESS",
        "CCACHE_STATSLOG",
        "CCACHE_TEMPDIR",
        "CCACHE_UMASK",
        "TOKENSPEED_KERNEL_NVCC_LAUNCHER",
    ):
        env.pop(name, None)

    result = run_bash(
        "configure_nvcc_cache >/dev/null; "
        "printf '%s|' "
        '"${TOKENSPEED_KERNEL_NVCC_LAUNCHER}" "${CCACHE_DIR}" '
        '"${CCACHE_BASEDIR}" "${CCACHE_COMPILERTYPE}" '
        '"${CCACHE_COMPILERCHECK}" "${CCACHE_SLOPPINESS}" '
        '"${CCACHE_NOHASHDIR}" "${CCACHE_MAXSIZE}" "${CCACHE_UMASK}" '
        '"${CCACHE_READONLY}" '
        '"${CCACHE_STATSLOG}"',
        env,
    )

    assert result.stdout == "|".join(
        [
            "ccache",
            str(cache_dir),
            str(workspace),
            "nvcc",
            "%compiler% --version; g++ --version",
            "include_file_ctime,include_file_mtime",
            "1",
            "100G",
            "002",
            read_only,
            str(workspace / ".ci-artifacts" / "ccache-stats.log"),
            "",
        ]
    )
    assert cache_dir.is_dir()
    assert (workspace / ".ccache-tmp").is_dir()


def test_nvcc_cache_falls_back_when_cache_directory_is_unavailable(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_ccache = bin_dir / "ccache"
    fake_ccache.write_text("#!/bin/bash\nexit 0\n")
    fake_ccache.chmod(0o755)
    blocked_path = tmp_path / "not-a-directory"
    blocked_path.write_text("blocked")
    env = os.environ.copy()
    env.update(
        {
            "CI_CCACHE_DIR": str(blocked_path),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "WORKSPACE": str(tmp_path / "checkout"),
        }
    )
    env.pop("CCACHE_DIR", None)
    env.pop("TOKENSPEED_KERNEL_NVCC_LAUNCHER", None)

    result = run_bash(
        "configure_nvcc_cache >/dev/null 2>/dev/null; "
        'printf "%s|%s" "${CI_CCACHE_DIR:-}" '
        '"${TOKENSPEED_KERNEL_NVCC_LAUNCHER:-}"',
        env,
    )

    assert result.stdout == "|"


def test_cached_remote_wheel_downloads_only_once(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    cache_dir = tmp_path / "wheelhouse"
    bin_dir.mkdir()
    cache_dir.mkdir()
    fake_curl = bin_dir / "curl"
    fake_curl.write_text("""#!/bin/bash
set -e
printf 'called\\n' >> "${CURL_CALLS}"
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--output" ]; then
        printf 'complete wheel' > "$2"
        exit 0
    fi
    shift
done
exit 1
""")
    fake_curl.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "CI_WHEEL_CACHE_DIR": str(cache_dir),
            "CURL_CALLS": str(tmp_path / "curl-calls"),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    (cache_dir / "pkg.whl").write_text("bad wheel")
    expected_sha256 = sha256(b"complete wheel").hexdigest()
    command = f'for i in 1 2 3 4; do cache_remote_wheel "https://example.test/pkg.whl" "{expected_sha256}" & done; wait'
    result = run_bash(command, env)
    assert result.stdout.splitlines() == [str(cache_dir / "pkg.whl")] * 4
    assert (tmp_path / "curl-calls").read_text().splitlines() == ["called"]
    assert (cache_dir / "pkg.whl").read_text() == "complete wheel"
    assert not list(cache_dir.glob("*.tmp.*"))
