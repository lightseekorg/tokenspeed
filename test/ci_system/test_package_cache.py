import os
import subprocess
from hashlib import sha256
from pathlib import Path

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
    result = run_bash(
        'configure_package_cache; printf "%s|%s" "${PIP_CACHE_DIR:-}" "${CI_WHEEL_CACHE_DIR:-}"',
        env,
    )
    assert result.stdout == "|"


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
    result = run_bash(
        'configure_package_cache >/dev/null; printf "%s|%s" "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"',
        env,
    )
    assert result.stdout == f"{tmp_path / 'pip'}|{tmp_path / 'wheelhouse'}"
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
    result = run_bash(
        'configure_package_cache >/dev/null; printf "%s|%s" "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"',
        env,
    )
    assert result.stdout == f"{tmp_path / 'pip'}|{tmp_path / 'wheelhouse'}"


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
