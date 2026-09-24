"""Exercise rocJITsu build reuse without installing the ROCm SDK."""

import os
import subprocess
from pathlib import Path

SETUP = Path(__file__).with_name("setup_mi450_sim.sh")


def _write_command(directory: Path, name: str, body: str) -> None:
    command = directory / name
    command.write_text("#!/bin/bash\nset -eu\n" + body)
    command.chmod(0o755)


def _fake_setup(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sim_root = tmp_path / "tokenspeed-mi450-sim"
    config = sim_root / "rocm-systems/emulation/rocjitsu/configs/gfx1250_mi455x.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"max_ticks": 100}\n')
    (sim_root / "rocm-systems/.git").mkdir()

    _write_command(bin_dir, "sudo", 'exec "$@"\n')
    for name in ("apt-get", "pip3", "uv"):
        _write_command(bin_dir, name, "exit 0\n")
    _write_command(
        bin_dir,
        "python3",
        'if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then exit 0; fi\n'
        'exec "${FAKE_REAL_PYTHON}" "$@"\n',
    )
    _write_command(
        bin_dir,
        "rocm-sdk",
        'if [ "$1" = "path" ]; then printf "%s\\n" "${FAKE_ROCM_ROOT}"; fi\n',
    )
    _write_command(bin_dir, "git", "exit 0\n")
    _write_command(
        bin_dir,
        "cmake",
        'if [ "$1" = "--build" ]; then\n'
        '    printf "build\\n" >> "${FAKE_BUILD_LOG}"\n'
        '    if [ "${FAKE_CMAKE_FAIL:-0}" = "1" ]; then exit 41; fi\n'
        '    mkdir -p "$2/tools/rocjitsu"\n'
        '    printf "#!/bin/sh\\n" > "$2/tools/rocjitsu/rocjitsu"\n'
        '    chmod +x "$2/tools/rocjitsu/rocjitsu"\n'
        '    touch "$2/librocjitsu.so"\n'
        "else\n"
        '    while [ "$#" -gt 0 ]; do\n'
        '        if [ "$1" = "-B" ]; then mkdir -p "$2"; break; fi\n'
        "        shift\n"
        "    done\n"
        "fi\n",
    )

    build_log = tmp_path / "builds.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_REAL_PYTHON": os.environ.get("PYTHON", "/usr/bin/python3"),
        "FAKE_ROCM_ROOT": str(tmp_path / "rocm-sdk"),
        "FAKE_BUILD_LOG": str(build_log),
        "TOKENSPEED_MI450_SIM_ROOT": str(sim_root),
        "ROCM_SYSTEMS_REF": "source-a",
        "ROCM_SDK_VERSION": "sdk-a",
    }
    return env, sim_root / "rocjitsu-build", build_log


def _run_setup(env: dict[str, str], *, succeeds: bool = True) -> str:
    result = subprocess.run(
        ["bash", str(SETUP)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
    return result.stdout


def test_rocjitsu_reuses_only_matching_complete_build(tmp_path: Path) -> None:
    env, build_dir, build_log = _fake_setup(tmp_path)
    stamp = build_dir / ".tokenspeed-build-id"

    _run_setup(env)
    assert len(build_log.read_text().splitlines()) == 1
    assert stamp.read_text().strip() == f"source-a:sdk-a:{env['FAKE_ROCM_ROOT']}"

    assert "Reusing cached rocJITsu" in _run_setup(env)
    assert len(build_log.read_text().splitlines()) == 1

    for change in (
        lambda: stamp.unlink(),
        lambda: stamp.write_text("stale\n"),
        lambda: (build_dir / "librocjitsu.so").unlink(),
        lambda: env.update(ROCM_SDK_VERSION="sdk-b"),
        lambda: env.update(ROCM_SYSTEMS_REF="source-b"),
        lambda: env.update(FAKE_ROCM_ROOT=str(tmp_path / "other-sdk")),
    ):
        previous = len(build_log.read_text().splitlines())
        change()
        _run_setup(env)
        assert len(build_log.read_text().splitlines()) == previous + 1


def test_failed_rocjitsu_build_cannot_be_reused(tmp_path: Path) -> None:
    env, build_dir, build_log = _fake_setup(tmp_path)
    _run_setup(env)
    env["ROCM_SDK_VERSION"] = "sdk-b"
    env["FAKE_CMAKE_FAIL"] = "1"

    _run_setup(env, succeeds=False)
    assert not (build_dir / ".tokenspeed-build-id").exists()

    env.pop("FAKE_CMAKE_FAIL")
    _run_setup(env)
    assert len(build_log.read_text().splitlines()) == 3
