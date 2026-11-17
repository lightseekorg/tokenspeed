# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``cute.compile`` binaries persist across processes and stay correct.

Every case runs its compilations in fresh processes sharing one cache
directory, as two server starts would.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an NVIDIA GPU"
)

_CHILD = Path(__file__).with_name("cute_compile_cache_child.py")

_KERNEL = """
import cutlass
import cutlass.cute as cute


@cute.kernel
def _add_kernel(x: cute.Tensor, value: cutlass.Float32):
    tidx, _, _ = cute.arch.thread_idx()
    x[tidx] = x[tidx] + value


@cute.jit
def add(x: cute.Tensor, value: cutlass.Float32, threads: cutlass.Constexpr):
    _add_kernel(x, value).launch(grid=(1, 1, 1), block=(threads, 1, 1))
"""


_TVM_FFI_KERNEL = """
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


@cute.kernel
def _add_kernel(x: cute.Tensor, value: cutlass.Float32):
    tidx, _, _ = cute.arch.thread_idx()
    x[tidx] = x[tidx] + value


@cute.jit
def add(
    x: cute.Tensor,
    value: cutlass.Float32,
    threads: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _add_kernel(x, value).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)
"""


def _spawn(
    cache_dir: Path | None, output: Path, case: str, *extra: str, **env_overrides: str
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("CUTE_DSL_CACHE_DIR", None)
    env.pop("CUTE_DSL_DISABLE_FILE_CACHING", None)
    if cache_dir is not None:
        env["CUTE_DSL_CACHE_DIR"] = str(cache_dir)
    env.update(env_overrides)
    done = subprocess.run(
        [sys.executable, str(_CHILD), case, str(output), *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return done


def _run(
    cache_dir: Path | None, output: Path, case: str, *extra: str, **env_overrides: str
) -> dict:
    done = _spawn(cache_dir, output, case, *extra, **env_overrides)
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    result["stderr"] = done.stderr
    return result


def _entries(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("cute_dsl_*.mlir"))


@pytest.fixture
def kernel_file(tmp_path: Path) -> Path:
    path = tmp_path / "kernel.py"
    path.write_text(_KERNEL)
    return path


def _add(cache_dir: Path, output: Path, kernel_file: Path, *extra: str) -> dict:
    return _run(cache_dir, output, "add", "--kernel-file", str(kernel_file), *extra)


def test_a_second_process_loads_what_the_first_compiled(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    first = _add(cache, tmp_path / "first.pt", kernel_file)
    assert (first["keyed"], first["hits"]) == (1, 0)
    assert len(_entries(cache)) == 1
    second = _add(cache, tmp_path / "second.pt", kernel_file)
    assert (second["keyed"], second["hits"]) == (1, 1)
    assert len(_entries(cache)) == 1
    assert torch.equal(
        torch.load(tmp_path / "first.pt")[0], torch.load(tmp_path / "second.pt")[0]
    )


def test_processes_compiling_at_once_leave_one_loadable_entry(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    env = dict(os.environ, CUTE_DSL_CACHE_DIR=str(cache))
    env.pop("CUTE_DSL_DISABLE_FILE_CACHING", None)
    racers = [
        subprocess.Popen(
            [
                sys.executable,
                str(_CHILD),
                "add",
                str(tmp_path / f"racer{i}.pt"),
                "--kernel-file",
                str(kernel_file),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(4)
    ]
    for racer in racers:
        _, stderr = racer.communicate(timeout=900)
        assert racer.returncode == 0, stderr[-4000:]
    assert len(_entries(cache)) == 1
    assert not list(cache.glob("tmp.*"))
    after = _add(cache, tmp_path / "after.pt", kernel_file)
    assert (after["keyed"], after["hits"]) == (1, 1)


def test_an_edit_to_the_kernel_source_misses(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    _add(cache, tmp_path / "first.pt", kernel_file)
    # A comment changes neither the trace nor the binary; only the source key sees it.
    kernel_file.write_text(_KERNEL + "\n# edited\n")
    edited = _add(cache, tmp_path / "edited.pt", kernel_file)
    assert edited["hits"] == 0
    assert len(_entries(cache)) == 2


def test_another_gpu_architecture_misses(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    major, minor = torch.cuda.get_device_capability()
    _add(cache, tmp_path / "first.pt", kernel_file)
    other = _add(
        cache,
        tmp_path / "other.pt",
        kernel_file,
        "--capability",
        f"{major},{minor + 1}",
    )
    assert other["hits"] == 0
    assert len(_entries(cache)) == 2


def test_other_compile_options_miss(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    _add(cache, tmp_path / "first.pt", kernel_file)
    other = _add(
        cache, tmp_path / "other.pt", kernel_file, "--options", "--opt-level 2"
    )
    assert other["hits"] == 0
    assert len(_entries(cache)) == 2


def test_tvm_ffi_stream_conventions_keep_separate_entries(tmp_path):
    # An explicit stream and the TVM-FFI environment stream trace the same IR;
    # loading one's entry for the other fails at its first call.
    kernel_file = tmp_path / "kernel.py"
    kernel_file.write_text(_TVM_FFI_KERNEL)
    cache = tmp_path / "cache"

    def run(name: str, stream: str) -> dict:
        return _run(
            cache,
            tmp_path / f"{name}.pt",
            "add_tvm_ffi",
            "--kernel-file",
            str(kernel_file),
            "--stream",
            stream,
        )

    explicit = run("explicit", "explicit")
    env = run("env", "env")
    assert (explicit["hits"], env["hits"]) == (0, 0)
    assert len(_entries(cache)) == 2
    again = {stream: run(f"{stream}_again", stream) for stream in ("explicit", "env")}
    assert again["explicit"]["hits"] == again["env"]["hits"] == 1
    expected = torch.arange(32, dtype=torch.float32) + 1.5
    for name in ("explicit", "env", "explicit_again", "env_again"):
        assert torch.equal(torch.load(tmp_path / f"{name}.pt")[0], expected)


def test_an_unset_cache_dir_persists_under_the_cache_home(tmp_path, kernel_file):
    cache_home = tmp_path / "cache_home"
    first = _run(
        None,
        tmp_path / "first.pt",
        "add",
        "--kernel-file",
        str(kernel_file),
        XDG_CACHE_HOME=str(cache_home),
    )
    assert first["hits"] == 0
    assert len(_entries(cache_home / "tokenspeed" / "cute-dsl")) == 1
    second = _run(
        None,
        tmp_path / "second.pt",
        "add",
        "--kernel-file",
        str(kernel_file),
        XDG_CACHE_HOME=str(cache_home),
    )
    assert second["hits"] == 1


def test_an_unwritable_cache_dir_turns_the_cache_off(tmp_path, kernel_file):
    # cutlass-dsl creates the directory outside its error handling when it
    # writes an entry; a directory under a regular file can never be created.
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    result = _add(blocker / "cache", tmp_path / "out.pt", kernel_file)
    assert result["installed"] is False
    assert result["stderr"].count("CuTe DSL compile cache off") == 1, result["stderr"]
    expected = torch.arange(32, dtype=torch.float32) + 1.5
    assert torch.equal(torch.load(tmp_path / "out.pt")[0], expected)


def test_kept_compiler_artifacts_are_never_served_from_the_cache(tmp_path, kernel_file):
    cache, dump = tmp_path / "cache", tmp_path / "dump"
    dump.mkdir()
    for name in ("first", "second"):
        for kept in dump.iterdir():
            kept.unlink()
        result = _add(
            cache,
            tmp_path / f"{name}.pt",
            kernel_file,
            f"--options=--keep-ptx --keep-cubin --dump-dir {dump}",
        )
        assert result["keyed"] == 0
        assert sorted(path.suffix for path in dump.iterdir()) == [".cubin", ".ptx"]
    assert not _entries(cache)


def test_disabling_file_caching_writes_no_entry(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    for name in ("first", "second"):
        result = _run(
            cache,
            tmp_path / f"{name}.pt",
            "add",
            "--kernel-file",
            str(kernel_file),
            CUTE_DSL_DISABLE_FILE_CACHING="1",
        )
        assert result["keyed"] == 0
    assert not _entries(cache)


def test_dropped_compiled_functions_are_released(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    for name in ("first", "second"):
        result = _run(
            cache,
            tmp_path / f"{name}.pt",
            "add_dropped",
            "--kernel-file",
            str(kernel_file),
        )
        assert result["keyed"] == 3
        assert result["alive"] == 0
    assert result["hits"] == 3


def test_a_corrupt_entry_is_recompiled_and_replaced(tmp_path, kernel_file):
    cache = tmp_path / "cache"
    first = _add(cache, tmp_path / "first.pt", kernel_file)
    (entry,) = _entries(cache)
    entry.write_bytes(b"not an MLIR bytecode file")
    recovered = _add(cache, tmp_path / "recovered.pt", kernel_file)
    assert recovered["hits"] == 0
    assert torch.equal(
        torch.load(tmp_path / "first.pt")[0], torch.load(tmp_path / "recovered.pt")[0]
    )
    again = _add(cache, tmp_path / "again.pt", kernel_file)
    assert (first["hits"], again["hits"]) == (0, 1)


@pytest.mark.parametrize("change", ["untested-version", "renamed-parameters"])
def test_an_unverified_cutlass_dsl_leaves_the_cache_off(tmp_path, kernel_file, change):
    cache = tmp_path / "cache"
    result = _add(
        cache,
        tmp_path / "out.pt",
        kernel_file,
        "--compiles",
        "2",
        "--pretend",
        change,
    )
    assert result["installed"] is False
    assert result["keyed"] is None
    assert result["stderr"].count("CuTe DSL compile cache off") == 1, result["stderr"]
    assert not _entries(cache)
    expected = torch.arange(32, dtype=torch.float32) + 3.0
    assert torch.equal(torch.load(tmp_path / "out.pt")[0], expected)


@pytest.mark.parametrize(
    "change,error",
    [("no-cache-shape", "TypeError"), ("keying-error", "AttributeError")],
)
def test_an_unexpected_call_on_a_verified_cutlass_dsl_raises(
    tmp_path, kernel_file, change, error
):
    # On a verified release these are bugs: they fail the compilation instead
    # of silently disabling the cache.
    done = _spawn(
        tmp_path / "cache",
        tmp_path / "out.pt",
        "add",
        "--kernel-file",
        str(kernel_file),
        "--pretend",
        change,
    )
    assert done.returncode != 0
    assert error in done.stderr, done.stderr[-4000:]
    assert not _entries(tmp_path / "cache")


@pytest.mark.parametrize(
    "case,kernel",
    [("stream_topk", "GuardMerge"), ("gated_residual", "FusedGatedResidualKernel")],
)
def test_a_loaded_kernel_matches_a_fresh_compile_bitwise(tmp_path, case, kernel):
    cache = tmp_path / "cache"
    fresh = _run(cache, tmp_path / "fresh.pt", case)
    assert any(kernel in name for name in fresh["kernels"]), fresh["kernels"]
    assert fresh["keyed"] > 0 and fresh["hits"] == 0
    loaded = _run(cache, tmp_path / "loaded.pt", case)
    assert loaded["hits"] == loaded["keyed"] == fresh["keyed"]
    for expected, actual in zip(
        torch.load(tmp_path / "fresh.pt"),
        torch.load(tmp_path / "loaded.pt"),
        strict=True,
    ):
        assert torch.equal(expected, actual)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
