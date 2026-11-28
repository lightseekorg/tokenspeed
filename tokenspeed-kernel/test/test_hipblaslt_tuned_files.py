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

"""Committed TunableOp BLAS files, their lookup and the online-tuning switch; no GPU."""

from __future__ import annotations

import contextlib
import glob
import os
from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel.ops.gemm import hipblaslt
from tokenspeed_kernel.ops.gemm._hipblaslt import tuning

# The keys TunableOp stamps on ROCm; a file missing one is rejected at load.
_STAMP_KEYS = {
    "PT_VERSION",
    "HIP_VERSION",
    "HIPBLASLT_VERSION",
    "ROCBLAS_VERSION",
    "GCN_ARCH_NAME",
}


def test_file_is_named_after_device():
    assert tuning.hipblaslt_tuned_file("AMD Instinct MI355X") == os.path.join(
        tuning.TUNED_CONFIG_DIR, "device_name=AMD_Instinct_MI355X.csv"
    )


def _committed_files() -> list[str]:
    return sorted(glob.glob(os.path.join(tuning.TUNED_CONFIG_DIR, "*.csv")))


def test_no_committed_device_file_is_named_after_a_generic_device_name():
    generic = {
        tuning.hipblaslt_tuned_file(name) for name in tuning.GENERIC_DEVICE_NAMES
    }
    assert not generic & set(_committed_files())


def test_tuning_refuses_a_default_device_file_for_a_generic_name():
    name = next(iter(tuning.GENERIC_DEVICE_NAMES))
    with (
        patch.object(torch.cuda, "get_device_name", return_value=name),
        pytest.raises(ValueError, match="generic device name"),
    ):
        # Raises before TunableOp or the GPU are touched.
        tuning.tune_hipblaslt_solutions(
            shapes=[(64, 64)], rows=[8], dtype=torch.bfloat16, cold_bytes=1
        )


def test_mi355x_file_is_committed():
    assert os.path.exists(tuning.hipblaslt_tuned_file("AMD Instinct MI355X"))
    assert os.path.exists(os.path.join(tuning.TUNED_CONFIG_DIR, "gcn_arch=gfx950.csv"))


@pytest.mark.parametrize("path", _committed_files(), ids=os.path.basename)
def test_committed_file_is_wellformed(path):
    name = os.path.basename(path)
    assert name.startswith(("device_name=", "gcn_arch=")) and " " not in name
    assert set(tuning.hipblaslt_file_validators(path)) == _STAMP_KEYS
    entries = 0
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split(",")
            assert len(fields) in (3, 4), line
            if fields[0] == "Validator":
                continue
            op, params, solution, ms = fields
            assert op.startswith("GemmTunableOp_"), line
            assert solution.startswith(("Gemm_Hipblaslt_", "Gemm_Rocblas_")), line
            assert float(ms) > 0
            entries += 1
    assert entries > 0


_GFX950 = "gfx950:sramecc+:xnack-"


def _gemm_keys(path) -> set[tuple[str, str]]:
    with open(path) as f:
        return {
            tuple(line.split(",")[:2]) for line in f if not line.startswith("Validator")
        }


@pytest.mark.parametrize(
    "device_name, arch_name, files, rows",
    [
        # Unchanged from the single MI355X file this split replaced.
        (
            "AMD Instinct MI355X",
            _GFX950,
            ["gcn_arch=gfx950.csv", "device_name=AMD_Instinct_MI355X.csv"],
            105,
        ),
        # A generic name some MI355X drivers report, and the MI350X, share the
        # architecture's rows only.
        ("AMD Radeon Graphics", _GFX950, ["gcn_arch=gfx950.csv"], 75),
        ("AMD Instinct MI350X", _GFX950, ["gcn_arch=gfx950.csv"], 75),
        ("AMD Instinct MI355X", "gfx942", [], 0),
    ],
)
def test_committed_files_of_a_gpu(device_name, arch_name, files, rows):
    paths = tuning.find_hipblaslt_tuned_files(device_name, arch_name)
    assert [os.path.basename(path) for path in paths] == files
    assert sum(len(_gemm_keys(path)) for path in paths) == rows


@pytest.mark.parametrize("path", _committed_files(), ids=os.path.basename)
def test_device_file_lists_no_gemm_of_its_architecture_file(path):
    # A device file adds GEMMs; one listed in both would leave the
    # architecture file's same-bits row to load order.
    arch_name = tuning.hipblaslt_file_validators(path)["GCN_ARCH_NAME"]
    shared = [
        other
        for other in tuning.find_hipblaslt_tuned_files("", arch_name)
        if other != path and _gemm_keys(other) & _gemm_keys(path)
    ]
    assert not shared


def test_device_file_of_another_architecture_is_skipped_loudly(tmp_path, caplog):
    with open(tmp_path / "device_name=AMD_Instinct_MI355X.csv", "w") as f:
        for key in sorted(_STAMP_KEYS - {"GCN_ARCH_NAME"}):
            f.write(f"Validator,{key},1\n")
        f.write("Validator,GCN_ARCH_NAME,gfx942\n")
    with (
        patch.object(tuning, "TUNED_CONFIG_DIR", str(tmp_path)),
        caplog.at_level("WARNING", logger=tuning.logger.name),
    ):
        assert tuning.find_hipblaslt_tuned_files("AMD Instinct MI355X", _GFX950) == []
    assert "stamped for 'gfx942'" in caplog.text


def test_unreadable_committed_file_is_skipped_loudly(tmp_path, caplog):
    # A directory named like a tuned file cannot be read as one.
    (tmp_path / "gcn_arch=broken.csv").mkdir()
    with (
        patch.object(tuning, "TUNED_CONFIG_DIR", str(tmp_path)),
        caplog.at_level("WARNING", logger=tuning.logger.name),
    ):
        assert tuning.find_hipblaslt_tuned_files("AMD Instinct MI355X", _GFX950) == []
    assert "gcn_arch=broken.csv" in caplog.text


class _Properties:
    """Device properties with an architecture that has no tuned file."""

    gcnArchName = "gfx000"


def test_gpu_without_file_is_skipped_loudly(caplog):
    with (
        patch.object(torch.version, "hip", "7.2"),
        patch.object(torch.cuda, "get_device_name", return_value="No Such GPU"),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_properties", return_value=_Properties()),
        caplog.at_level("WARNING", logger=tuning.logger.name),
    ):
        # Returns before TunableOp or the GPU are touched.
        assert not tuning.load_hipblaslt_tuned_solutions()
    assert "GEMMs run untuned" in caplog.text
    assert "device_name=No_Such_GPU.csv" in caplog.text


def test_default_load_is_noop_off_rocm():
    hipblaslt._load_default.cache_clear()
    try:
        with (
            patch.object(torch.version, "hip", None),
            patch.object(tuning, "load_hipblaslt_tuned_solutions") as load,
        ):
            assert not hipblaslt.load_hipblaslt_tuned_solutions()
            assert not hipblaslt.load_hipblaslt_tuned_solutions()
        load.assert_not_called()
    finally:
        hipblaslt._load_default.cache_clear()


def test_default_load_runs_once_per_process():
    hipblaslt._load_default.cache_clear()
    try:
        with (
            patch.object(torch.version, "hip", "7.2"),
            patch.object(
                tuning, "load_hipblaslt_tuned_solutions", return_value=False
            ) as load,
        ):
            hipblaslt.load_hipblaslt_tuned_solutions()
            hipblaslt.load_hipblaslt_tuned_solutions()
        load.assert_called_once_with()
    finally:
        hipblaslt._load_default.cache_clear()


class _FakeTunable:
    """The ``torch.cuda.tunable`` calls online tuning makes, with their state."""

    def __init__(self, *, enabled: bool, results: int):
        self.enabled = enabled
        # TunableOp tunes by default; loading a committed file turns it off.
        self.tuning = not enabled
        self.filename = "committed.csv" if enabled else ""
        self.rotating_mb = 0
        self.results = [("op", "params", "solution", 1.0)] * results
        self.calls = []

    def is_enabled(self):
        return self.enabled

    def enable(self, value):
        self.calls.append(("enable", value))
        self.enabled = value

    def tuning_enable(self, value):
        self.calls.append(("tuning_enable", value))
        self.tuning = value

    def record_untuned_enable(self, value):
        self.calls.append(("record_untuned_enable", value))

    def set_filename(self, name, insert_device_ordinal=False):
        self.calls.append(("set_filename", name))
        self.filename = name

    def get_rotating_buffer_size(self):
        return self.rotating_mb

    def set_rotating_buffer_size(self, mb):
        self.rotating_mb = mb

    def get_results(self):
        return self.results

    def get_validators(self):
        return [("PT_VERSION", "2.9")]

    def read_file(self, path):
        self.calls.append(("read_file", path, self.tuning))
        with open(path) as f:
            rows = [line.rstrip("\n").split(",") for line in f]
        self.results = self.results + [
            (op, params, solution, float(ms))
            for op, params, solution, ms in (r for r in rows if r[0] != "Validator")
        ]
        return True


@contextlib.contextmanager
def _patched_tunable(fake):
    with (
        patch.object(torch.cuda, "tunable", fake),
        patch.object(torch.cuda, "synchronize"),
    ):
        yield


def test_online_tuning_extends_loaded_file_in_memory():
    fake = _FakeTunable(enabled=True, results=30)
    with _patched_tunable(fake):
        with tuning.hipblaslt_online_tuning(cold_bytes=512 * 2**20):
            assert fake.tuning and fake.enabled
            # New results go nowhere on disk; the committed file stays as is.
            assert fake.filename == os.devnull
            assert fake.rotating_mb == 512
            fake.results = fake.results + [("op", "new", "solution", 1.0)]
    assert not fake.tuning and fake.enabled  # replay continues
    assert fake.filename == os.devnull
    assert fake.rotating_mb == 0
    # The loaded file's TunableOp state is kept, not reset.
    assert ("enable", True) not in fake.calls


def test_online_tuning_without_file_avoids_default_file():
    fake = _FakeTunable(enabled=False, results=0)
    with _patched_tunable(fake):
        with tuning.hipblaslt_online_tuning():
            assert fake.tuning and fake.enabled
    # The file name is set before TunableOp's first lookup reads it.
    assert fake.calls.index(("set_filename", os.devnull)) < fake.calls.index(
        ("enable", True)
    )
    assert not fake.tuning and fake.enabled


def test_online_tuning_stops_on_error():
    fake = _FakeTunable(enabled=True, results=30)
    with _patched_tunable(fake):
        with pytest.raises(RuntimeError):
            with tuning.hipblaslt_online_tuning():
                raise RuntimeError("warmup failed")
    assert not fake.tuning and fake.rotating_mb == 0


_COMMITTED = ("GemmTunableOp_BFloat16_TN", "tn_5120_4_6144", "Gemm_Hipblaslt_1", 0.01)
_ONLINE = ("GemmTunableOp_BFloat16_TN", "tn_5120_8_6144", "Gemm_Rocblas_2", 0.02)


def _online_tuning(fake, cache, *, write=True, tuned=((),)):
    # One tuning block per entry of ``tuned``, inside one cache block.
    with (
        _patched_tunable(fake),
        patch.object(tuning, "_current_validators", lambda: {"PT_VERSION": "2.9"}),
    ):
        with tuning.hipblaslt_online_tuning_cache(cache, write=write):
            for block in tuned:
                with tuning.hipblaslt_online_tuning():
                    fake.results = fake.results + list(block)


def _rows(path):
    with open(path) as f:
        return f.read().splitlines()


def test_online_tuning_saves_its_picks_for_the_next_start(tmp_path):
    cache = str(tmp_path / "sub" / "env.hipblaslt.csv")
    fake = _FakeTunable(enabled=True, results=0)
    fake.results = [_COMMITTED]
    _online_tuning(fake, cache, tuned=[[_ONLINE]])
    # The committed rows stay in their file; the new pick is saved, stamped.
    saved = ["Validator,PT_VERSION,2.9", ",".join(map(str, _ONLINE))]
    assert _rows(cache) == saved

    # The next start reads the picks before tuning starts and tunes nothing.
    fake = _FakeTunable(enabled=True, results=0)
    fake.results = [_COMMITTED]
    _online_tuning(fake, cache)
    assert ("read_file", cache, False) in fake.calls
    assert fake.results == [_COMMITTED, _ONLINE]
    assert _rows(cache) == saved


def test_online_tuning_saves_the_picks_of_every_block(tmp_path):
    cache = str(tmp_path / "env.hipblaslt.csv")
    _online_tuning(_FakeTunable(enabled=False, results=0), cache, tuned=[[_ONLINE]])
    later = ("GemmTunableOp_BFloat16_TN", "tn_5120_16_6144", "Gemm_Hipblaslt_3", 0.03)
    # A later start reads the first block's pick, with tuning off, and saves
    # it with the picks of both of its blocks.
    fake = _FakeTunable(enabled=False, results=0)
    _online_tuning(fake, cache, tuned=[[], [later]])
    assert ("read_file", cache, False) in fake.calls
    assert _rows(cache) == [
        "Validator,PT_VERSION,2.9",
        ",".join(map(str, _ONLINE)),
        ",".join(map(str, later)),
    ]


def test_online_tuning_keeps_picks_in_process_without_write(tmp_path):
    cache = str(tmp_path / "env.hipblaslt.csv")
    _online_tuning(_FakeTunable(enabled=False, results=0), cache, write=False)
    assert not os.path.exists(cache)


def test_online_tuning_writes_no_cache_on_error(tmp_path):
    cache = str(tmp_path / "env.hipblaslt.csv")
    with pytest.raises(RuntimeError):
        with (
            _patched_tunable(_FakeTunable(enabled=False, results=0)),
            patch.object(tuning, "_current_validators", lambda: {}),
        ):
            with tuning.hipblaslt_online_tuning_cache(cache, write=True):
                with tuning.hipblaslt_online_tuning():
                    raise RuntimeError("warmup failed")
    assert not os.path.exists(cache)


@pytest.mark.parametrize(
    "rows",
    [
        # Another stack's picks.
        ["Validator,PT_VERSION,2.8", ",".join(map(str, _ONLINE))],
        # A GEMM a committed file lists now.
        ["Validator,PT_VERSION,2.9", ",".join(map(str, _COMMITTED))],
    ],
)
def test_stale_online_cache_is_retuned_and_rewritten(tmp_path, caplog, rows):
    cache = tmp_path / "env.hipblaslt.csv"
    cache.write_text("\n".join(rows) + "\n")
    fake = _FakeTunable(enabled=True, results=0)
    fake.results = [_COMMITTED]
    _online_tuning(fake, str(cache), tuned=[[_ONLINE]])
    assert not any(call[0] == "read_file" for call in fake.calls)
    assert "Ignoring online-tuned BLAS solutions" in caplog.text
    assert _rows(cache) == ["Validator,PT_VERSION,2.9", ",".join(map(str, _ONLINE))]


@pytest.mark.parametrize(
    "hip, env, unified, expected",
    [
        ("7.2", None, None, False),
        ("7.2", "0", None, False),
        ("7.2", "1", None, True),
        (None, "1", None, False),
        # TOKENSPEED_KERNEL_AUTOTUNE=1 turns it on unless its own flag is set.
        ("7.2", None, "1", True),
        ("7.2", "0", "1", False),
        (None, None, "1", False),
    ],
)
def test_autotune_flag(monkeypatch, hip, env, unified, expected):
    for name, value in (
        (hipblaslt.AUTOTUNE_ENV, env),
        ("TOKENSPEED_KERNEL_AUTOTUNE", unified),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with patch.object(torch.version, "hip", hip):
        assert hipblaslt.hipblaslt_autotune_enabled() is expected


def test_online_tuning_is_noop_off_rocm():
    with (
        patch.object(torch.version, "hip", None),
        patch.object(tuning, "hipblaslt_online_tuning") as online,
        patch.object(tuning, "hipblaslt_online_tuning_cache") as cache,
    ):
        with hipblaslt.hipblaslt_online_tuning_cache("unused.csv", write=True):
            with hipblaslt.hipblaslt_online_tuning():
                pass
    online.assert_not_called()
    cache.assert_not_called()
