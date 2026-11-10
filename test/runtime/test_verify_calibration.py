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

"""Cost calibration cannot silently cross runtime identities or contexts."""

import json

import pytest

from tokenspeed.runtime.execution.drafter.verify_calibration import (
    VerifyCalibration,
    VerifyCalibrationCache,
)


def profile():
    return VerifyCalibration.measure(
        batch_size=32,
        min_context=131072,
        max_context=132096,
        verify_width=6,
        native_samples=[20.0, 21.0, 19.0],
        adaptive_samples={128: [14.0, 13.0, 15.0], 160: [17.0, 16.0, 18.0]},
        minimum_samples=3,
    )


def test_measure_and_context_coverage():
    value = profile()
    assert value.native_step_ms == 20.0
    assert value.adaptive_step_ms == ((128, 14.0), (160, 17.0))
    assert value.sample_count == 3
    assert value.covers(batch_size=32, min_context=131072, max_context=132096)
    assert not value.covers(batch_size=31, min_context=131072, max_context=132096)
    assert not value.covers(batch_size=32, min_context=131071, max_context=132096)
    assert not value.covers(batch_size=32, min_context=131072, max_context=132097)


def test_cache_roundtrip_and_configuration_change(tmp_path):
    cache = VerifyCalibrationCache(
        tmp_path, identity={"device": "test", "revision": "a"}
    )
    assert cache.load() is None
    cache.save([profile()])
    assert cache.load() == (profile(),)
    changed = VerifyCalibrationCache(
        tmp_path, identity={"device": "test", "revision": "b"}
    )
    assert changed.load() is None
    # A copied file must not defeat identity validation.
    changed.path.write_bytes(cache.path.read_bytes())
    assert changed.load() is None
    assert not list(tmp_path.glob(".verify-*"))


@pytest.mark.parametrize("payload", ["{", "null", "[]", '{"version": 1}'])
def test_corrupt_cache_is_a_miss(tmp_path, payload):
    cache = VerifyCalibrationCache(tmp_path, identity={"test": 1})
    cache.path.write_text(payload)
    assert cache.load() is None


def test_invalid_cached_profile_and_version_are_misses(tmp_path):
    cache = VerifyCalibrationCache(tmp_path, identity={"test": 1})
    cache.save([profile()])
    raw = json.loads(cache.path.read_text())
    raw["profiles"][0]["native_step_ms"] = -1
    cache.path.write_text(json.dumps(raw))
    assert cache.load() is None
    raw["version"] = 2
    cache.path.write_text(json.dumps(raw))
    assert cache.load() is None


def test_insufficient_measurements_rejected():
    with pytest.raises(ValueError, match="insufficient"):
        VerifyCalibration.measure(
            batch_size=1,
            min_context=1,
            max_context=1,
            verify_width=6,
            native_samples=[1.0],
            adaptive_samples={2: [0.5]},
            minimum_samples=3,
        )


def test_ambiguous_context_profiles_rejected(tmp_path):
    cache = VerifyCalibrationCache(tmp_path, identity={"test": 1})
    with pytest.raises(ValueError, match="overlapping"):
        cache.save([profile(), profile()])


def test_calibration_source_fingerprint_tracks_editable_changes(tmp_path):
    from tokenspeed.runtime.execution.drafter.verify_calibration import (
        calibration_source_fingerprint,
    )

    source = tmp_path / "operator.py"
    source.write_text("value = 1\n")
    first = calibration_source_fingerprint([tmp_path])
    assert calibration_source_fingerprint([tmp_path]) == first
    source.write_text("value = 2\n")
    second = calibration_source_fingerprint([tmp_path])
    assert second != first
    native = tmp_path / "native.so"
    native.write_bytes(b"build one")
    third = calibration_source_fingerprint([tmp_path])
    assert third != second
    native.write_bytes(b"build two with changed size")
    assert calibration_source_fingerprint([tmp_path]) != third
    with pytest.raises(ValueError, match="does not exist"):
        calibration_source_fingerprint([tmp_path / "missing"])
