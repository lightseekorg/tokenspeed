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


def test_collector_requires_aligned_ranks_and_complete_tiers():
    from tokenspeed.runtime.execution.drafter.verify_calibration import (
        VerifyCalibrationCollector,
    )

    collector = VerifyCalibrationCollector(
        batch_size=4,
        verify_width=6,
        context_bounds=(100, 200),
        capacities=(12, 24),
        world_size=2,
        minimum_samples=2,
        warmup_samples=0,
    )
    assert collector.next_capacity() == 24
    assert collector.result() is None
    assert not collector.observe(
        step=0,
        query_capacity=24,
        context_bounds=(110, 120),
        rank_steps=[0, 1],
        rank_ms=[1.0, 2.0],
    )
    assert not collector.observe(
        step=0,
        query_capacity=24,
        context_bounds=(90, 120),
        rank_steps=[0, 0],
        rank_ms=[1.0, 2.0],
    )
    for step, q in enumerate((24, 24, 12, 12)):
        assert collector.next_capacity() == q
        assert collector.observe(
            step=step,
            query_capacity=q,
            context_bounds=(110, 120),
            rank_steps=[step, step],
            rank_ms=[q / 12, q / 6],
        )
        assert not collector.observe(
            step=step,
            query_capacity=q,
            context_bounds=(110, 120),
            rank_steps=[step, step],
            rank_ms=[1.0, 2.0],
        )
    assert collector.next_capacity() is None
    result = collector.result()
    assert result.native_step_ms == 4.0
    assert result.adaptive_step_ms == ((12, 2.0),)
    assert result.sample_count == 2


def test_rank_history_alignment_rejects_skew_and_mismatched_metadata():
    from dataclasses import replace

    from tokenspeed.runtime.execution.drafter.verify_calibration import (
        VerifyStepSample,
        align_verify_samples,
    )

    first = VerifyStepSample(10, 4, 12, 120, 130, 1.0)
    second = replace(first, step=11, elapsed_ms=2.0)
    third = replace(first, step=12, elapsed_ms=3.0)
    assert align_verify_samples([[first, second], [second, third]]) == (second,)
    assert align_verify_samples([[first], [replace(first, elapsed_ms=4.0)]]) == (
        replace(first, elapsed_ms=4.0),
    )
    for changes in (
        {"batch_size": 8},
        {"query_capacity": 16},
        {"min_context": 121},
        {"max_context": 131},
        {"elapsed_ms": float("nan")},
        {"elapsed_ms": 0.0},
    ):
        assert not align_verify_samples([[first], [replace(first, **changes)]])
    assert not align_verify_samples([[first, first], [first]])
    assert not align_verify_samples([[first], []])
    assert not align_verify_samples([])


def test_collector_excludes_warmup_per_capacity():
    from tokenspeed.runtime.execution.drafter.verify_calibration import (
        VerifyCalibrationCollector,
    )

    collector = VerifyCalibrationCollector(
        batch_size=4,
        verify_width=6,
        context_bounds=(100, 200),
        capacities=(12, 24),
        world_size=2,
        minimum_samples=2,
        warmup_samples=1,
    )
    for step, (capacity, elapsed, accepted) in enumerate(
        (
            (24, 1000.0, False),
            (24, 2.0, True),
            (24, 4.0, True),
            (12, 1000.0, False),
            (12, 1.0, True),
            (12, 3.0, True),
        )
    ):
        assert (
            collector.observe(
                step=step,
                query_capacity=capacity,
                context_bounds=(120, 130),
                rank_steps=[step, step],
                rank_ms=[elapsed, elapsed],
            )
            is accepted
        )
    profile = collector.result()
    assert profile.native_step_ms == 3.0
    assert profile.adaptive_step_ms == ((12, 2.0),)


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
