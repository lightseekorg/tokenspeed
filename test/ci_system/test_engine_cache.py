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

import os
import shutil
import time
from pathlib import Path

import pytest
from engine_cache import (
    WHEEL_CACHE_RETENTION_SECONDS,
    build_requirements,
    environment_digest,
    prune_wheel_cache,
    publish_wheel,
    source_digest,
    wheel_cache_root,
)


def test_wheel_key_tracks_scheduler_source_across_checkouts(tmp_path: Path):
    first = tmp_path / "first"
    engine = first / "tokenspeed-engine"
    scheduler = first / "tokenspeed-scheduler"
    engine.mkdir(parents=True)
    scheduler.mkdir()
    (engine / "Cargo.lock").write_text("locked dependencies")
    (scheduler / "scheduler.cpp").write_text("scheduler implementation")
    (engine / "scheduler").symlink_to("../tokenspeed-scheduler")
    second = tmp_path / "second"
    shutil.copytree(first, second, symlinks=True)
    original = source_digest(first)
    assert source_digest(second) == original

    (engine / "target").mkdir()
    (engine / "target" / "ts-host").write_bytes(b"compiled host")
    assert source_digest(first) == original

    (second / "tokenspeed-scheduler" / "scheduler.cpp").write_text("changed scheduler")
    assert source_digest(second) != original
    (second / "tokenspeed-scheduler" / "scheduler.cpp").unlink()
    assert source_digest(second) != original


def test_fork_and_external_sources_do_not_use_shared_wheel_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_WHEEL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TOKENSPEED_CI_FORK_PR", raising=False)
    monkeypatch.delenv("FETCHCONTENT_SOURCE_DIR_XGRAMMAR", raising=False)
    assert wheel_cache_root() == tmp_path / "tokenspeed-engine"

    monkeypatch.setenv("TOKENSPEED_CI_FORK_PR", "true")
    assert wheel_cache_root() is None
    monkeypatch.setenv("TOKENSPEED_CI_FORK_PR", "false")
    monkeypatch.setenv("FETCHCONTENT_SOURCE_DIR_XGRAMMAR", str(tmp_path / "xgrammar"))
    assert wheel_cache_root() is None


def test_native_include_path_invalidates_environment_key(monkeypatch, tmp_path):
    requirements = build_requirements(Path(__file__).resolve().parents[2])
    monkeypatch.delenv("CPLUS_INCLUDE_PATH", raising=False)
    original = environment_digest(requirements)
    monkeypatch.setenv("CPLUS_INCLUDE_PATH", str(tmp_path))
    assert environment_digest(requirements) != original


def test_publish_wheel_tolerates_concurrent_publish(tmp_path: Path):
    built = tmp_path / "build-new"
    built.mkdir()
    (built / "tokenspeed_engine-1.0.0-cp312.whl").write_text("new wheel")
    destination = tmp_path / "key"
    publish_wheel(built, destination)
    assert (
        destination / "tokenspeed_engine-1.0.0-cp312.whl"
    ).read_text() == "new wheel"

    raced = tmp_path / "build-raced"
    raced.mkdir()
    (raced / "tokenspeed_engine-1.0.0-cp312.whl").write_text("duplicate wheel")
    publish_wheel(raced, destination)
    assert (
        destination / "tokenspeed_engine-1.0.0-cp312.whl"
    ).read_text() == "new wheel"

    with pytest.raises(OSError):
        publish_wheel(tmp_path / "missing", tmp_path / "other")


def test_prune_wheel_cache_removes_only_stale_entries(tmp_path: Path):
    stale_wheel = tmp_path / "stale-wheel"
    stale_wheel.mkdir()
    (stale_wheel / "engine.whl").write_text("stale wheel")
    fresh_wheel = tmp_path / "fresh-wheel"
    fresh_wheel.mkdir()
    (fresh_wheel / "engine.whl").write_text("fresh wheel")
    stale_build = tmp_path / "build-stale"
    stale_build.mkdir()
    fresh_build = tmp_path / "build-fresh"
    fresh_build.mkdir()
    orphan_lock = tmp_path / "orphan.lock"
    orphan_lock.touch()
    kept_lock = tmp_path / "fresh-wheel.lock"
    kept_lock.touch()

    stale = time.time() - WHEEL_CACHE_RETENTION_SECONDS - 1
    for entry in (stale_wheel, stale_build, orphan_lock, kept_lock):
        os.utime(entry, (stale, stale))

    prune_wheel_cache(tmp_path)

    assert not stale_wheel.exists()
    assert not stale_build.exists()
    assert not orphan_lock.exists()
    assert (fresh_wheel / "engine.whl").read_text() == "fresh wheel"
    assert fresh_build.exists()
    assert kept_lock.exists()
