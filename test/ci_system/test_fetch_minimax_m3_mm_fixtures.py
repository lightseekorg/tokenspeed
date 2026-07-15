from __future__ import annotations

import hashlib
import io

import pytest

from fetch_minimax_m3_mm_fixtures import Fixture, fetch_fixture


def _fixture(data: bytes) -> Fixture:
    return Fixture(name="image.jpg", sha256=hashlib.sha256(data).hexdigest())


def test_fetch_fixture_verifies_and_writes_atomically(tmp_path):
    data = b"fixture-bytes"
    calls = []

    def opener(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(data)

    target = fetch_fixture(_fixture(data), tmp_path, opener=opener)

    assert target.read_bytes() == data
    assert calls == [(_fixture(data).url, 60)]
    assert not target.with_suffix(".jpg.tmp").exists()


def test_fetch_fixture_reuses_verified_file(tmp_path):
    data = b"cached"
    target = tmp_path / "image.jpg"
    target.write_bytes(data)

    def opener(*args, **kwargs):
        raise AssertionError("verified file should not be downloaded again")

    assert fetch_fixture(_fixture(data), tmp_path, opener=opener) == target


def test_fetch_fixture_rejects_wrong_digest(tmp_path):
    def opener(url, *, timeout):
        return io.BytesIO(b"wrong")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetch_fixture(_fixture(b"expected"), tmp_path, opener=opener)
    assert not (tmp_path / "image.jpg").exists()
