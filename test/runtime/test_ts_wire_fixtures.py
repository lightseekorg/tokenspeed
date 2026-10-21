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

"""The Python half of the ``ts-wire`` conformance contract.

``tokenspeed-engine/crates/ts-wire/tests/fixtures/`` holds messages written by
the Python encoder; the Rust crate decodes them and re-encodes them byte for
byte. These tests fail when a Python wire struct changes without the fixtures
being regenerated (and so the Rust schema being updated with them), and check
that the engine's own decoders accept every fixture.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine import zmq_wire  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    AbortReq,
    BatchTokenIDOutSlim,
    MsgpackDecoder,
    TokenizedGenerateReqInput,
)

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-engine"
    / "crates"
    / "ts-wire"
    / "tests"
    / "fixtures"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "ts_wire_fixture_generator", _FIXTURE_DIR / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GENERATOR = _load_generator()
_EXPECTED = _GENERATOR.build_fixtures()


def _frames(name: str) -> list[bytes]:
    return _GENERATOR.read_frames(f"{name}.hex")


@pytest.mark.parametrize("file_name", sorted(_EXPECTED))
def test_fixture_matches_python_encoding(file_name: str) -> None:
    on_disk = (_FIXTURE_DIR / file_name).read_text()
    assert on_disk == _EXPECTED[file_name], (
        f"{file_name} no longer matches the Python encoding. If the wire "
        "schema changed on purpose, run "
        "tokenspeed-engine/crates/ts-wire/tests/fixtures/generate.py and "
        "update the ts-wire structs to match."
    )


def test_no_stale_fixture_files() -> None:
    on_disk = {path.name for path in _FIXTURE_DIR.glob("*.hex")}
    assert on_disk == set(_EXPECTED)


@pytest.mark.parametrize(
    "file_name, rid",
    [
        ("generate_minimal", "r-min"),
        ("generate_text", "r-text"),
        ("generate_multimodal", "r-mm"),
    ],
)
def test_engine_decodes_add_fixtures(file_name: str, rid: str) -> None:
    (request,) = zmq_wire.decode_request_frames(_frames(file_name))
    assert isinstance(request, TokenizedGenerateReqInput)
    assert request.rid == rid


def test_engine_decodes_multimodal_tensors() -> None:
    (request,) = zmq_wire.decode_request_frames(_frames("generate_multimodal"))
    mm = request.multimodal_inputs
    assert mm.mrope_positions.tolist() == [
        list(range(r * 20, r * 20 + 20)) for r in range(3)
    ]
    image, video, audio = mm.mm_items
    assert image.feature.flatten().tolist() == [float(v) for v in range(8)]
    assert video.feature.shape == (16, 16)
    assert audio.feature_shm.shm_name == "/ts_shm_0"


def test_engine_decodes_abort_fixture() -> None:
    aborts = zmq_wire.decode_request_frames(_frames("abort"))
    assert [type(a) for a in aborts] == [AbortReq, AbortReq]
    assert [a.rid for a in aborts] == ["r1", "r2"]


@pytest.mark.parametrize("file_name", ["output_slim", "output_slim_prefix9"])
def test_frontend_decodes_output_fixtures(file_name: str) -> None:
    batch = MsgpackDecoder(BatchTokenIDOutSlim).decode(_frames(file_name))
    assert len(batch.output_ids) == len(batch.rids)
