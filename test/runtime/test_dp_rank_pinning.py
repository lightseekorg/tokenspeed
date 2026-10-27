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

"""Attention-DP hard-pin (``data_parallel_rank``) plumbing tests (cheap, no GPU).

Covers ``GenerateReqInput`` normalize/broadcast and ``__getitem__`` fan-out,
the append-only wire position on ``TokenizedGenerateReqInput``, ingress
validation in ``InputProcessor._validate_data_parallel_rank`` (the choke
point every GenerateReqInput path traverses) and the ``Engine`` entry points
forwarding the pin. The frontend client places a pinned request
(``ts-client``'s routing).
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import msgspec.structs  # noqa: E402

from tokenspeed.runtime.engine.input_processor import InputProcessor  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    GenerateReqInput,
    TokenizedGenerateReqInput,
)


def _make_engine_stub(dp_size=8, has_attn_dp=True, disaggregation_mode="null"):
    return SimpleNamespace(
        server_args=SimpleNamespace(
            mapping=SimpleNamespace(
                has_attn_dp=has_attn_dp,
                attn=SimpleNamespace(dp_size=dp_size),
            ),
            disaggregation_mode=disaggregation_mode,
        ),
        logger=mock.MagicMock(),
        is_generation=True,
    )


def _make_input_processor(**engine_kwargs):
    processor = InputProcessor.__new__(InputProcessor)
    processor.engine = _make_engine_stub(**engine_kwargs)
    return processor


class TestGenerateReqInputPlumbing:
    def test_scalar_broadcast_on_n_gt_1(self):
        obj = GenerateReqInput(
            text="hello",
            sampling_params={"n": 3},
            data_parallel_rank=5,
        )
        obj.normalize_batch_and_arguments()
        assert (obj.data_parallel_rank) == ([5, 5, 5])

    def test_list_rejected_on_n_gt_1(self):
        obj = GenerateReqInput(
            text="hello",
            sampling_params={"n": 2},
            data_parallel_rank=[0, 1],
        )
        with pytest.raises((ValueError), match=("data_parallel_rank")):
            obj.normalize_batch_and_arguments()

    def test_getitem_carries_rank(self):
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            data_parallel_rank=[3, 4],
        )
        obj.normalize_batch_and_arguments()
        assert (obj[0].data_parallel_rank) == (3)
        assert (obj[1].data_parallel_rank) == (4)

    def test_single_request_length1_list_unwrapped(self):
        # A list pin surviving normalize on a single request would fail the
        # int|None typed msgpack decode in the DP controller's dispatch loop.
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=[2])
        obj.normalize_batch_and_arguments()
        assert (obj.data_parallel_rank) == (2)

    def test_batch_list_length_mismatch_rejected(self):
        obj = GenerateReqInput(
            text=["a", "b", "c"],
            sampling_params=[{}, {}, {}],
            data_parallel_rank=[0, 1],
        )
        with pytest.raises((ValueError), match=("batch size")):
            obj.normalize_batch_and_arguments()

    def test_tokenized_field_is_last_on_the_wire(self):
        # array_like msgspec structs encode positionally: declaration order IS
        # the wire contract, documented append-only. The SMG-direct msgpack
        # transport decodes TokenizedGenerateReqInput straight off the wire,
        # so the new field must stay the last one.
        fields = msgspec.structs.fields(TokenizedGenerateReqInput)
        assert (fields[-1].name) == ("data_parallel_rank")


class TestIngressValidation:
    def test_colocated_out_of_range_raises(self):
        processor = _make_input_processor(dp_size=8)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=8)
        with pytest.raises((ValueError), match=(r"\[0, 8\)")):
            processor.validate_request(obj)

    def test_colocated_in_range_kept(self):
        processor = _make_input_processor(dp_size=8)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=7)
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) == (7)

    def test_no_attn_dp_pin_dropped_silently(self):
        processor = _make_input_processor(has_attn_dp=False)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=3)
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) is None

    def test_non_int_pin_rejected_everywhere(self, subtests):
        # bool passes a bare range compare and floats/lists fail the int|None
        # typed msgpack decode downstream — reject at the choke point.
        for bad in (True, 1.5, "3"):
            for mode in ("null", "prefill"):
                processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
                obj = GenerateReqInput(
                    text="hello", sampling_params={}, data_parallel_rank=bad
                )
                with subtests.test(bad=bad, mode=mode):
                    with pytest.raises((ValueError), match=("must be an int")):
                        processor.validate_request(obj)

    def test_colocated_batch_list_partial_pin_kept(self):
        processor = _make_input_processor(dp_size=8)
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            data_parallel_rank=[None, 3],
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) == ([None, 3])

    def test_prefill_room_authoritative_conflict_warns_and_drops(self):
        processor = _make_input_processor(dp_size=8, disaggregation_mode="prefill")
        obj = GenerateReqInput(
            text="hello",
            sampling_params={},
            bootstrap_room=10,  # 10 % 8 == 2
            data_parallel_rank=5,
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) is None
        processor.engine.logger.warning.assert_called_once()

    def test_prefill_consistent_pin_dropped_without_warning(self):
        processor = _make_input_processor(dp_size=8, disaggregation_mode="prefill")
        obj = GenerateReqInput(
            text="hello",
            sampling_params={},
            bootstrap_room=10,
            data_parallel_rank=2,  # == 10 % 8, consistent
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) is None
        processor.engine.logger.warning.assert_not_called()

    def test_decode_pin_kept_over_the_room_residue(self):
        # Any decode rank finds the prefill rank from the room, so the pin
        # places the request even where it differs from the residue.
        processor = _make_input_processor(dp_size=8, disaggregation_mode="decode")
        obj = GenerateReqInput(
            text="hello",
            sampling_params={},
            bootstrap_room=10,  # 10 % 8 == 2
            data_parallel_rank=5,
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) == (5)
        processor.engine.logger.warning.assert_not_called()

    def test_decode_out_of_range_pin_warns_and_drops_never_raises(self):
        processor = _make_input_processor(dp_size=8, disaggregation_mode="decode")
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            bootstrap_room=[10, 11],
            data_parallel_rank=[3, 8],
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) == ([3, None])
        processor.engine.logger.warning.assert_called_once()

    def test_disagg_non_int_room_warns_and_drops_never_raises(self):
        # A pinning gateway with a room-format bug must not turn into
        # request rejections on disaggregation engines (never-reject
        # contract) — the room is unverifiable, so warn + drop the pin.
        processor = _make_input_processor(dp_size=8, disaggregation_mode="prefill")
        obj = GenerateReqInput(
            text="hello",
            sampling_params={},
            bootstrap_room="oops",
            data_parallel_rank=2,
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) is None
        processor.engine.logger.warning.assert_called_once()

    def test_disagg_unequal_lists_still_warn(self):
        # A dangling rank with no room to check against is itself a
        # conflict signal — padding must surface it, not zip-truncate it.
        processor = _make_input_processor(dp_size=8, disaggregation_mode="prefill")
        obj = GenerateReqInput(
            text=["a", "b", "c"],
            sampling_params=[{}, {}, {}],
            bootstrap_room=[16, 8],  # both ≡ 0 (mod 8)
            data_parallel_rank=[0, 0, 5],
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) is None
        processor.engine.logger.warning.assert_called_once()

    def test_disagg_scalar_pin_checked_against_every_room(self):
        # zip truncation would pair a scalar pin only with rooms[0]; the
        # conflict against rooms[1] must still warn.
        processor = _make_input_processor(dp_size=8, disaggregation_mode="prefill")
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            bootstrap_room=[10, 11],  # 10 % 8 == 2 matches, 11 % 8 == 3 conflicts
            data_parallel_rank=2,
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) is None
        processor.engine.logger.warning.assert_called_once()


class TestEngineEntrypointForwarding:
    """Engine.generate / async_generate are the documented API surface for
    the pin; each must forward it into GenerateReqInput."""

    @staticmethod
    def _engine():
        try:
            from tokenspeed.runtime.entrypoints.engine import Engine
        except ImportError as exc:  # pragma: no cover - env-dependent import
            pytest.skip(f"entrypoints.engine unimportable: {exc}")

        return Engine.__new__(Engine)

    def test_generate_forwards_pin(self):
        eng = self._engine()
        captured = {}

        def _capture(obj):
            captured["obj"] = obj
            return {}

        eng.llm = SimpleNamespace(generate=_capture, generate_stream=_capture)
        eng.generate(prompt="hi", data_parallel_rank=3)
        assert (captured["obj"].data_parallel_rank) == (3)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_generate_forwards_pin(self):
        eng = self._engine()
        captured = {}

        async def _agen(obj):
            captured["obj"] = obj
            yield {}

        eng.tokenizer_manager = SimpleNamespace(generate_request=_agen)
        result = await eng.async_generate(prompt="hi", data_parallel_rank=3)
        assert (result) == ({})
        assert (captured["obj"].data_parallel_rank) == (3)
