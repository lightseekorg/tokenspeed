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
    @pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
    @pytest.mark.parametrize("rank", [-1, 8])
    def test_out_of_range_raises(self, mode, rank):
        processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
        obj = GenerateReqInput(
            text="hello", sampling_params={}, data_parallel_rank=rank
        )
        with pytest.raises((ValueError), match=(r"\[0, 8\)")):
            processor.validate_request(obj)

    @pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
    def test_in_range_kept(self, mode):
        processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=7)
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) == (7)

    @pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
    def test_no_attn_dp_pin_dropped_silently(self, mode):
        processor = _make_input_processor(has_attn_dp=False, disaggregation_mode=mode)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=3)
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) is None

    def test_non_int_pin_rejected_everywhere(self, subtests):
        # bool passes a bare range compare and floats/lists fail the int|None
        # typed msgpack decode downstream — reject at the choke point.
        for bad in (True, 1.5, "3"):
            for mode in ("null", "prefill", "decode"):
                processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
                obj = GenerateReqInput(
                    text="hello", sampling_params={}, data_parallel_rank=bad
                )
                with subtests.test(bad=bad, mode=mode):
                    with pytest.raises((ValueError), match=("must be an int")):
                        processor.validate_request(obj)

    @pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
    def test_batch_list_out_of_range_raises(self, mode):
        processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            data_parallel_rank=[1, 99],
        )
        with pytest.raises((ValueError), match=("99")):
            processor.validate_request(obj)

    @pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
    def test_batch_list_partial_pin_kept(self, mode):
        processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            data_parallel_rank=[None, 3],
        )
        processor.validate_request(obj)
        assert (obj.data_parallel_rank) == ([None, 3])

    @pytest.mark.parametrize(("mode", "rank"), [("prefill", 2), ("decode", 7)])
    def test_disagg_scalar_pin_and_rooms_survive_batch_normalization(self, mode, rank):
        processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            bootstrap_room=[10, 18],
            data_parallel_rank=rank,
        )
        processor.validate_request(obj)
        obj.normalize_batch_and_arguments()
        assert obj.data_parallel_rank == [rank, rank]
        assert obj.bootstrap_room == [10, 18]
        assert obj[0].data_parallel_rank == rank
        assert obj[1].data_parallel_rank == rank


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


class TestTokenizeBridge:
    @pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
    @pytest.mark.asyncio(loop_scope="function")
    async def test_pin_reaches_tokenized_request(self, mode):
        # The only bridge that puts the pin onto the wire is
        # tokenize_one_request copying obj.data_parallel_rank into
        # TokenizedGenerateReqInput.
        processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
        engine = processor.engine
        engine.server_args.enable_prefix_caching = False
        engine.server_args.language_model_only = False
        engine.server_args.enable_output_logprobs = False
        engine.model_config = SimpleNamespace(
            is_multimodal=False,
            is_multimodal_active=False,
            hf_config=None,
            vocab_size=32000,
        )
        engine.tokenizer = None
        engine.max_req_input_len = 1024
        engine.context_len = 2048

        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 4},
            data_parallel_rank=6,
            bootstrap_room=14,
        )
        processor.validate_request(obj)
        obj.normalize_batch_and_arguments()
        tokenized = await processor.tokenize_one_request(obj)
        assert isinstance((tokenized), (TokenizedGenerateReqInput))
        assert (tokenized.data_parallel_rank) == (6)
        assert tokenized.bootstrap_room == 14


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
