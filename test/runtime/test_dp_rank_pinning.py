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

Covers the end-to-end field revival: ``GenerateReqInput`` normalize/broadcast
and ``__getitem__`` fan-out, the append-only wire position on
``TokenizedGenerateReqInput``, ingress validation in
``InputProcessor._validate_data_parallel_rank`` (the choke point every
GenerateReqInput path traverses), the ``DataParallelController`` pinned
dispatch (defensive fallback, never raising), the ``DPBudget`` reservation
accounting for pinned traffic, and the ``SINGLE_WORKER_ID`` range fix.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import msgspec.structs  # noqa: E402

from tokenspeed.runtime.engine.data_parallel_controller import (  # noqa: E402
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from tokenspeed.runtime.engine.input_processor import InputProcessor  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    GenerateReqInput,
    GetLoadReqOutput,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402


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


class TestGenerateReqInputPlumbing(unittest.TestCase):
    def test_scalar_broadcast_on_n_gt_1(self):
        obj = GenerateReqInput(
            text="hello",
            sampling_params={"n": 3},
            data_parallel_rank=5,
        )
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.data_parallel_rank, [5, 5, 5])

    def test_list_rejected_on_n_gt_1(self):
        obj = GenerateReqInput(
            text="hello",
            sampling_params={"n": 2},
            data_parallel_rank=[0, 1],
        )
        with self.assertRaisesRegex(ValueError, "data_parallel_rank"):
            obj.normalize_batch_and_arguments()

    def test_getitem_carries_rank(self):
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            data_parallel_rank=[3, 4],
        )
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj[0].data_parallel_rank, 3)
        self.assertEqual(obj[1].data_parallel_rank, 4)

    def test_default_is_none(self):
        obj = GenerateReqInput(text="hello", sampling_params={})
        obj.normalize_batch_and_arguments()
        self.assertIsNone(obj.data_parallel_rank)

    def test_single_request_length1_list_unwrapped(self):
        # A list pin surviving normalize on a single request would fail the
        # int|None typed msgpack decode in the DP controller's dispatch loop.
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=[2])
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.data_parallel_rank, 2)

    def test_single_request_longer_list_rejected(self):
        obj = GenerateReqInput(
            text="hello", sampling_params={}, data_parallel_rank=[1, 2]
        )
        with self.assertRaisesRegex(ValueError, "data_parallel_rank"):
            obj.normalize_batch_and_arguments()

    def test_batch_list_length_mismatch_rejected(self):
        obj = GenerateReqInput(
            text=["a", "b", "c"],
            sampling_params=[{}, {}, {}],
            data_parallel_rank=[0, 1],
        )
        with self.assertRaisesRegex(ValueError, "batch size"):
            obj.normalize_batch_and_arguments()

    def test_tokenized_field_is_last_on_the_wire(self):
        # array_like msgspec structs encode positionally: declaration order IS
        # the wire contract, documented append-only. The SMG-direct msgpack
        # transport decodes TokenizedGenerateReqInput straight off the wire,
        # so the new field must stay the last one.
        fields = msgspec.structs.fields(TokenizedGenerateReqInput)
        self.assertEqual(fields[-1].name, "data_parallel_rank")


class TestIngressValidation(unittest.TestCase):
    def test_colocated_out_of_range_raises(self):
        processor = _make_input_processor(dp_size=8)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=8)
        with self.assertRaisesRegex(ValueError, r"\[0, 8\)"):
            processor.validate_request(obj)

    def test_colocated_in_range_kept(self):
        processor = _make_input_processor(dp_size=8)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=7)
        processor.validate_request(obj)
        self.assertEqual(obj.data_parallel_rank, 7)

    def test_no_attn_dp_pin_dropped_silently(self):
        processor = _make_input_processor(has_attn_dp=False)
        obj = GenerateReqInput(text="hello", sampling_params={}, data_parallel_rank=3)
        processor.validate_request(obj)
        self.assertIsNone(obj.data_parallel_rank)

    def test_non_int_pin_rejected_everywhere(self):
        # bool passes a bare range compare and floats/lists fail the int|None
        # typed msgpack decode downstream — reject at the choke point.
        for bad in (True, 1.5, "3"):
            for mode in ("null", "prefill"):
                processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
                obj = GenerateReqInput(
                    text="hello", sampling_params={}, data_parallel_rank=bad
                )
                with self.subTest(bad=bad, mode=mode):
                    with self.assertRaisesRegex(ValueError, "must be an int"):
                        processor.validate_request(obj)

    def test_colocated_batch_list_out_of_range_raises(self):
        processor = _make_input_processor(dp_size=8)
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            data_parallel_rank=[1, 99],
        )
        with self.assertRaisesRegex(ValueError, "99"):
            processor.validate_request(obj)

    def test_colocated_batch_list_partial_pin_kept(self):
        processor = _make_input_processor(dp_size=8)
        obj = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{}, {}],
            data_parallel_rank=[None, 3],
        )
        processor.validate_request(obj)
        self.assertEqual(obj.data_parallel_rank, [None, 3])

    def test_disagg_room_authoritative_conflict_warns_and_drops(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
                obj = GenerateReqInput(
                    text="hello",
                    sampling_params={},
                    bootstrap_room=10,  # 10 % 8 == 2
                    data_parallel_rank=5,
                )
                processor.validate_request(obj)
                self.assertIsNone(obj.data_parallel_rank)
                processor.engine.logger.warning.assert_called_once()

    def test_disagg_consistent_pin_dropped_without_warning(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                processor = _make_input_processor(dp_size=8, disaggregation_mode=mode)
                obj = GenerateReqInput(
                    text="hello",
                    sampling_params={},
                    bootstrap_room=10,
                    data_parallel_rank=2,  # == 10 % 8, consistent
                )
                processor.validate_request(obj)
                self.assertIsNone(obj.data_parallel_rank)
                processor.engine.logger.warning.assert_not_called()

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
        self.assertIsNone(obj.data_parallel_rank)
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
        self.assertIsNone(obj.data_parallel_rank)
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
        self.assertIsNone(obj.data_parallel_rank)
        processor.engine.logger.warning.assert_called_once()


def _make_controller(dp_size=4, disaggregation_mode="null"):
    controller = DataParallelController.__new__(DataParallelController)
    controller.server_args = SimpleNamespace(
        disaggregation_mode=disaggregation_mode,
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=dp_size)),
    )
    controller.workers = [mock.MagicMock() for _ in range(dp_size)]
    controller.round_robin_counter = 0
    controller.load_balance_method = LoadBalanceMethod.SHORTEST_QUEUE
    controller.dp_budget = DPBudget(LoadBalanceMethod.SHORTEST_QUEUE)
    controller.dispatching = mock.MagicMock()
    return controller


def _tokenized(data_parallel_rank=None, bootstrap_room=None):
    return SimpleNamespace(
        data_parallel_rank=data_parallel_rank, bootstrap_room=bootstrap_room
    )


class TestEngineEntrypointForwarding(unittest.IsolatedAsyncioTestCase):
    """Engine.generate / async_generate are the documented API surface for
    the pin; each must forward it into GenerateReqInput."""

    @staticmethod
    def _engine():
        try:
            from tokenspeed.runtime.entrypoints.engine import Engine
        except ImportError as exc:  # pragma: no cover - env-dependent import
            raise unittest.SkipTest(f"entrypoints.engine unimportable: {exc}")

        return Engine.__new__(Engine)

    def test_generate_forwards_pin(self):
        eng = self._engine()
        captured = {}

        def _capture(obj):
            captured["obj"] = obj
            return {}

        eng.llm = SimpleNamespace(generate=_capture, generate_stream=_capture)
        eng.generate(prompt="hi", data_parallel_rank=3)
        self.assertEqual(captured["obj"].data_parallel_rank, 3)

    async def test_async_generate_forwards_pin(self):
        eng = self._engine()
        captured = {}

        async def _agen(obj):
            captured["obj"] = obj
            yield {}

        eng.tokenizer_manager = SimpleNamespace(generate_request=_agen)
        result = await eng.async_generate(prompt="hi", data_parallel_rank=3)
        self.assertEqual(result, {})
        self.assertEqual(captured["obj"].data_parallel_rank, 3)


class TestDispatcherWiring(unittest.TestCase):
    def test_pinned_request_routes_via_request_dispatcher(self):
        # The event loop hands messages to _request_dispatcher; a revert of
        # the init_dispatcher registration back to self.dispatching would
        # disable pinning in production while every direct-call test below
        # still passes.
        controller = _make_controller()
        controller.init_dispatcher()
        req = TokenizedGenerateReqInput(
            rid="wire-1",
            input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            stream=False,
            data_parallel_rank=1,
        )
        controller._request_dispatcher(req)
        controller.workers[1].send_pyobj.assert_called_once_with(req)
        controller.dispatching.assert_not_called()


class TestControllerPinnedDispatch(unittest.TestCase):
    def test_pin_dispatches_directly(self):
        controller = _make_controller()
        req = _tokenized(data_parallel_rank=2)
        controller.dispatch_generate_request(req)
        controller.workers[2].send_pyobj.assert_called_once_with(req)
        controller.dispatching.assert_not_called()

    def test_no_pin_falls_through_to_policy(self):
        controller = _make_controller()
        req = _tokenized()
        controller.dispatch_generate_request(req)
        controller.dispatching.assert_called_once_with(req)
        for worker in controller.workers:
            worker.send_pyobj.assert_not_called()

    def test_out_of_range_pin_falls_back_never_raises(self):
        # The dispatch loop is fed by a fire-and-forget PUSH socket with no
        # per-request error path; a bad pin must degrade, not kill the loop.
        controller = _make_controller(dp_size=4)
        req = _tokenized(data_parallel_rank=4)
        controller.dispatch_generate_request(req)
        controller.dispatching.assert_called_once_with(req)

    def test_malformed_pin_payload_degrades_never_raises(self):
        # bool passes a bare range compare; a list raises TypeError on it —
        # both must fall back to the policy, not escape the dispatch loop.
        for bad in (True, [1], 2.5):
            with self.subTest(bad=bad):
                controller = _make_controller(dp_size=4)
                req = _tokenized(data_parallel_rank=bad)
                controller.dispatch_generate_request(req)
                controller.dispatching.assert_called_once_with(req)
                for worker in controller.workers:
                    worker.send_pyobj.assert_not_called()

    def test_disagg_mode_ignores_pin(self):
        controller = _make_controller(disaggregation_mode="prefill")
        req = _tokenized(data_parallel_rank=1, bootstrap_room=7)
        controller.dispatch_generate_request(req)
        controller.dispatching.assert_called_once_with(req)
        controller.workers[1].send_pyobj.assert_not_called()

    def test_pin_notes_budget_reservation(self):
        controller = _make_controller(dp_size=2)
        loads = [
            GetLoadReqOutput(dp_rank=0, num_reqs=0),
            GetLoadReqOutput(dp_rank=1, num_reqs=0),
        ]
        controller.dp_budget.update_budget(loads)
        controller.dp_budget.update_budget(loads)  # same order -> reservations live
        controller.dispatch_generate_request(_tokenized(data_parallel_rank=1))
        self.assertEqual(controller.dp_budget.reservations[1], 1)
        self.assertEqual(controller.dp_budget.reservations[0], 0)


class TestDPBudgetPinnedDispatch(unittest.TestCase):
    def test_pinned_reservation_steers_budget_away(self):
        budget = DPBudget(LoadBalanceMethod.SHORTEST_QUEUE)
        loads = [
            GetLoadReqOutput(dp_rank=0, num_reqs=1),
            GetLoadReqOutput(dp_rank=1, num_reqs=1),
        ]
        budget.update_budget(loads)
        budget.update_budget(loads)
        self.assertEqual(budget.dispatch(pinned_rank=0), 0)
        # rank 0 now carries a pinned in-flight request; the next budget
        # dispatch must prefer rank 1.
        self.assertEqual(budget.dispatch(), 1)

    def test_pinned_before_first_snapshot_is_noop(self):
        budget = DPBudget(LoadBalanceMethod.SHORTEST_QUEUE)
        # no rank_order yet — must not raise
        self.assertEqual(budget.dispatch(pinned_rank=0), 0)
        self.assertEqual(budget.reservations, {})


class TestTokenizeBridge(unittest.IsolatedAsyncioTestCase):
    async def test_pin_reaches_tokenized_request(self):
        # The only bridge that puts the pin onto the wire is
        # tokenize_one_request copying obj.data_parallel_rank into
        # TokenizedGenerateReqInput.
        processor = _make_input_processor(dp_size=8)
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
        )
        obj.normalize_batch_and_arguments()
        tokenized = await processor.tokenize_one_request(obj)
        self.assertIsInstance(tokenized, TokenizedGenerateReqInput)
        self.assertEqual(tokenized.data_parallel_rank, 6)


class TestSingleWorkerIdRange(unittest.TestCase):
    def test_last_rank_is_valid(self):
        controller = _make_controller(dp_size=4)
        with mock.patch.dict(os.environ, {"SINGLE_WORKER_ID": "3"}):
            controller.single_robin_scheduler(_tokenized())
        controller.workers[3].send_pyobj.assert_called_once()

    def test_out_of_range_still_rejected(self):
        controller = _make_controller(dp_size=4)
        with mock.patch.dict(os.environ, {"SINGLE_WORKER_ID": "4"}):
            with self.assertRaises(ValueError):
                controller.single_robin_scheduler(_tokenized())


if __name__ == "__main__":
    unittest.main()
