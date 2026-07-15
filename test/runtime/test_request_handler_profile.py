import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed_kernel import profiling

from tokenspeed.runtime.engine import request_handler as request_handler_mod
from tokenspeed.runtime.engine.io_struct import ProfileReq, ProfileReqType
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_control_client import SchedulerControlClient
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode


def _attn_mapping(
    tp_rank: int = 0, dp_rank: int | None = None, cp_rank: int | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        tp_rank=tp_rank,
        has_dp=dp_rank is not None,
        dp_rank=dp_rank or 0,
        has_cp=cp_rank is not None,
        cp_rank=cp_rank or 0,
    )


def _make_handler(attn_mapping: SimpleNamespace | None = None) -> RequestHandler:
    handler = RequestHandler.__new__(RequestHandler)
    handler.forward_ct = 0
    attn_mapping = attn_mapping or _attn_mapping()
    handler.attn_tp_rank = attn_mapping.tp_rank
    handler.attn_tp_cpu_group = None
    handler.profile_rank_tag = request_handler_mod._profile_rank_tag(attn_mapping)
    handler.init_profiler()
    return handler


def _start_req(output_dir: str, **kwargs) -> ProfileReq:
    return ProfileReq(
        type=ProfileReqType.START_PROFILE,
        output_dir=output_dir,
        activities=["PROTON"],
        profile_id="test-profile",
        **kwargs,
    )


class TestRequestHandlerProtonProfile(unittest.TestCase):
    def setUp(self):
        self.handler = _make_handler()
        self.output_dir = tempfile.mkdtemp()
        profiling.ProfilingState.reset()
        self.addCleanup(profiling.ProfilingState.reset)
        # stop_profile barriers the attn-TP CPU group; there is no real
        # process group in unit tests.
        barrier_patcher = mock.patch.object(
            request_handler_mod.torch.distributed, "barrier"
        )
        self.barrier = barrier_patcher.start()
        self.addCleanup(barrier_patcher.stop)

    def test_init_fails_when_proton_unavailable(self):
        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=False
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        self.assertFalse(result.success)
        self.assertIn("Proton is not available", result.message)
        self.assertFalse(self.handler.profile_in_progress)

    def test_start_profile_preserves_existing_positional_argument_order(self):
        class FakeAsyncLLM:
            def auto_create_handle_loop(self):
                pass

            async def _execute_profile(self, req):
                return req

        req = asyncio.run(
            SchedulerControlClient.start_profile(
                FakeAsyncLLM(),
                "/profiles",
                1,
                2,
                ["CPU"],
                False,
                True,
                True,
                "legacy-profile-id",
            )
        )

        self.assertEqual(req.output_dir, "/profiles")
        self.assertEqual(req.start_step, 1)
        self.assertEqual(req.num_steps, 2)
        self.assertEqual(req.activities, ["CPU"])
        self.assertFalse(req.with_stack)
        self.assertTrue(req.record_shapes)
        self.assertTrue(req.profile_by_stage)
        self.assertEqual(req.profile_id, "legacy-profile-id")

        wire_req = ProfileReq(
            ProfileReqType.START_PROFILE,
            "/wire-profiles",
            3,
            4,
            ["VIZTRACER"],
            True,
            False,
            True,
            "legacy-wire-id",
        )
        self.assertEqual(wire_req.profile_id, "legacy-wire-id")

    def test_profile_defaults_ignore_legacy_environment(self):
        class FakeAsyncLLM:
            def auto_create_handle_loop(self):
                pass

            async def _execute_profile(self, req):
                return req

        with mock.patch.dict(
            os.environ,
            {
                "TOKENSPEED_PROFILER_DIR": "/environment-profile",
                "TOKENSPEED_PROFILE_WITH_STACK": "false",
                "TOKENSPEED_VIZTRACER_MIN_DURATION_US": "999",
                "TOKENSPEED_KERNEL_PROFILE_DATA": "trace",
            },
        ):
            req = asyncio.run(SchedulerControlClient.start_profile(FakeAsyncLLM()))

        self.assertEqual(req.output_dir, "/tmp")
        self.assertTrue(req.with_stack)

        nullable_req = asyncio.run(
            SchedulerControlClient.start_profile(
                FakeAsyncLLM(), output_dir=None, with_stack=None, record_shapes=None
            )
        )
        self.assertEqual(nullable_req.output_dir, "/tmp")
        self.assertTrue(nullable_req.with_stack)
        self.assertFalse(nullable_req.record_shapes)

        wire_req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=None,
            activities=None,
            with_stack=None,
            record_shapes=None,
            profile_id="wire-defaults",
        )
        result = self.handler.init_profile(
            wire_req.output_dir,
            wire_req.start_step,
            wire_req.num_steps,
            wire_req.activities,
            wire_req.with_stack,
            wire_req.record_shapes,
            wire_req.profile_by_stage,
            wire_req.profile_id,
        )
        self.assertTrue(result.success)
        self.assertEqual(self.handler.profiler_output_dir, "/tmp")
        self.assertTrue(self.handler.torch_profiler_with_stack)
        self.assertFalse(self.handler.torch_profiler_record_shapes)

    def test_invalid_profile_request_fails_before_state_mutation(self):
        invalid_requests = [
            (
                ProfileReq(
                    type=ProfileReqType.START_PROFILE,
                    activities=["CPU", "UNKNOWN"],
                    profile_id="invalid-activity",
                ),
                "Unsupported profiling activities",
            ),
            (
                ProfileReq(
                    type=ProfileReqType.START_PROFILE,
                    activities=[],
                    profile_id="empty-activities",
                ),
                "non-empty list",
            ),
            (
                ProfileReq(
                    type=ProfileReqType.START_PROFILE,
                    activities=["CPU", "CPU"],
                    profile_id="duplicate-activities",
                ),
                "must not contain duplicates",
            ),
            (
                ProfileReq(
                    type=ProfileReqType.START_PROFILE,
                    activities=["CPU"],
                    num_steps=0,
                    profile_id="zero-steps",
                ),
                "positive integer",
            ),
            (
                ProfileReq(
                    type=ProfileReqType.START_PROFILE,
                    activities=["CPU"],
                    profile_by_stage=True,
                    profile_id="missing-stage-steps",
                ),
                "requires num_steps",
            ),
            (
                ProfileReq(
                    type=ProfileReqType.START_PROFILE,
                    activities=["CPU"],
                    start_step=1,
                    num_steps=2,
                    profile_by_stage=True,
                    profile_id="stage-with-start-step",
                ),
                "cannot be combined with start_step",
            ),
        ]

        for request, message in invalid_requests:
            with self.subTest(profile_id=request.profile_id):
                result = self.handler.profile(request)
                self.assertFalse(result.success)
                self.assertIn(message, result.message)
                self.assertFalse(self.handler.profile_in_progress)
                self.assertIsNone(self.handler.profiler_activities)

    def test_init_rejects_proton_mixed_with_gpu_profilers(self):
        for conflicting in (["GPU"], ["CUDA_PROFILER"], ["GPU", "CUDA_PROFILER"]):
            req = _start_req(self.output_dir)
            req.activities = ["CPU", "PROTON", *conflicting]

            result = self.handler.profile(req)

            self.assertFalse(result.success)
            for activity in conflicting:
                self.assertIn(activity, result.message)
            self.assertFalse(self.handler.profile_in_progress)

    def test_rejected_request_leaves_profiler_state_untouched(self):
        req = _start_req(self.output_dir, profile_by_stage=True, num_steps=2)
        req.activities = ["PROTON", "GPU"]

        result = self.handler.profile(req)

        self.assertFalse(result.success)
        self.assertFalse(self.handler.profile_by_stage)
        self.handler.forward_ct = 1
        self.handler._profile_batch_predicate(ForwardMode.EXTEND)
        self.assertFalse(self.handler.profile_in_progress)

    def test_init_allows_proton_with_host_side_profilers(self):
        req = _start_req(self.output_dir)
        req.activities = ["CPU", "MEM", "VIZTRACER", "PROTON"]

        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ):
            result = self.handler.init_profile(
                output_dir=self.output_dir,
                start_step=None,
                num_steps=None,
                activities=req.activities,
                with_stack=None,
                record_shapes=None,
                profile_by_stage=False,
                profile_id="test-profile",
            )

        self.assertTrue(result.success)

    def test_init_fails_when_proton_session_active(self):
        state = profiling.ProfilingState.get()
        state.enabled = True
        state._session = 1

        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        self.assertFalse(result.success)
        self.assertIn("already active", result.message)
        self.assertFalse(self.handler.profile_in_progress)

    def test_init_rejects_hip_visible_devices_for_amd_proton(self):
        with (
            mock.patch.object(
                request_handler_mod, "proton_available", return_value=True
            ),
            mock.patch.object(request_handler_mod.torch.version, "hip", "7.2"),
            mock.patch.dict(
                request_handler_mod.os.environ, {"HIP_VISIBLE_DEVICES": "0"}
            ),
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        self.assertFalse(result.success)
        self.assertIn("ROCR_VISIBLE_DEVICES", result.message)
        self.assertFalse(self.handler.profile_in_progress)

    def test_start_returns_failure_when_proton_cannot_initialize(self):
        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(
            request_handler_mod,
            "start_profiling",
            side_effect=RuntimeError("rocprofiler unavailable"),
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        self.assertFalse(result.success)
        self.assertIn("Failed to start Proton profiling", result.message)
        self.assertFalse(self.handler.profile_in_progress)

    def test_start_and_stop_drive_proton_session_per_rank(self):
        self.handler = _make_handler(_attn_mapping(tp_rank=3))
        with mock.patch.dict(
            os.environ,
            {
                "TOKENSPEED_KERNEL_PROFILE_DATA": "trace",
                "TOKENSPEED_KERNEL_PROFILE_BACKEND": "cupti",
                "TOKENSPEED_KERNEL_PROFILE_MODE": "pcsampling",
                "TOKENSPEED_KERNEL_PROFILE_HOOK": "none",
                "TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT": "chrome_trace",
            },
        ), mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(
            request_handler_mod, "start_profiling"
        ) as start_profiling, mock.patch.object(
            request_handler_mod, "stop_profiling"
        ) as stop_profiling:
            result = self.handler.profile(_start_req(self.output_dir))
            self.assertTrue(result.success)
            self.assertTrue(self.handler.profile_in_progress)

            config = start_profiling.call_args[0][0]
            self.assertEqual(config.output.rsplit("/", 1)[0], self.output_dir)
            self.assertTrue(config.output.endswith("test-profile-TP3.proton"))
            self.assertEqual(config.data, "tree")
            self.assertIsNone(config.backend)
            self.assertIsNone(config.mode)
            self.assertEqual(config.hook, "triton")
            self.assertEqual(config.output_format, "")
            stop_profiling.assert_not_called()

            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
            self.assertTrue(result.success)
            stop_profiling.assert_called_once()
            self.assertFalse(self.handler.profile_in_progress)

    def test_viztracer_min_duration_uses_stable_default(self):
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=self.output_dir,
            activities=["VIZTRACER"],
            profile_id="test-profile",
        )

        with mock.patch.dict(
            os.environ, {"TOKENSPEED_VIZTRACER_MIN_DURATION_US": "999"}
        ), mock.patch.object(request_handler_mod, "VizTracer") as viztracer:
            result = self.handler.profile(req)

        self.assertTrue(result.success)
        viztracer.assert_called_once_with(
            output_file=mock.ANY,
            min_duration=100,
            log_async=True,
        )
        viztracer.return_value.start.assert_called_once_with()

    def test_rank_tag_includes_dp_cp_ranks_when_present(self):
        self.assertEqual(
            request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=3)), "TP3"
        )
        self.assertEqual(
            request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=0, dp_rank=1)),
            "DP1-TP0",
        )
        self.assertEqual(
            request_handler_mod._profile_rank_tag(
                _attn_mapping(tp_rank=2, dp_rank=1, cp_rank=0)
            ),
            "DP1-CP0-TP2",
        )

    def test_proton_outputs_do_not_collide_across_dp_ranks(self):
        # Two DP peers share attn_tp_rank=0 but must write distinct files.
        outputs = []
        for dp_rank in (0, 1):
            handler = _make_handler(_attn_mapping(tp_rank=0, dp_rank=dp_rank))
            with mock.patch.object(
                request_handler_mod, "proton_available", return_value=True
            ), mock.patch.object(
                request_handler_mod, "start_profiling"
            ) as start_profiling, mock.patch.object(
                request_handler_mod, "stop_profiling"
            ):
                result = handler.profile(_start_req(self.output_dir))
                self.assertTrue(result.success)
                outputs.append(start_profiling.call_args[0][0].output)
                handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))

        self.assertNotEqual(outputs[0], outputs[1])
        self.assertTrue(outputs[0].endswith("test-profile-DP0-TP0.proton"))
        self.assertTrue(outputs[1].endswith("test-profile-DP1-TP0.proton"))

    def test_stop_profile_barriers_tp_peers_after_proton_finalize(self):
        # Only attn-TP rank 0 replies to /stop_profile; the reply must wait
        # until every TP peer has finalized its Proton file.
        self.handler.attn_tp_cpu_group = object()

        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(request_handler_mod, "start_profiling"), mock.patch.object(
            request_handler_mod, "stop_profiling"
        ) as stop_profiling:
            self.barrier.side_effect = lambda group: self.assertTrue(
                stop_profiling.called
            )
            self.handler.profile(_start_req(self.output_dir))
            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))

        self.assertTrue(result.success)
        self.barrier.assert_called_once_with(self.handler.attn_tp_cpu_group)

    def test_stop_profile_reports_proton_finalize_failure(self):
        self.handler.attn_tp_cpu_group = object()

        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(request_handler_mod, "start_profiling"), mock.patch.object(
            request_handler_mod,
            "stop_profiling",
            side_effect=RuntimeError("finalize failed"),
        ):
            self.handler.profile(_start_req(self.output_dir))
            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))

        self.assertFalse(result.success)
        self.assertIn("Failed to finalize Proton profiling", result.message)
        self.assertFalse(self.handler.profile_in_progress)
        self.barrier.assert_called_once_with(self.handler.attn_tp_cpu_group)

    def test_num_steps_window_finalizes_proton(self):
        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(request_handler_mod, "start_profiling"), mock.patch.object(
            request_handler_mod, "stop_profiling"
        ) as stop_profiling:
            result = self.handler.profile(_start_req(self.output_dir, num_steps=2))
            self.assertTrue(result.success)
            self.assertTrue(self.handler.profile_in_progress)

            self.handler.forward_ct = 1
            self.handler._profile_batch_predicate()
            stop_profiling.assert_not_called()

            self.handler.forward_ct = 2
            self.handler._profile_batch_predicate()
            stop_profiling.assert_called_once()
            self.assertFalse(self.handler.profile_in_progress)


if __name__ == "__main__":
    unittest.main()
