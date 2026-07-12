import tempfile
import unittest
from unittest import mock

from tokenspeed_kernel import profiling

from tokenspeed.runtime.engine import request_handler as request_handler_mod
from tokenspeed.runtime.engine.io_struct import ProfileReq, ProfileReqType
from tokenspeed.runtime.engine.request_handler import RequestHandler


def _make_handler() -> RequestHandler:
    handler = RequestHandler.__new__(RequestHandler)
    handler.forward_ct = 0
    handler.attn_tp_rank = 0
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

    def test_init_fails_when_proton_unavailable(self):
        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=False
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        self.assertFalse(result.success)
        self.assertIn("Proton is not available", result.message)
        self.assertFalse(self.handler.profile_in_progress)

    def test_init_rejects_proton_mixed_with_gpu_profilers(self):
        for conflicting in (["GPU"], ["CUDA_PROFILER"], ["GPU", "CUDA_PROFILER"]):
            req = _start_req(self.output_dir)
            req.activities = ["CPU", "PROTON", *conflicting]

            result = self.handler.profile(req)

            self.assertFalse(result.success)
            for activity in conflicting:
                self.assertIn(activity, result.message)
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

    def test_init_fails_when_env_bootstrap_session_active(self):
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

    def test_start_and_stop_drive_proton_session_per_rank(self):
        self.handler.attn_tp_rank = 3
        with mock.patch.object(
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
            self.assertTrue(config.output.endswith("test-profile-TP-3.proton"))
            stop_profiling.assert_not_called()

            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
            self.assertTrue(result.success)
            stop_profiling.assert_called_once()
            self.assertFalse(self.handler.profile_in_progress)

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
