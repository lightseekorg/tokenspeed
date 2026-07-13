import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed_kernel import profiling

from tokenspeed.runtime.engine import request_handler as request_handler_mod
from tokenspeed.runtime.engine.io_struct import ProfileReq, ProfileReqType
from tokenspeed.runtime.engine.request_handler import RequestHandler
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
        self.handler = _make_handler(_attn_mapping(tp_rank=3))
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
            self.assertTrue(config.output.endswith("test-profile-TP3.proton"))
            stop_profiling.assert_not_called()

            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
            self.assertTrue(result.success)
            stop_profiling.assert_called_once()
            self.assertFalse(self.handler.profile_in_progress)

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
        # until every TP peer has finalized its proton file.
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
