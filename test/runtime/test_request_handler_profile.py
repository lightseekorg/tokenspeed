import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import tokenspeed_kernel.profiling as profiling
import torch

from tokenspeed.runtime.engine import request_handler as request_handler_mod
from tokenspeed.runtime.engine.io_struct import ProfileReq, ProfileReqType
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode


def _attn_mapping(tp_rank: int = 0, dp_rank: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        tp_rank=tp_rank,
        has_dp=dp_rank is not None,
        dp_rank=dp_rank or 0,
    )


def _make_handler(attn_mapping: SimpleNamespace | None = None) -> RequestHandler:
    handler = RequestHandler.__new__(RequestHandler)
    handler.forward_ct = 0
    # EXPERT_LOAD is refused under --enable-eplb; plain serving here.
    handler.server_args = SimpleNamespace(enable_eplb=False)
    attn_mapping = attn_mapping or _attn_mapping()
    handler.attn_tp_rank = attn_mapping.tp_rank
    handler.attn_tp_cpu_group = None
    # __init__ is bypassed here, so mirror the state _profile_sync needs: the
    # peer count it short-circuits on and the preallocated rendezvous buffer.
    handler.attn_tp_size = getattr(attn_mapping, "tp_size", 2)
    handler._profile_sync_buf = torch.zeros(1, dtype=torch.int32, device="cpu")
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
        # stop_profile rendezvouses the attn-TP CPU group through
        # _profile_sync, which all-reduces a preallocated CPU tensor rather
        # than calling barrier (barrier would allocate on the bound CUDA
        # device and trip the control-plane guard). No real process group
        # exists in unit tests.
        sync_patcher = mock.patch.object(
            request_handler_mod.torch.distributed, "all_reduce"
        )
        self.profile_sync = sync_patcher.start()
        self.addCleanup(sync_patcher.stop)

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

    def test_rank_tag_includes_dp_rank_when_present(self):
        self.assertEqual(
            request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=3)), "TP3"
        )
        self.assertEqual(
            request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=0, dp_rank=1)),
            "DP1-TP0",
        )
        self.assertEqual(
            request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=2, dp_rank=1)),
            "DP1-TP2",
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

    def test_stop_profile_syncs_tp_peers_after_proton_finalize(self):
        # Only attn-TP rank 0 replies to /stop_profile; the reply must wait
        # until every TP peer has finalized its Proton file.
        self.handler.attn_tp_cpu_group = object()

        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(request_handler_mod, "start_profiling"), mock.patch.object(
            request_handler_mod, "stop_profiling"
        ) as stop_profiling:
            self.profile_sync.side_effect = lambda tensor, group: self.assertTrue(
                stop_profiling.called
            )
            self.handler.profile(_start_req(self.output_dir))
            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))

        self.assertTrue(result.success)
        self.profile_sync.assert_called_once_with(
            self.handler._profile_sync_buf, group=self.handler.attn_tp_cpu_group
        )

    def test_profile_sync_never_touches_the_device(self):
        """The rendezvous must not allocate, and must not be a barrier.

        stop_profile runs on the control-plane thread, where _NoDeviceWork
        rejects CUDA factories. torch.distributed.barrier prefers
        group.bound_device_id over its CPU branch, and every group is bound to
        cuda:N for eager NCCL init, so a barrier here allocated on CUDA and
        killed all ranks mid-profile. Pin both halves: no barrier, and a CPU
        rendezvous tensor.
        """
        self.handler.attn_tp_cpu_group = object()

        with mock.patch.object(
            request_handler_mod.torch.distributed, "barrier"
        ) as barrier:
            self.handler._profile_sync()

        barrier.assert_not_called()
        self.profile_sync.assert_called_once()
        tensor = self.profile_sync.call_args.args[0]
        self.assertEqual(tensor.device.type, "cpu")

    def test_profile_sync_is_a_noop_without_peers(self):
        self.handler.attn_tp_size = 1
        self.handler._profile_sync()
        self.profile_sync.assert_not_called()

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
        self.profile_sync.assert_called_once_with(
            self.handler._profile_sync_buf, group=self.handler.attn_tp_cpu_group
        )

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


class TestRequestHandlerExpertLoadProfile(unittest.TestCase):
    """EXPERT_LOAD zeroes the routing load counters at start and dumps at stop."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.device = mock.Mock()
        self.device.dump_expert_load.return_value = {
            "physical_count": torch.tensor([[3, 1], [2, 2]]),
            "ep_rank": 0,
        }
        self.handler = _make_handler(_attn_mapping(tp_rank=2))
        self.handler._device = self.device
        self.handler.attn_tp_size = 1
        recording = mock.patch.object(
            request_handler_mod, "expert_load_recording_enabled", return_value=True
        )
        recording.start()
        self.addCleanup(recording.stop)

    def _start(self, **kwargs) -> ProfileReq:
        return ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=self.output_dir,
            activities=["EXPERT_LOAD"],
            profile_id="load",
            **kwargs,
        )

    def test_start_resets_and_stop_dumps_per_rank_record(self):
        result = self.handler.profile(self._start())
        self.assertTrue(result.success)
        self.device.reset_expert_load.assert_called_once_with()
        self.device.dump_expert_load.assert_not_called()

        result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
        self.assertTrue(result.success)
        self.device.dump_expert_load.assert_called_once_with(
            f"{self.output_dir}/load-TP2.expert-load.pt"
        )
        self.assertFalse(self.handler.profile_in_progress)

    def test_stage_profiles_dump_one_record_per_stage(self):
        result = self.handler.profile(self._start(profile_by_stage=True, num_steps=1))
        self.assertTrue(result.success)
        self.handler._profile_batch_predicate(ForwardMode.EXTEND)
        self.handler._profile_batch_predicate(ForwardMode.EXTEND)
        self.handler._profile_batch_predicate(ForwardMode.DECODE)
        self.handler._profile_batch_predicate(ForwardMode.DECODE)
        dumped = [call.args[0] for call in self.device.dump_expert_load.call_args_list]
        self.assertEqual(
            dumped,
            [
                f"{self.output_dir}/load-TP2-EXTEND.expert-load.pt",
                f"{self.output_dir}/load-TP2-DECODE.expert-load.pt",
            ],
        )
        self.assertEqual(self.device.reset_expert_load.call_count, 2)

    def test_init_refuses_without_recording_or_device(self):
        with mock.patch.object(
            request_handler_mod, "expert_load_recording_enabled", return_value=False
        ):
            result = self.handler.profile(self._start())
        self.assertFalse(result.success)
        self.assertIn("--expert-distribution-recorder-mode stat", result.message)
        self.assertFalse(self.handler.profile_in_progress)
        self.device.reset_expert_load.assert_not_called()

        self.handler._device = None
        result = self.handler.profile(self._start())
        self.assertFalse(result.success)
        self.assertIn("device handle", result.message)

    def test_init_refuses_under_online_rebalancing(self):
        # The rebalance snapshots and zeroes the same counters, so a profile
        # window would be cut at every snapshot.
        self.handler.server_args = SimpleNamespace(enable_eplb=True)
        result = self.handler.profile(self._start())
        self.assertFalse(result.success)
        self.assertIn("--enable-eplb", result.message)
        self.assertFalse(self.handler.profile_in_progress)
        self.device.reset_expert_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
