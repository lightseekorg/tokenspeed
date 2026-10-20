import gzip
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import tokenspeed_kernel.profiling as profiling
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import current_platform

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


_CANARY_VISIBILITY_NAMES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def _canary_child_environment(parent_environment, *, expected_handler):
    # Preserve the hardware grant. Raising guards keep the child CPU-only.
    environment = dict(parent_environment)
    environment["TOKENSPEED_CPU_PROFILE_CANARY_CHILD"] = "1"
    environment["TOKENSPEED_CPU_PROFILE_EXPECTED_HANDLER"] = expected_handler
    return environment


class TestCanaryChildVisibility:
    @pytest.mark.parametrize(
        "selectors",
        [
            {},
            {"ROCR_VISIBLE_DEVICES": "GPU-test"},
            {"ROCR_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""},
            {
                "ROCR_VISIBLE_DEVICES": "GPU-test",
                "HIP_VISIBLE_DEVICES": "3,1",
                "CUDA_VISIBLE_DEVICES": "7",
                "GPU_DEVICE_ORDINAL": "2",
            },
        ],
    )
    def test_fresh_child_preserves_absent_and_conflicting_parent_selectors(
        self, selectors
    ):
        # This subprocess never imports Torch or queries hardware.
        probe = textwrap.dedent("""
            from datetime import datetime, timezone
            import json, os, sys
            names = ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                     'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
            audit = {'pid': os.getpid(),
                     'started_utc': datetime.now(timezone.utc).isoformat(),
                     'visibility': {name: os.environ.get(name) for name in names}}
            expected = json.loads(sys.argv[1])
            assert set(expected) == set(names)
            assert audit['visibility'] == expected
            audit['ended_utc'] = datetime.now(timezone.utc).isoformat()
            print(json.dumps({'visibility_only_child': audit}), flush=True)
        """)
        parent = {
            key: value
            for key, value in os.environ.items()
            if key not in _CANARY_VISIBILITY_NAMES
        }
        parent.update(selectors)
        environment = _canary_child_environment(
            parent, expected_handler="visibility-only probe"
        )
        assert environment == {
            **parent,
            "TOKENSPEED_CPU_PROFILE_CANARY_CHILD": "1",
            "TOKENSPEED_CPU_PROFILE_EXPECTED_HANDLER": "visibility-only probe",
        }
        expected = {name: parent.get(name) for name in _CANARY_VISIBILITY_NAMES}
        child = subprocess.run(
            [sys.executable, "-c", probe, json.dumps(expected)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert (
            json.loads(child.stdout)["visibility_only_child"]["visibility"] == expected
        )


class TestRequestHandlerProtonProfile:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self.handler = _make_handler()
        self.output_dir = tempfile.mkdtemp()
        profiling.ProfilingState.reset()
        request.addfinalizer(profiling.ProfilingState.reset)
        # stop_profile rendezvouses the attn-TP CPU group through
        # _profile_sync, which all-reduces a preallocated CPU tensor rather
        # than calling barrier (barrier would allocate on the bound CUDA
        # device and trip the control-plane guard). No real process group
        # exists in unit tests.
        sync_patcher = mock.patch.object(
            request_handler_mod.torch.distributed, "all_reduce"
        )
        self.profile_sync = sync_patcher.start()
        request.addfinalizer(sync_patcher.stop)

    def test_init_fails_when_proton_unavailable(self):
        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=False
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        assert not (result.success)
        assert ("Proton is not available") in (result.message)
        assert not (self.handler.profile_in_progress)

    def test_cpu_profile_captures_a_worker_started_before_the_session(self):
        if os.environ.get("TOKENSPEED_CPU_PROFILE_CANARY_CHILD") != "1":
            # A legitimate hardware test may have initialized CUDA earlier in
            # this suite. Prove the CPU-only canary in its own fresh process,
            # without resetting or hiding that parent's state.
            bootstrap = textwrap.dedent("""
                from datetime import datetime, timezone
                import json, os, sys
                import pytest
                from pathlib import Path
                from types import SimpleNamespace
                from unittest import mock

                names = ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                         'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
                audit = {'pid': os.getpid(),
                         'started_utc': datetime.now(timezone.utc).isoformat(),
                         'visibility': {name: os.environ.get(name) for name in names},
                         'blocked_initializations': []}
                print(json.dumps({'canary_start': audit}), flush=True)
                expected = json.loads(sys.argv[2])
                assert set(expected) == set(names)
                assert audit['visibility'] == expected
                import torch
                audit['cuda_initialized_before'] = torch.cuda.is_initialized()
                assert not audit['cuda_initialized_before']
                def deny(api):
                    def denied(*args, **kwargs):
                        audit['blocked_initializations'].append(api)
                        raise RuntimeError(f'CPU-only initialization guard: {api}')
                    return denied
                status = 1
                try:
                    with (
                        mock.patch.object(torch.cuda, '_lazy_init', side_effect=deny('torch.cuda._lazy_init')),
                        mock.patch.object(torch._C, '_cuda_init', side_effect=deny('torch._C._cuda_init')),
                        mock.patch.object(torch.cuda, 'is_available', return_value=False),
                        mock.patch.object(torch.cuda, 'device_count', return_value=0),
                        mock.patch.object(torch.cuda, 'current_device', side_effect=AssertionError('hardware discovery forbidden')),
                        mock.patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('hardware discovery forbidden')),
                    ):
                        from tokenspeed_triton.backends.compiler import GPUTarget
                        from tokenspeed_triton.runtime.driver import DriverConfig
                        from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo
                        # Keep the parent's vendor classification so importing the
                        # real handler does not require another vendor's packages.
                        vendor, major, minor, backend, arch, warp_size = json.loads(sys.argv[3])
                        Platform.override(PlatformInfo(
                            vendor=vendor, arch_version=ArchVersion(major, minor),
                            device_name='CPU test mock', device_count=0,
                            total_memory=0, memory_bandwidth=0, sm_count=0,
                            max_threads_per_sm=0, max_shared_memory_per_sm=0))
                        target = GPUTarget(backend, arch, warp_size)
                        discovery = SimpleNamespace(get_current_target=lambda: target)
                        with mock.patch.object(DriverConfig, 'active', new_callable=mock.PropertyMock, return_value=discovery):
                            from tokenspeed.runtime.engine import request_handler as handler_module
                            audit['real_request_handler_import'] = str(Path(handler_module.__file__).resolve())
                            assert audit['real_request_handler_import'] == os.environ['TOKENSPEED_CPU_PROFILE_EXPECTED_HANDLER']
                            status = int(pytest.main([
                                '-q', '-p', 'no:cacheprovider',
                                sys.argv[1] + '::TestRequestHandlerProtonProfile::test_cpu_profile_captures_a_worker_started_before_the_session',
                            ]))
                        assert not audit['blocked_initializations'], 'Blocked initialization is a failed canary'
                        assert not torch.cuda.is_initialized()
                finally:
                    audit['cuda_initialized_after'] = torch.cuda.is_initialized()
                    audit['ended_utc'] = datetime.now(timezone.utc).isoformat()
                    audit['exit_code'] = status
                    print(json.dumps({'canary_end': audit}), flush=True)
                raise SystemExit(status)
            """)
            environment = _canary_child_environment(
                os.environ,
                expected_handler=str(Path(request_handler_mod.__file__).resolve()),
            )
            platform = current_platform()
            target = triton.runtime.driver.active.get_current_target()
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    str(Path(__file__).resolve()),
                    json.dumps(
                        {
                            name: environment.get(name)
                            for name in _CANARY_VISIBILITY_NAMES
                        }
                    ),
                    json.dumps(
                        [
                            platform.vendor,
                            platform.arch_version.major,
                            platform.arch_version.minor,
                            target.backend,
                            target.arch,
                            target.warp_size,
                        ]
                    ),
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            print(child.stdout, end="")
            assert child.returncode == 0, child.stdout + child.stderr
            return
        assert not torch.cuda.is_initialized()
        self.handler.attn_tp_size = 1

        def work():
            # The TLS C++ enabled query is false here under all-thread capture.
            enabled = torch.autograd.profiler._is_profiler_enabled
            with torch.profiler.record_function("existing_forward_thread_canary"):
                torch.ones(4, device="cpu").add_(2)
            return threading.get_native_id(), enabled

        req = _start_req(self.output_dir, record_shapes=False, with_stack=False)
        req.activities = ["CPU"]
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with ThreadPoolExecutor(max_workers=1) as worker:
                before = worker.submit(threading.get_native_id).result(timeout=10)
                assert self.handler.profile(req).success
                thread, enabled = worker.submit(work).result(timeout=10)
                assert thread == before
                assert enabled
                assert self.handler.stop_profile().success
                assert (
                    worker.submit(threading.get_native_id).result(timeout=10) == before
                )
        trace = Path(self.output_dir) / "test-profile-TP0.trace.json.gz"
        with gzip.open(trace, "rt") as source:
            events = json.load(source)["traceEvents"]
        ranges = [
            event
            for event in events
            if event.get("name") == "existing_forward_thread_canary"
        ]
        assert len(ranges) == 1
        phase = ranges[0]
        assert phase["tid"] == before
        nested = [
            event
            for event in events
            if event.get("name") == "aten::ones"
            and event.get("pid") == phase["pid"]
            and event.get("tid") == phase["tid"]
            and phase["ts"] <= event["ts"]
            and event["ts"] + event.get("dur", 0) <= phase["ts"] + phase["dur"]
        ]
        assert nested
        assert not torch.autograd.profiler._is_profiler_enabled
        assert not torch.cuda.is_initialized()

    def test_missing_global_profiler_state_refuses_capture_explicitly(self):
        req = _start_req(self.output_dir)
        req.activities = ["CPU"]
        with mock.patch.dict(torch.autograd.profiler.__dict__):
            del torch.autograd.profiler._is_profiler_enabled
            result = self.handler.profile(req)
        assert not result.success
        assert "missing process-global profiler state" in result.message
        assert self.handler.torch_profiler is None
        assert not self.handler.profile_in_progress

    def test_missing_all_thread_option_refuses_capture_explicitly(self):
        req = _start_req(self.output_dir)
        req.activities = ["CPU"]
        with mock.patch.object(
            torch.profiler,
            "_ExperimentalConfig",
            side_effect=TypeError("unsupported option"),
        ):
            result = self.handler.profile(req)
        assert not result.success
        assert "all-thread phase profiling" in result.message
        assert self.handler.torch_profiler is None
        assert not self.handler.profile_in_progress

    @pytest.mark.parametrize(
        "schedule", [{"start_step": 10}, {"profile_by_stage": True}]
    )
    @pytest.mark.parametrize("missing", ["global_state", "all_thread_option"])
    def test_unsupported_delayed_or_staged_capture_leaves_clean_retry(
        self, schedule, missing
    ):
        fields = (
            "profile_by_stage",
            "profiler_start_forward_ct",
            "profiler_target_forward_ct",
            "profiler_target_prefill_ct",
            "profiler_target_decode_ct",
            "profiler_prefill_ct",
            "profiler_decode_ct",
            "profiler_activities",
            "profiler_output_dir",
            "profile_id",
            "profile_in_progress",
            "torch_profiler",
        )
        self.handler = _make_handler()
        before = {name: getattr(self.handler, name) for name in fields}
        req = _start_req(self.output_dir, num_steps=2, **schedule)
        req.activities = ["CPU"]
        if missing == "global_state":
            unsupported = mock.patch.object(
                torch.autograd.profiler, "_is_profiler_enabled", None
            )
        else:
            unsupported = mock.patch.object(
                torch.profiler,
                "_ExperimentalConfig",
                side_effect=TypeError("unsupported option"),
            )
        with unsupported:
            result = self.handler.profile(req)
        assert not result.success
        assert "all-thread phase profiling" in result.message
        assert {name: getattr(self.handler, name) for name in fields} == before
        self.handler._profile_batch_predicate(ForwardMode.EXTEND)
        self.handler._profile_batch_predicate(ForwardMode.DECODE)
        assert self.handler.torch_profiler is None
        retry = _start_req(self.output_dir)
        retry.activities = ["CPU"]
        assert self.handler.profile(retry).success
        assert self.handler.profile_in_progress
        assert self.handler.profile(
            ProfileReq(type=ProfileReqType.STOP_PROFILE)
        ).success

    def test_init_rejects_proton_mixed_with_gpu_profilers(self):
        for conflicting in (["GPU"], ["CUDA_PROFILER"], ["GPU", "CUDA_PROFILER"]):
            req = _start_req(self.output_dir)
            req.activities = ["CPU", "PROTON", *conflicting]

            result = self.handler.profile(req)

            assert not (result.success)
            for activity in conflicting:
                assert (activity) in (result.message)
            assert not (self.handler.profile_in_progress)

    def test_rejected_request_leaves_profiler_state_untouched(self):
        req = _start_req(self.output_dir, profile_by_stage=True, num_steps=2)
        req.activities = ["PROTON", "GPU"]

        result = self.handler.profile(req)

        assert not (result.success)
        assert not (self.handler.profile_by_stage)
        self.handler.forward_ct = 1
        self.handler._profile_batch_predicate(ForwardMode.EXTEND)
        assert not (self.handler.profile_in_progress)

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

        assert result.success

    def test_init_fails_when_env_bootstrap_session_active(self):
        state = profiling.ProfilingState.get()
        state.enabled = True
        state._session = 1

        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        assert not (result.success)
        assert ("already active") in (result.message)
        assert not (self.handler.profile_in_progress)

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

        assert not (result.success)
        assert ("ROCR_VISIBLE_DEVICES") in (result.message)
        assert not (self.handler.profile_in_progress)

    def test_start_returns_failure_when_proton_cannot_initialize(self):
        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(
            request_handler_mod,
            "start_profiling",
            side_effect=RuntimeError("rocprofiler unavailable"),
        ):
            result = self.handler.profile(_start_req(self.output_dir))

        assert not (result.success)
        assert ("Failed to start Proton profiling") in (result.message)
        assert not (self.handler.profile_in_progress)

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
            assert result.success
            assert self.handler.profile_in_progress

            config = start_profiling.call_args[0][0]
            assert (config.output.rsplit("/", 1)[0]) == (self.output_dir)
            assert config.output.endswith("test-profile-TP3.proton")
            stop_profiling.assert_not_called()

            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
            assert result.success
            stop_profiling.assert_called_once()
            assert not (self.handler.profile_in_progress)

    def test_rank_tag_includes_dp_rank_when_present(self):
        assert (request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=3))) == (
            "TP3"
        )
        assert (
            request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=0, dp_rank=1))
        ) == ("DP1-TP0")
        assert (
            request_handler_mod._profile_rank_tag(_attn_mapping(tp_rank=2, dp_rank=1))
        ) == ("DP1-TP2")

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
                assert result.success
                outputs.append(start_profiling.call_args[0][0].output)
                handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))

        assert (outputs[0]) != (outputs[1])
        assert outputs[0].endswith("test-profile-DP0-TP0.proton")
        assert outputs[1].endswith("test-profile-DP1-TP0.proton")

    def test_stop_profile_syncs_tp_peers_after_proton_finalize(self):
        # Only attn-TP rank 0 replies to /stop_profile; the reply must wait
        # until every TP peer has finalized its Proton file.
        self.handler.attn_tp_cpu_group = object()

        with mock.patch.object(
            request_handler_mod, "proton_available", return_value=True
        ), mock.patch.object(request_handler_mod, "start_profiling"), mock.patch.object(
            request_handler_mod, "stop_profiling"
        ) as stop_profiling:

            def sync_after_finalize(tensor, group):
                assert stop_profiling.called

            self.profile_sync.side_effect = sync_after_finalize
            self.handler.profile(_start_req(self.output_dir))
            result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))

        assert result.success
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
        assert (tensor.device.type) == ("cpu")

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

        assert not (result.success)
        assert ("Failed to finalize Proton profiling") in (result.message)
        assert not (self.handler.profile_in_progress)
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
            assert result.success
            assert self.handler.profile_in_progress

            self.handler.forward_ct = 1
            self.handler._profile_batch_predicate()
            stop_profiling.assert_not_called()

            self.handler.forward_ct = 2
            self.handler._profile_batch_predicate()
            stop_profiling.assert_called_once()
            assert not (self.handler.profile_in_progress)


class TestRequestHandlerExpertLoadProfile:
    """EXPERT_LOAD zeroes the routing load counters at start and dumps at stop."""

    @pytest.fixture(autouse=True)
    def _setup(self, request):
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
        request.addfinalizer(recording.stop)

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
        assert result.success
        self.device.reset_expert_load.assert_called_once_with()
        self.device.dump_expert_load.assert_not_called()

        result = self.handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
        assert result.success
        self.device.dump_expert_load.assert_called_once_with(
            f"{self.output_dir}/load-TP2.expert-load.pt"
        )
        assert not (self.handler.profile_in_progress)

    def test_stage_profiles_dump_one_record_per_stage(self):
        result = self.handler.profile(self._start(profile_by_stage=True, num_steps=1))
        assert result.success
        self.handler._profile_batch_predicate(ForwardMode.EXTEND)
        self.handler._profile_batch_predicate(ForwardMode.EXTEND)
        self.handler._profile_batch_predicate(ForwardMode.DECODE)
        self.handler._profile_batch_predicate(ForwardMode.DECODE)
        dumped = [call.args[0] for call in self.device.dump_expert_load.call_args_list]
        assert (dumped) == (
            [
                f"{self.output_dir}/load-TP2-EXTEND.expert-load.pt",
                f"{self.output_dir}/load-TP2-DECODE.expert-load.pt",
            ]
        )
        assert (self.device.reset_expert_load.call_count) == (2)

    def test_init_refuses_without_recording_or_device(self):
        with mock.patch.object(
            request_handler_mod, "expert_load_recording_enabled", return_value=False
        ):
            result = self.handler.profile(self._start())
        assert not (result.success)
        assert ("--expert-distribution-recorder-mode stat") in (result.message)
        assert not (self.handler.profile_in_progress)
        self.device.reset_expert_load.assert_not_called()

        self.handler._device = None
        result = self.handler.profile(self._start())
        assert not (result.success)
        assert ("device handle") in (result.message)

    def test_init_refuses_under_online_rebalancing(self):
        # The rebalance snapshots and zeroes the same counters, so a profile
        # window would be cut at every snapshot.
        self.handler.server_args = SimpleNamespace(enable_eplb=True)
        result = self.handler.profile(self._start())
        assert not (result.success)
        assert ("--enable-eplb") in (result.message)
        assert not (self.handler.profile_in_progress)
        self.device.reset_expert_load.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
