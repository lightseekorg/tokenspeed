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

import faulthandler
import signal
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import psutil
import setproctitle
import torch
import torch.distributed as dist
import zmq
from tokenspeed_scheduler import PD, Cache, ExecutionEvent, ForwardEvent, Scheduler

from tokenspeed.runtime.cache.l2.executor import L2CacheExecutor
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser
from tokenspeed.runtime.engine.io_struct import IpcReceiver, IpcSender
from tokenspeed.runtime.engine.load_snapshot import (
    LoadSnapshotPublisher,
    NullLoadSnapshotPublisher,
)
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    advance_forward,
    aligned_max_scheduled_tokens,
    cache_event_from_payload,
    cache_event_key,
    cache_event_to_payload,
    cache_sync_debug_enabled,
    log_gpu_memory_summary,
    make_config,
    pool_to_cache_groups,
    pop_common_cache_event_payloads,
    resolve_dspark_prefix_replay_tokens,
    scheduler_cache_geometry_from_pool,
    should_use_overlap_schedule,
)
from tokenspeed.runtime.execution.distributed_initializer import (
    DistributedConfig,
    DistributedInitializer,
)
from tokenspeed.runtime.execution.factory import (
    ModelExecutorConfig,
    create_model_executor,
    create_model_runner,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import ModelExecutionResult
from tokenspeed.runtime.grammar.capturable_grammar import GrammarStepInputs
from tokenspeed.runtime.layers.attention.registry import create_attn_components
from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.factory import (
    create_kv_transfer,
    get_kv_args,
)
from tokenspeed.runtime.pd.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    NullEventPublisher,
    drain_scheduler_kv_events,
    scheduler_kv_events_to_wire_events,
)
from tokenspeed.runtime.pd.mooncake.entities import KVManagerArgs
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
from tokenspeed.runtime.pd.topology import PDParallelTopology
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import (
    configure_logger,
    get_colorful_logger,
    get_zmq_socket,
)
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.process import register_usr_signal
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


# Sleep between iterations while frozen (PAUSED_ALL) so the keep-mode pause does
# not busy-spin a CPU core waiting for /resume.
_PAUSED_IDLE_SLEEP_S = 0.001


def _forward_op_executes_model_forward(forward_op, *, is_disagg_decode: bool) -> bool:
    """Return whether ``forward_op`` will enter the model forward path.

    On decode-side PD, EXTEND ops only start remote KV receive; the model
    forward runs after the remote prefill completes and the scheduler advances
    the request into decode. Treating those EXTEND ops as model work makes
    idle DP ranks enter dummy collectives that the active rank will not match.
    """
    if forward_op is None:
        return False
    if sum(forward_op.input_lengths) <= 0:
        return False
    if (
        is_disagg_decode
        and forward_op.num_extends() > 0
        and not forward_op.is_local_prefill()
    ):
        return False
    return True


class _NullSender:
    """No-op ZMQ sender for non-rank-0 workers."""

    @staticmethod
    def send_pyobj(x):
        return None


# SMG decodes the ready response's dtype into a fixed enum of these strings, so
# map tokenspeed's dtype onto the nearest one.
_WIRE_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "half": "float16",
    "fp16": "float16",
    "float32": "float32",
    "float": "float32",
    "fp32": "float32",
}


def _wire_dtype(dtype) -> str:
    key = str(dtype).lower().replace("torch.", "")
    mapped = _WIRE_DTYPE_MAP.get(key)
    if mapped is None:
        # Fail at handshake time: misreporting the dtype to the frontend is
        # worse than refusing to start.
        raise ValueError(
            f"dtype {dtype!r} has no SMG wire mapping; extend _WIRE_DTYPE_MAP"
        )
    return mapped


def _tokenspeed_version() -> str:
    try:
        from tokenspeed.version import __version__

        return __version__
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class DpForwardMetadata:
    global_num_tokens: list[int]
    global_batch_size: list[int]
    global_forward_mode: list[int]
    all_decode_or_idle: bool
    all_extend: bool
    need_idle_forward: bool


class EventLoop:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        attn_tp_rank: int,
        dp_rank: int,
        global_rank: int,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        # Do not pass server_args further down the stack after this point.

        self.server_args = server_args
        pd_topology = None
        if server_args.disaggregation_mode != "null":
            pd_topology = PDParallelTopology.from_mapping(server_args.mapping)
            pd_topology.require_cache_pd_supported()
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.global_rank = global_rank
        self.shutdown_event = shutdown_event or threading.Event()

        self.model_config = self._load_model_config(server_args.model)
        if server_args.speculative_draft_model_path is not None:
            draft_model_config = self._load_model_config(
                server_args.speculative_draft_model_path,
                is_draft_worker=True,
            )
        else:
            draft_model_config = None

        prefix_replay_tokens = resolve_dspark_prefix_replay_tokens(
            speculative_algorithm=server_args.speculative_algorithm,
            enable_prefix_caching=server_args.enable_prefix_caching,
            enable_kvstore=server_args.enable_kvstore,
            disaggregation_mode=server_args.disaggregation_mode,
            draft_model_path_use_base=server_args.draft_model_path_use_base,
            draft_model_config=draft_model_config,
        )

        min_per_gpu_mem = self._init_distributed()

        target, draft = create_model_runner(
            server_args, self.model_config, draft_model_config, gpu_id, global_rank
        )
        self.multimodal_encoder_dtype = target.multimodal_encoder_dtype
        if server_args.disaggregation_mode in ("null", "prefill"):
            # Keep this after all target/draft weights are loaded and before
            # create_attn_components profiles memory for the KV-cache budget.
            target.prepare_multimodal_runtime()
        self.use_overlap_schedule = should_use_overlap_schedule(
            disable_overlap_schedule=server_args.disable_overlap_schedule,
            disaggregation_mode=server_args.disaggregation_mode,
        )
        self.overlap_schedule_depth = int(self.use_overlap_schedule)
        decode_input_tokens = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm is not None
            else 1
        )

        (
            attn_backend,
            token_to_kv_pool,
            draft_attn_backend,
            draft_token_to_kv_pool,
            self.cache_storage,
        ) = create_attn_components(
            server_args,
            self.model_config,
            gpu_id,
            global_rank,
            min_per_gpu_mem,
            server_args.enable_memory_saver,
            draft_model_config,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=self.overlap_schedule_depth,
        )

        self._scheduler_cache_geometry = scheduler_cache_geometry_from_pool(
            token_to_kv_pool
        )
        geometry = self._scheduler_cache_geometry
        # The contract is the one source of admitted capacity.
        self.max_total_num_tokens = geometry.token_capacity
        num_total_pages = geometry.num_device_pages
        cache_groups = pool_to_cache_groups(token_to_kv_pool)
        # Resolve the scheduler limit before ModelExecutorConfig sizes input
        # buffers. Lowering the limit is safe; a configured chunk smaller than
        # one state checkpoint block is rejected by aligned_max_scheduled_tokens instead of
        # silently increasing a frozen buffer limit.
        max_scheduled_tokens = server_args.chunked_prefill_size
        if server_args.enable_prefix_caching:
            max_scheduled_tokens = aligned_max_scheduled_tokens(
                server_args.chunked_prefill_size,
                cache_groups,
            )
            if max_scheduled_tokens != server_args.chunked_prefill_size:
                logger.warning(
                    "chunked_prefill_size=%s is not a multiple of the "
                    "state-snapshot checkpoint grain; using %s so recurrent-state "
                    "pages can register for prefix-cache reuse.",
                    server_args.chunked_prefill_size,
                    max_scheduled_tokens,
                )
                server_args.chunked_prefill_size = max_scheduled_tokens
        mapping = server_args.mapping
        # The C++ scheduler's req_pool_idx range is rank-local and 1-based:
        # real rows are 1..max_batch_size, row 0 is reserved, and CUDA graph
        # padding needs one non-real sink row after the scheduler-owned range.
        per_rank_max_batch = server_args.max_num_seqs // max(mapping.attn.dp_size, 1)
        req_pool_padding_index = per_rank_max_batch + 1

        model_executor_config = ModelExecutorConfig.from_server_args(
            server_args=server_args,
            model_config=self.model_config,
            max_req_pool_size=req_pool_padding_index,
            gpu_id=gpu_id,
            global_rank=global_rank,
            num_total_pages=num_total_pages,
            prefix_granularity=geometry.prefix_granularity,
            overlap_schedule_depth=self.overlap_schedule_depth,
        )
        self.model_executor = create_model_executor(
            server_args=server_args,
            config=model_executor_config,
            model_runner=target,
            draft_model_runner=draft,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            draft_attn_backend=draft_attn_backend,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
        )

        # Per-rank GPU memory breakdown (weights by group, KV/graph/non-torch).
        # rank0 only; best-effort, never fails startup.
        if attn_tp_rank == 0:
            log_gpu_memory_summary(
                target.model,
                gpu_id,
                global_rank,
                logger,
                draft_model=draft.model if draft is not None else None,
                kv_pool=token_to_kv_pool,
                draft_kv_pool=draft_token_to_kv_pool,
            )

        self.max_model_len = self.model_config.context_len
        self.max_single_request_tokens = self.model_config.context_len
        self.max_req_input_len = self.model_config.context_len - 1
        self.attn_tp_size = server_args.attn_tp_size or mapping.attn.tp_size
        self.world_size = server_args.world_size or mapping.world_size
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_cpu_group = pg_manager.get_process_group(
            "gloo", server_args.mapping.attn.tp_group
        )
        self._pending_cache_event_payloads: OrderedDict[tuple[str, int], dict] = (
            OrderedDict()
        )
        # All ranks submit identical cache plans (the C++ scheduler is mirrored),
        # so a local in-flight counter mirrors across ranks: if it's 0 here, no
        # rank has anything pending. Lets us skip the TP collective in
        # _commit_cache_results entirely when nothing is in flight.
        self._num_inflight_cache_ops = 0
        self.dp_rank = dp_rank
        self.dp_size = mapping.attn.dp_size
        self.has_dp = mapping.has_attn_dp
        if self.has_dp:
            self.world_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.world_group
            )
            self._dp_local_info = torch.zeros(1, 3, dtype=torch.int32)
            self._dp_global_info = torch.zeros(mapping.world_size, 3, dtype=torch.int32)
        if server_args.enable_kvstore:
            if server_args.kvstore_storage_backend is not None:
                raise NotImplementedError(
                    "the cache-group scheduler has no L3 storage tier; unset "
                    "--kvstore-storage-backend"
                )
            self.l2_cache_executor = L2CacheExecutor(
                device_pool=token_to_kv_pool,
                draft_pool=draft_token_to_kv_pool,
                host_ratio=server_args.kvstore_ratio,
                host_size_gb=server_args.kvstore_size,
                io_backend=server_args.kvstore_io_backend,
            )
            num_host_pages = self.l2_cache_executor.num_host_pages
        else:
            self.l2_cache_executor = None
            num_host_pages = 0

        self._kv_events_enabled = (
            EventPublisherFactory.is_enabled(server_args.kv_events_config)
            and attn_tp_rank == 0
        )

        self._pd_cache_enabled = server_args.disaggregation_mode in (
            "prefill",
            "decode",
        )
        if self._pd_cache_enabled:
            if not token_to_kv_pool.arena.supports_disaggregation:
                raise RuntimeError(
                    "PD disaggregation requires a unified cache contract"
                )
            unsupported = []
            if server_args.enable_mixed_batch:
                unsupported.append("mixed prefill/decode batches")
            if (
                server_args.speculative_algorithm is not None
                and server_args.disaggregation_layerwise_interval > 0
                and not getattr(
                    self.model_executor.drafter,
                    "supports_pd_layerwise_finalization",
                    False,
                )
            ):
                unsupported.append(
                    f"{server_args.speculative_algorithm} layerwise transfer"
                )
            if server_args.enable_memory_saver:
                unsupported.append("memory saver/release")
            # Prefill is forced onto the non-overlap loop by
            # should_use_overlap_schedule(). Decode uses the ordinary overlap
            # loop and the scheduler's one-step protected cache reservation.
            if (
                self.use_overlap_schedule
                and server_args.disaggregation_mode != "decode"
            ):
                unsupported.append("overlap scheduling outside the Decode role")
            backend = server_args.disaggregation_transfer_backend
            if getattr(backend, "value", backend) != "mooncake":
                unsupported.append("non-Mooncake transfer backend")
            if unsupported:
                raise NotImplementedError(
                    "Paged-cache PD currently does not support: "
                    + ", ".join(unsupported)
                )
        # Backend/pool compatibility is validated inside ModelExecutor
        # (validate_scheduler_config), before CUDA-graph capture.
        self._cache_groups = cache_groups
        scheduler_cfg = make_config(
            num_device_pages=geometry.num_device_pages,
            max_scheduled_tokens=max_scheduled_tokens,
            max_batch_size=per_rank_max_batch,
            prefix_granularity=geometry.prefix_granularity,
            num_host_pages=num_host_pages,
            disable_l2_cache=not server_args.enable_kvstore,
            enable_l3_storage=server_args.kvstore_storage_backend is not None,
            role=server_args.disaggregation_mode,
            enable_kv_cache_events=self._kv_events_enabled,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=self.overlap_schedule_depth,
            disable_prefix_cache=not server_args.enable_prefix_caching,
            prefix_replay_tokens=prefix_replay_tokens,
            cache_groups=cache_groups,
            enable_mixed_prefill_decode=server_args.enable_mixed_batch,
        )
        scheduler_cfg.enable_pd_cache = self._pd_cache_enabled
        logger.info(
            "Scheduler config: prefix_granularity=%s num_device_pages=%s "
            "max_scheduled_tokens=%s decode_input_tokens=%s "
            "overlap_schedule_depth=%s disable_l2_cache=%s "
            "max_batch_size=%s (global max_num_seqs=%s, dp_size=%s) "
            "disable_prefix_cache=%s prefix_replay_tokens=%s "
            "cache_groups=%s",
            scheduler_cfg.prefix_granularity,
            scheduler_cfg.num_device_pages,
            scheduler_cfg.max_scheduled_tokens,
            scheduler_cfg.decode_input_tokens,
            scheduler_cfg.overlap_schedule_depth,
            scheduler_cfg.disable_l2_cache,
            scheduler_cfg.max_batch_size,
            server_args.max_num_seqs,
            self.dp_size,
            scheduler_cfg.disable_prefix_cache,
            scheduler_cfg.prefix_replay_tokens,
            [group.group_id for group in cache_groups],
        )
        self.scheduler = Scheduler(scheduler_cfg)
        self.max_single_request_tokens = self.scheduler.max_single_request_tokens()
        self.max_model_len = min(
            self.model_config.context_len, self.max_single_request_tokens
        )
        input_reserve = (
            1
            if server_args.disaggregation_mode == "prefill"
            else max(decode_input_tokens, 1)
        )
        self.max_req_input_len = self.max_model_len - input_reserve
        if self.max_req_input_len < 1:
            raise RuntimeError(
                "Paged cache cannot admit one request with the configured "
                f"decode reserve: max_single_request_tokens="
                f"{self.max_single_request_tokens}, reserve={input_reserve}"
            )
        logger.info(
            "Single-request token limit: cache=%s model=%s effective=%s max_input=%s",
            self.max_single_request_tokens,
            self.model_config.context_len,
            self.max_model_len,
            self.max_req_input_len,
        )
        token_to_kv_pool.bind_cache_scheduler(self.scheduler)
        if attn_tp_rank == 0:
            self.kv_event_publisher = EventPublisherFactory.create(
                server_args.kv_events_config,
                attn_dp_rank=dp_rank,
            )
        else:
            self.kv_event_publisher = NullEventPublisher(attn_dp_rank=dp_rank)

        self._init_interprocess_comm()
        self._init_load_snapshot_publisher()

        # Pause/resume control state. Shared with the request handler, which
        # drives the control-request side; the event loop reads the gate.
        self._pause = PauseController(self.send_to_tokenizer)

        # GPU-memory data plane (release/resume_memory_occupation). Reuses the
        # pause controller's drain machinery; frees memory via the memory-saver
        # adapter once the scheduler drains. See memory_occupation.py.
        # Releasing KV is only safe if any prefix cache it backs can be cleared:
        # either prefix caching is off, or the scheduler exposes a clear. Decide
        # once here (static config) and let the controller reject unsafe releases.
        kv_cache_release_allowed = (
            not self.server_args.enable_prefix_caching
            or callable(getattr(self.scheduler, "clear_l1_cache", None))
        )
        self._memory = MemoryOccupationController(
            send_func=self.send_to_tokenizer,
            pause_controller=self._pause,
            adapter=TorchMemorySaverAdapter.create(
                enable=self.server_args.enable_memory_saver
            ),
            enabled=self.server_args.enable_memory_saver,
            reset_caches_fn=self._reset_caches_for_release,
            kv_repair_fn=self._kv_repair_after_wake,
            kv_cache_release_allowed=kv_cache_release_allowed,
        )

        self.metrics = EngineMetrics(
            labels={
                "model_name": server_args.served_model_name,
                "app_key": server_args.app_key or "",
                "dp_rank": str(dp_rank),
            },
            enabled=(
                server_args.enable_metrics
                and attn_tp_rank == 0
                and "prometheus" in (server_args.metrics_reporters or [])
            ),
        )

        self.request_handler = RequestHandler(
            server_args=self.server_args,
            hf_eos_token_id=self.model_config.hf_eos_token_id,
            max_req_len=self.max_model_len - 1,
            vocab_size=self.model_config.vocab_size,
            recv_func=self.recv_from_tokenizer,
            send_func=self.send_to_tokenizer,
            clear_cache_fn=self.scheduler.clear_cache,
            architectures=self.model_config.hf_config.architectures,
            pause_controller=self._pause,
            memory_controller=self._memory,
            model_runner=target,
        )

        self.output_processor = OutputProcesser(
            send_to_tokenizer=self.send_to_tokenizer,
            attn_tp_rank=attn_tp_rank,
            spec_algorithm=self.server_args.speculative_algorithm,
            spec_num_tokens=(
                self.server_args.speculative_num_draft_tokens
                if self.server_args.speculative_algorithm is not None
                else None
            ),
            stream_interval=self.server_args.stream_interval,
            enable_log_request_stats=self.server_args.enable_log_request_stats,
            physical_context_len=(
                self.model_config.context_len + self.server_args.spec_context_pad
            ),
            metrics=self.metrics,
        )
        if server_args.disaggregation_mode != "null":
            assert pd_topology is not None
            kv_args = get_kv_args(
                global_rank,
                global_rank,
                server_args.disaggregation_ib_device,
                token_to_kv_pool,
                model_config=self.model_config,
                draft_model_config=draft_model_config,
            )
            pd_manager_args = KVManagerArgs(
                bootstrap_port=server_args.disaggregation_bootstrap_port,
                dist_init_addr=server_args.dist_init_addr,
                topology=pd_topology,
                enable_metrics=False,
                served_model_name=server_args.served_model_name,
                app_key=server_args.app_key,
                metrics_reporters=server_args.metrics_reporters,
                enable_dp_attention=self.has_dp,
            )
            self.kv_transfer = create_kv_transfer(
                mode=server_args.disaggregation_mode,
                backend=server_args.disaggregation_transfer_backend,
                args=pd_manager_args,
                kv_args=kv_args,
                gloo_group=self.attn_tp_cpu_group,
            )
            self._setup_pd_layerwise_transfer(
                server_args.disaggregation_layerwise_interval
            )
            # EPD: a multimodal prefill node is also the encode->prefill embedding
            # SINK (independent of kv_transfer, its P->D KV source) -- it receives
            # each image's embedding from encode workers over Mooncake so the
            # prefill skips the vision tower. The admission controller owns the
            # receive jobs, the rank-synced admission drain, and the optional NCCL
            # row-shard reassembly; None for decode/encode/text-only nodes.
            from tokenspeed.runtime.epd.prefill_admission import (
                make_epd_prefill_admission,
            )

            self.epd_admission = make_epd_prefill_admission(
                server_args,
                global_rank,
                model_config=self.model_config,
                model_executor=self.model_executor,
                mapping=mapping,
                attn_tp_rank=self.attn_tp_rank,
                attn_tp_size=self.attn_tp_size,
                attn_tp_cpu_group=self.attn_tp_cpu_group,
                pg_manager=pg_manager,
            )
            # Staged EPD request payloads (request_id -> (spec, state, bootstrap)),
            # held here while the controller (rid-keyed, like kv_transfer) runs the
            # async receive; popped in _drain_ready_epd_embeddings on admit/abort.
            self._epd_staged: dict = {}
        else:
            self.kv_transfer = None
            self.epd_admission = None
            self._epd_staged: dict = {}

    def _setup_pd_layerwise_transfer(self, interval: int) -> None:
        if not isinstance(self.kv_transfer, DisaggPrefillExecutor):
            return
        if interval <= 0:
            return

        from tokenspeed.runtime.pd.utils import StepCounter

        draft_attn_backend = self.model_executor.draft_attn_backend
        step_counter = StepCounter(self.model_executor.device, self.gpu_id)
        self.model_executor.attn_backend.register_step_counter(step_counter)
        if draft_attn_backend is not None:
            self.model_executor.register_draft_final_step_counter(step_counter)
        self.kv_transfer.register_layerwise_step_counter(step_counter, interval)

    def _is_epd_request(self, state) -> bool:
        """True iff this request's images are encode-routed (smg injected per-image
        encode handshakes) -- it must wait for its embeddings (staged via the EPD
        admission controller, polled in _drain_ready_epd_embeddings) before being
        scheduled. Caller guards on self.epd_admission (only a multimodal prefill
        node has one); everything else admits immediately.
        """
        mm = getattr(state, "multimodal_inputs", None)
        return mm is not None and any(
            getattr(it, "encode_handshake", None) for it in mm.mm_items
        )

    def _assert_epd_embeddings_received(self, multimodal_context) -> None:
        """EPD invariant: every handshaked item is filled with its embedding by the
        async EPD admission drain (EpdPrefillAdmission.drain) BEFORE admission, so by
        it is already encoded. This is a defensive check, not a receive: a handshaked
        item that reached the forward un-received leaked past async admission (the
        only EPD admission path) -- fail loud instead of running the tower or
        publishing shard-only rows. No-op for non-EPD / text-only requests.
        """
        if (
            self.epd_admission is None
            or multimodal_context is None
            or not multimodal_context.has_extend_inputs()
        ):
            return
        for mm in multimodal_context.mm_inputs:
            if mm is None:
                continue
            missing = [
                i
                for i, item in enumerate(mm.mm_items)
                if getattr(item, "encode_handshake", None) is not None
                and item.encoded is None
            ]
            if missing:
                raise RuntimeError(
                    f"EPD: handshaked items {missing} reached the prefill forward "
                    "un-received; they must be admitted via the EPD admission drain"
                )

    def _drain_ready_epd_embeddings(self) -> None:
        """Admit EPD requests whose async embedding receives completed this cycle.

        The EpdPrefillAdmission controller DECIDES (poll + rank-lockstep MIN
        all-reduce + reassemble) and returns (admitted, failed); here we ACT on
        those decisions with the EventLoop's collaborators -- register/abort the
        P->D sender, submit admitted requests, finish failed ones. No-op (and no
        collective) on non-EPD nodes.
        """
        if self.epd_admission is None:
            return
        # Pause gate: withhold EPD admission while paused, mirroring the non-EPD
        # admit_blocked gate -- else the drain below would submit and RUN reassembled
        # specs during the pause. Staged receives wait in _pending until resume.
        # Rank-safe: admit_blocked is rank-identical, so all ranks skip together.
        if self._pause.admit_blocked:
            return
        admitted_ids, failed_ids = self.epd_admission.drain()
        for rid in failed_ids:
            spec, state, bootstrap = self._epd_staged.pop(rid)
            # Signal the dual-dispatched decode that this request failed so its KV
            # receiver fails (FailedEvent -> _process_kv_transfer_events abort)
            # instead of waiting forever for KV the prefill will never send. The
            # prefill never registered a P->D sender (deferred to admission), so the
            # decode has no other reliable way to learn (heartbeat only trips on a
            # dead prefill /health). Best-effort: only reaches decodes that already
            # pre-allocated.
            if (
                isinstance(self.kv_transfer, DisaggPrefillExecutor)
                and bootstrap is not None
            ):
                try:
                    self.kv_transfer.abort(rid, bootstrap)
                except Exception as exc:  # never let it wedge the loop
                    logger.warning(
                        "EPD abort->decode signal failed for rid=%s: %s",
                        rid,
                        exc,
                    )
            state.set_finish_with_abort("EPD embedding receive failed or timed out")
            self.output_processor.publish_finished_at_admission(rid, state)
        admitted_specs = []
        for rid in admitted_ids:
            spec, state, bootstrap = self._epd_staged.pop(rid)
            # Aborted mid-receive (no abort path, so drain still returns it admitted):
            # don't register the P->D sender or submit -- that runs a wasted forward
            # and leaks the sender. Stream its finish instead.
            if state.finished:
                self.output_processor.publish_finished_at_admission(rid, state)
                continue
            # Register the P->D sender now (deferred from admission) -- the request
            # is about to enter the scheduler.
            if self.kv_transfer is not None:
                self.kv_transfer.register(rid, bootstrap)
            admitted_specs.append(spec)
        if admitted_specs:
            self.scheduler.submit_requests(admitted_specs)
        elif self.epd_admission.has_pending():
            # Nothing advanced this cycle but requests are still receiving; yield the
            # GIL so the Python daemon transfer/recv threads run (rank-consistent:
            # admitted/leftover are rank-identical here).
            time.sleep(0.0005)

    def _commit_cache_results(self) -> None:
        if self.l2_cache_executor is None:
            return
        cache_results = self.l2_cache_executor.poll_results()
        self._num_inflight_cache_ops -= len(cache_results)
        for event in cache_results:
            payload = cache_event_to_payload(event)
            self._pending_cache_event_payloads[cache_event_key(payload)] = payload

        # The gather below is a collective, but cache-op completion is async and
        # not lock-step across ranks, so local state (_num_inflight_cache_ops /
        # _pending_cache_event_payloads) diverges transiently. A rank-local skip
        # would let some ranks gather while others return, deadlocking the group.
        # Agree on the skip via a cheap single-int all_reduce.
        # NOTE: For non-DFLASH algorithms, cache ops are deterministic across
        # ranks, so the local short-circuit is safe and avoids collective overhead.
        local_has_work = bool(
            self._num_inflight_cache_ops != 0 or self._pending_cache_event_payloads
        )
        if self.server_args.speculative_algorithm in ("DFLASH", "DSPARK"):
            if not self._cache_group_has_work(local_has_work):
                return
        else:
            if not local_has_work:
                return

        ready_payloads = self._pop_ready_cache_event_payloads()
        if not ready_payloads:
            return
        logger.debug(
            "[cache_poll] got %s synchronized results, advancing scheduler",
            len(ready_payloads),
        )
        ec = ExecutionEvent()
        for payload in ready_payloads:
            e = cache_event_from_payload(payload)
            logger.debug(
                "[cache_poll] event: op_id=%s type=%s",
                e.op_id,
                type(e).__name__,
            )
            ec.add_event(e)
        self.scheduler.advance(ec)
        logger.debug("[cache_poll] scheduler.advance() done")
        self._publish_scheduler_kv_events()

    def _publish_scheduler_kv_events(self) -> None:
        raw_events = drain_scheduler_kv_events(
            self.scheduler,
            enabled=self._kv_events_enabled,
        )
        if not raw_events:
            return

        events = scheduler_kv_events_to_wire_events(raw_events)
        if not events:
            return

        self.kv_event_publisher.publish(
            KVEventBatch(ts=time.time(), events=events, attn_dp_rank=self.dp_rank)
        )

    def _cache_group_has_work(self, local_has_work: bool) -> bool:
        """Whether ANY attn-tp rank has cache work this step (unanimous via a
        single-int MAX all_reduce, far cheaper than the payload gather it
        guards). Deciding from rank-local state alone deadlocks the group; see
        _commit_cache_results.

        Args:
            local_has_work: This rank's view of whether any cache op is in
                flight or any polled payload awaits commit.

        Returns:
            ``True`` if any rank has work (all must gather); ``False`` only when
            every rank is idle.
        """
        if self.attn_tp_size == 1:
            return local_has_work
        flag = torch.tensor([1 if local_has_work else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.attn_tp_cpu_group)
        return bool(flag.item())

    def _pop_ready_cache_event_payloads(self) -> list[dict]:
        local_payloads = list(self._pending_cache_event_payloads.values())
        if self.attn_tp_size == 1:
            ready_payloads = local_payloads
        else:
            gathered_payloads = [None] * self.attn_tp_size
            dist.all_gather_object(
                gathered_payloads,
                local_payloads,
                group=self.attn_tp_cpu_group,
            )
            ready_payloads = pop_common_cache_event_payloads(gathered_payloads)
            if self.attn_tp_rank == 0 and cache_sync_debug_enabled():
                pending_ops = [
                    [(payload["kind"], payload["op_id"]) for payload in rank_payloads]
                    for rank_payloads in gathered_payloads
                ]
                if len({tuple(rank_ops) for rank_ops in pending_ops}) > 1:
                    logger.info(
                        "[cache_sync] rank=%s pending_ops=%s ready_ops=%s",
                        self.global_rank,
                        pending_ops,
                        [
                            (payload["kind"], payload["op_id"])
                            for payload in ready_payloads
                        ],
                    )

        for payload in ready_payloads:
            self._pending_cache_event_payloads.pop(cache_event_key(payload), None)
        return ready_payloads

    def _dispatch_forward(
        self,
        forward_op,
        sampling_params_list,
        execution_plan,
        dp_metadata=None,
        stats=None,
        grammar_inputs=None,
        cache_zero_event=None,
    ):
        """Execute one forward step; return (results, on_first_token).

        results is None when the step produces no model output (Path 2/3).
        Both event_loop and event_loop_overlap call this method; they differ
        only in *when* they call post_process on the returned results.

        Path 1 — no PD:              run forward, return (results, None)
        Path 2 — decode, extend:     trigger RDMA receive, return (None, None)
        Path 3 — prefill, decode:    send KV to decode side, return (None, None)
        Path 4 — prefill, extend:    run prefill forward, return (results, on_first_token)
        """
        if stats is None:
            stats = {}
        dp_global_num_tokens = (
            dp_metadata.global_num_tokens if dp_metadata is not None else None
        )
        dp_global_bs = (
            dp_metadata.global_batch_size if dp_metadata is not None else None
        )
        dp_all_decode_or_idle = (
            dp_metadata.all_decode_or_idle if dp_metadata is not None else False
        )
        dp_all_extend = dp_metadata.all_extend if dp_metadata is not None else False
        multimodal_context = self._get_multimodal_context_for_forward(forward_op)

        if self.kv_transfer is None:
            # Path 1: normal (no disaggregation)
            self.model_executor.reset_valid_cache_length(forward_op)
            return (
                self.model_executor.execute_forward_op_with_log(
                    forward_op,
                    sampling_params_list,
                    dp_global_num_tokens=dp_global_num_tokens,
                    dp_global_bs=dp_global_bs,
                    dp_all_decode_or_idle=dp_all_decode_or_idle,
                    dp_all_extend=dp_all_extend,
                    grammar_inputs=grammar_inputs,
                    multimodal_context=multimodal_context,
                    **stats,
                ),
                None,
            )

        elif isinstance(self.kv_transfer, DisaggDecodeExecutor):
            # Decode node
            if forward_op.num_extends() > 0 and not forward_op.is_local_prefill():
                # Path 2: new requests waiting for remote KV — trigger RDMA receive
                self.model_executor.prepare_remote_cache_slots(
                    list(forward_op.request_pool_indices[: forward_op.num_extends()])
                )
                self.kv_transfer.reset_valid_cache_length(
                    forward_op,
                    self.model_executor.runtime_states,
                    self.model_executor.execution_stream,
                    self.model_executor.device,
                )
                if self._pd_cache_enabled and cache_zero_event is not None:
                    # Page zeroing runs asynchronously on a CUDA stream,
                    # while Mooncake/GPUDirect writes are not ordered by that
                    # stream. Do not publish the destination manifest until the
                    # newly assigned pages are fully sanitized.
                    cache_zero_event.synchronize()
                self.kv_transfer.execute(forward_op)
                return None, None
            else:
                # Decode and local recovery-prefill batches execute normally.
                self.model_executor.reset_valid_cache_length(forward_op)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        dp_all_extend=dp_all_extend,
                        multimodal_context=multimodal_context,
                        **stats,
                    ),
                    None,
                )

        else:
            # Prefill node (only reached from event_loop, never event_loop_overlap)
            if not isinstance(self.kv_transfer, DisaggPrefillExecutor):
                raise TypeError("kv_transfer must be a DisaggPrefillExecutor.")
            if forward_op.num_extends() == 0:
                # Path 3: all prefill done — send KV to decode side
                self.kv_transfer.execute(forward_op)
                return None, None
            else:
                # Path 4: extend batch — run prefill forward
                self.model_executor.reset_valid_cache_length(forward_op)
                self.kv_transfer.prepare_prefill(forward_op)
                # EPD invariant: handshaked items are filled by the async
                # EPD admission drain before admission; assert none reached
                # the forward un-received (no-op for non-EPD / text-only requests).
                self._assert_epd_embeddings_received(multimodal_context)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        dp_all_extend=dp_all_extend,
                        grammar_inputs=grammar_inputs,
                        multimodal_context=multimodal_context,
                        capture_next_input_ids=True,
                        **stats,
                    ),
                    self.kv_transfer.store_prefill_token,
                )

    def _get_multimodal_context_for_forward(self, forward_op):
        if not self.model_config.is_multimodal_active:
            return None

        num_extends = forward_op.num_extends()
        mm_inputs = []
        has_mm = False
        for index, rid in enumerate(forward_op.request_ids):
            state = self.output_processor.rid_to_state.get(rid)
            if state is not None and index < num_extends:
                state.maybe_extend_multimodal_mrope_positions()
            item = getattr(state, "multimodal_inputs", None) if state else None
            mm_inputs.append(item)
            has_mm = has_mm or item is not None
        if not has_mm:
            return None

        from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext

        return MultimodalForwardContext(
            mm_inputs=mm_inputs,
            extend_prefix_lens=list(forward_op.extend_prefix_lens),
            extend_seq_lens=list(forward_op.input_lengths[:num_extends]),
        )

    def _submit_cache_ops(self, execution_plan) -> None:
        if self.l2_cache_executor is None:
            return
        self.l2_cache_executor.submit_plan(execution_plan)
        for op in execution_plan.cache:
            if isinstance(op, Cache.WriteBackOp):
                self._num_inflight_cache_ops += len(op.op_ids)
            elif isinstance(op, Cache.LoadBackOp):
                self._num_inflight_cache_ops += len(op.op_ids)
            else:
                raise ValueError(f"unsupported cache op kind: {type(op).__name__}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model_config(
        self, model_path: str, is_draft_worker: bool = False
    ) -> ModelConfig:
        server_args = self.server_args
        quantization = server_args.quantization
        dtype = server_args.dtype
        if is_draft_worker:
            quantization = server_args.speculative_draft_model_quantization
            if dtype == "auto":
                # A draft is fed the target's hidden states and borrows its
                # embedding and LM head, so the two dtypes have to agree.
                dtype = self.model_config.dtype
        return ModelConfig(
            model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.max_model_len,
            model_override_args=server_args.hf_overrides,
            dtype=dtype,
            quantization=quantization,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
        )

    def _init_distributed(self) -> float:
        max_num_input_tokens = (
            self.server_args.chunked_prefill_size
            if self.server_args.chunked_prefill_size > 0
            else self.server_args.max_prefill_tokens + self.server_args.max_model_len
        )
        distributed_config = DistributedConfig.from_server_args(
            server_args=self.server_args,
            port_args=self.port_args,
            gpu_id=self.gpu_id,
            global_rank=self.global_rank,
            hidden_size=self.model_config.hidden_size,
            max_num_tokens=max_num_input_tokens,
        )
        return DistributedInitializer.initialize(distributed_config)

    def _init_interprocess_comm(self):
        context = zmq.Context(2)
        if self.attn_tp_rank == 0:
            if self.server_args.zmq_msgpack:
                # SMG drives the scheduler directly: it binds the sockets and
                # this engine connects in over the msgpack wire (see zmq_msgpack).
                self.recv_from_tokenizer, self.send_to_tokenizer = (
                    self._init_msgpack_transport(context)
                )
            else:
                self.recv_from_tokenizer = IpcReceiver(
                    get_zmq_socket(
                        context,
                        zmq.PULL,
                        self.port_args.scheduler_input_ipc_name,
                        False,
                    )
                )
                self.send_to_tokenizer = IpcSender(
                    get_zmq_socket(
                        context, zmq.PUSH, self.port_args.tokenizer_ipc_name, False
                    )
                )
        else:
            self.recv_from_tokenizer = None
            self.send_to_tokenizer = _NullSender()

    def _init_load_snapshot_publisher(self) -> None:
        if self.attn_tp_rank == 0 and not self.server_args.zmq_msgpack:
            self.load_snapshot_publisher = LoadSnapshotPublisher(
                self.port_args.metrics_ipc_name,
                self.dp_rank,
                self.server_args.load_watch_interval,
            )
        else:
            # Direct-ZMQ will project the observation onto ordinary output
            # batches; it must not create this standard-serving PUSH socket.
            self.load_snapshot_publisher = NullLoadSnapshotPublisher()

    def _init_msgpack_transport(self, context):
        """Complete the SMG startup handshake and return the wrapped msgpack
        input/output sockets."""
        from tokenspeed.runtime.engine import zmq_msgpack, zmq_wire

        # Each DP rank dials SMG with its own engine identity
        # (zmq_engine_index + dp_rank): the frontend's grouped worker awaits
        # dp_size engines on one socket set and tells the ranks apart — and
        # routes inputs back — by this index. The ready response carries the
        # true dp rank/size alongside.
        engine_index = self.server_args.zmq_engine_index + self.dp_rank
        geometry = self._scheduler_cache_geometry
        ready_response = zmq_wire.WireEngineCoreReadyResponse(
            max_model_len=self.model_config.context_len,
            num_gpu_blocks=geometry.num_device_pages,
            prefix_granularity=geometry.prefix_granularity,
            dtype=_wire_dtype(self.model_config.dtype),
            multimodal_encoder_dtype=self.multimodal_encoder_dtype,
            vllm_version=f"tokenspeed-{_tokenspeed_version()}",
            world_size=self.world_size,
            data_parallel_size=self.dp_size,
            tensor_parallel_size=self.attn_tp_size,
            data_parallel_rank=self.dp_rank,
            max_num_seqs=self.server_args.max_num_seqs,
            # chunked_prefill_size=-1 means "disabled"; the wire field is a
            # non-negative integer for the frontend, so clamp to 0 (= no cap).
            max_num_batched_tokens=max(0, self.server_args.chunked_prefill_size),
            instance_id=self.server_args.served_model_name or self.server_args.model,
            kv_cache_size_tokens=self.max_total_num_tokens,
        )
        return zmq_msgpack.connect_msgpack_engine(
            context,
            self.server_args.zmq_handshake_endpoint(),
            engine_index,
            ready_response,
            self.model_config.vocab_size,
            enable_output_logprobs=self.server_args.enable_output_logprobs,
        )

    # ------------------------------------------------------------------
    # Shared step helpers
    # ------------------------------------------------------------------

    def _reap_or_keep_buffered_spec(self, spec) -> bool:
        """Resolve a buffered spec on resume; return True if it should be admitted.

        A buffered spec was already registered in ``rid_to_state`` before it was
        withheld, so if it was aborted while paused it never reached the
        scheduler and the forward path can never reap it. Handle that here:

        - state missing  -> already published and reaped; drop silently.
        - state finished -> aborted in place. Stream a terminating finish for
          pause-initiated aborts (the passive client is still waiting) and drop
          the registered state so the rid does not leak; client-initiated aborts
          already tore down their own state, so just reap.
        - otherwise      -> still live; admit it.
        """
        state = self.output_processor.rid_to_state.get(spec.request_id)
        if state is None:
            return False
        if state.finished:
            self.output_processor.reap_finished_orphan(spec.request_id, state)
            return False
        return True

    def _request_abort_or_mark(
        self, request_id: str, _reason: str, *, notify_client: bool = False
    ) -> None:
        """Mark an abort; scheduler release follows the normal lifecycle event."""
        if notify_client:
            self.output_processor.mark_abort(request_id, notify_client=True)
        else:
            self.output_processor.mark_abort(request_id)

    def _process_new_requests(self):
        recv_reqs = self.request_handler.recv_reqs()
        # Snapshot the pause state before dispatch: process_requests may flip it
        # mid-batch. If it was not blocked before but is after, a pause control
        # message was processed in this very batch — which is what makes the
        # FIFO edge below detectable (see TODO(pause-fifo)).
        pause_blocked_before = self._pause.admit_blocked
        new_req_specs, new_req_states, bootstrap_infos, abort_rids = (
            self.request_handler.process_requests(recv_reqs)
        )
        # Sweep TTL-expired abort markers every iteration. Without this
        # the map only gets cleaned inside ``mark_abort``, so a burst of
        # stale-cancel traffic followed by silence leaves the last batch
        # of entries sitting past their TTL (and potentially re-aborting
        # reused rids). Amortized O(1): expired entries are always at
        # the front of the insertion-ordered dict.
        self.output_processor.sweep_pending_aborts()
        # Abort both registered and grammar-queued requests. Without the
        # grammar_manager.mark_abort call, a request aborted mid-compile
        # would finish compiling and get admitted before being noticed.
        grammar_manager = self.request_handler.grammar_manager
        for rid in abort_rids:
            self._request_abort_or_mark(rid, "client cancelled request")
            grammar_manager.mark_abort(rid)

        # A pause(mode="abort") cancels every in-flight request through the same
        # marker path as a client abort; they finish on their next scheduled
        # step, then the drain check resolves the pause reply.
        if self._pause.consume_abort_all():
            for rid in list(self.output_processor.rid_to_state.keys()):
                # notify_client=True: pause aborts a passive client's request,
                # so it must receive a terminating finish (unlike a client abort).
                self._request_abort_or_mark(
                    rid, "request aborted by pause", notify_client=True
                )
                grammar_manager.mark_abort(rid)

        # abort/wait also cancel requests still compiling in the grammar queue:
        # they are not yet in rid_to_state or the scheduler, so the sweep above
        # and the drain check both miss them. A finished state makes the next
        # get_ready_grammar_requests pass publish them instead of admitting, so
        # they never run under post-resume weights or strand the drain.
        if self._pause.consume_cancel_grammar():
            for _, state, _ in grammar_manager.grammar_queue:
                state.set_finish_with_abort("Aborted by pause", notify_client=True)

        # On resume, flush specs buffered while paused even when no new request
        # arrives this iteration. This must run before the ``if not ready:
        # return`` guard below, which would otherwise strand buffered specs
        # until the next inbound request. Specs aborted while paused are reaped
        # in place (terminating finish + state cleanup) rather than admitted, so
        # they don't burn a scheduler slot or leak their rid — see
        # ``_reap_or_keep_buffered_spec``.
        if not self._pause.admit_blocked and self._pause.buffered_specs:
            specs = [
                spec
                for spec in self._pause.take_buffered_specs()
                if self._reap_or_keep_buffered_spec(spec)
            ]
            if specs:
                self.scheduler.submit_requests(specs)

        # Partition new requests by grammar readiness. Compile-bound requests
        # are queued in GrammarManager and admitted in a later iteration when
        # their futures resolve (see _drain_ready_grammar_requests below).
        ready = []
        for spec, state, bootstrap in zip(
            new_req_specs, new_req_states, bootstrap_infos
        ):
            # Requests pre-marked finished (e.g. invalid session ID aborted
            # in RequestHandler) skip grammar compilation entirely — we'd
            # just be wasting a compile slot on a response we're about to
            # abort anyway, and the terminal response would be delayed by
            # the compile/timeout window.
            if state.finished:
                ready.append((spec, state, bootstrap))
                continue
            if grammar_manager.process_req_with_grammar(state):
                ready.append((spec, state, bootstrap))
            else:
                grammar_manager.add_to_queue(spec, state, bootstrap)

        # Drain any previously-queued requests whose grammar just finished
        # compiling. With attn_tp > 1 this also drives the per-iter all_gather
        # that keeps grammar admission in sync across ranks.
        ready.extend(grammar_manager.get_ready_grammar_requests())

        if not ready:
            return

        admitted_specs = []
        for spec, state, bootstrap in ready:
            # Grammar-aborted (invalid grammar, timed-out compile, or missing
            # backend) requests must not enter the scheduler — they have no
            # valid grammar to mask logits with, and we don't want to spend a
            # prefill slot on a request that's already finished. Publish the
            # finish_reason directly so the client still gets a response.
            if state.finished:
                self.output_processor.publish_finished_at_admission(
                    spec.request_id, state
                )
                continue

            if self._pd_cache_enabled:
                if bootstrap is None:
                    raise ValueError(
                        "Paged cache PD request is missing bootstrap information"
                    )
            if isinstance(self.kv_transfer, DisaggDecodeExecutor):
                state.computed_length = state.input_length
            self.output_processor.register(spec.request_id, state)
            is_epd = self.epd_admission is not None and self._is_epd_request(state)
            # EPD: DEFER the P->D sender registration to admission (in
            # EpdPrefillAdmission.drain, just before submit_requests). Registering
            # it now -- while the request is staged and NOT yet in the C++ scheduler
            # -- would let DisaggPrefillExecutor.generate_events poll the sender and
            # emit a BootstrappedEvent that the scheduler's requests_.at(rid) THROWS
            # on (no such request yet). Non-EPD requests register now (submitted this
            # same call).
            if self.kv_transfer is not None and not is_epd:
                self.kv_transfer.register(spec.request_id, bootstrap)

            # EPD prefill: hold a request whose images are encode-routed OUT of the
            # scheduler until its per-image embeddings have been received (started
            # here, polled in EpdPrefillAdmission.drain, which registers the P->D
            # sender + submits once ready). It is output_processor-registered above;
            # the sender registration + submission are both deferred. Non-EPD
            # requests admit immediately as before. Rank-identical because `ready` is
            # rank-synced (recv_reqs broadcast + grammar gather).
            if is_epd:
                self.epd_admission.stage(
                    spec.request_id, state.multimodal_inputs.mm_items
                )
                self._epd_staged[spec.request_id] = (spec, state, bootstrap)
            else:
                admitted_specs.append(spec)

        # Pause gate: while paused, withhold new requests from the scheduler
        # (running requests keep stepping); buffered specs are flushed on resume
        # above, ahead of any newly-admitted ones, preserving FIFO order.
        #
        # TODO(pause-fifo): recv_reqs() drains the socket non-blocking, so a
        # generate request that arrived *before* a pause control message can be
        # coalesced into the same batch and reach here after the pause flipped
        # admit_blocked. Such a pre-pause request is buffered as post-pause work
        # instead of running (wait) / being aborted (abort). Correct handling
        # needs the batch processed as an ordered stream that respects the
        # control request's FIFO position. Tracked as a follow-up; until then we
        # warn when the coalescing condition is observed so it is not silent.
        if self._pause.admit_blocked:
            if admitted_specs and not pause_blocked_before:
                logger.warning(
                    "Pause engaged in the same recv batch as %d generate "
                    "request(s) (rids=%s); their FIFO order relative to the "
                    "pause is not preserved, so a pre-pause request may be "
                    "buffered as post-pause work and run only after resume. "
                    "See TODO(pause-fifo).",
                    len(admitted_specs),
                    [spec.request_id for spec in admitted_specs],
                )
            self._pause.buffer_specs(admitted_specs)
            return

        if admitted_specs:
            self.scheduler.submit_requests(admitted_specs)

    @nvtx_range("loop:commit", color="rapids")
    def _commit_forward_results(
        self,
        forward_op,
        results: ModelExecutionResult,
        on_first_token=None,
    ):
        self.request_handler.forward_ct += 1
        forward_mode = ForwardMode.from_num_extends(
            forward_op.num_extends(),
            len(forward_op.request_ids),
        )
        self.request_handler._profile_batch_predicate(forward_mode)

        # post_process_forward_op calls sync() — after this, CPU tensors are ready
        is_prefill_instance = isinstance(self.kv_transfer, DisaggPrefillExecutor)
        request_changes = self.output_processor.post_process_forward_op(
            forward_op,
            results,
            is_prefill_instance=is_prefill_instance,
            on_first_token=on_first_token,
        )

        # Accumulate decode stats from synced results (no GPU sync)
        if forward_op.num_extends() <= 0:
            bs = len(forward_op.request_ids)
            self.model_executor.accumulate_decode_stats(results, bs)

        return request_changes

    def _get_forward_op(self, execution_plan):
        """Return the next forward op from the given plan, or None if there is nothing to run."""
        forward_ops = execution_plan.forward
        if len(forward_ops) == 0 or len(forward_ops[0].request_ids) == 0:
            return None
        return forward_ops[0]

    def _process_kv_transfer_events(self, kv_transfer_events: list) -> list:
        processed = []
        for event in kv_transfer_events:
            processed.append(event)
            if isinstance(event, PD.SucceededEvent) and isinstance(
                self.kv_transfer, DisaggPrefillExecutor
            ):
                req_id = event.request_id
                processed.extend(self.output_processor.finish_prefill_request(req_id))
            elif isinstance(event, PD.RemotePrefillDoneEvent):
                req_id = event.request_id
                bootstrap_token = event.bootstrap_token
                state = self.output_processor.rid_to_state.get(req_id)
                if state is None or not state.to_abort:
                    self.output_processor.on_remote_prefill_done(
                        req_id, bootstrap_token
                    )
                if self._pd_cache_enabled:
                    processed.extend(
                        self.output_processor.finish_remote_prefill_only_request(req_id)
                    )
                if isinstance(self.kv_transfer, DisaggDecodeExecutor):
                    remote_cache_slot = self.kv_transfer.pop_remote_cache_slot(req_id)
                    candidate_info = self.kv_transfer.pop_remote_spec_candidate_ids(
                        req_id
                    )
                    if candidate_info is not None:
                        req_pool_idx, candidate_ids = candidate_info
                        self.model_executor.write_remote_spec_candidate_ids(
                            req_pool_idx, candidate_ids
                        )
                    remaining_state = self.output_processor.rid_to_state.get(req_id)
                    if (
                        remote_cache_slot is not None
                        and remaining_state is not None
                        and not remaining_state.to_abort
                        and not remaining_state.finished
                    ):
                        self.model_executor.mark_remote_cache_ready(remote_cache_slot)
            elif isinstance(event, PD.FailedEvent):
                # A PD/EPD transfer failed: the decode KV receiver timed out (e.g. the
                # prefill aborted on embedding timeout so the KV never arrives), or a
                # transfer errored. Publish the client-visible failure here. An
                # encode-only EPD flow still needs a following Forward.Abort because
                # its C++ FailedEvent handler is a no-op; CachePD FailedEvent
                # atomically terminalizes and fences the leased scheduler resources.
                req_id = event.request_id
                state = self.output_processor.rid_to_state.get(req_id)
                if state is not None:
                    if state.finished:
                        self.output_processor.reap_finished_orphan(req_id, state)
                    else:
                        state.set_finish_with_abort(
                            "PD/EPD remote transfer failed or timed out"
                        )
                        self.output_processor.publish_finished_at_admission(
                            req_id, state
                        )
                    if not self._pd_cache_enabled:
                        abort = ForwardEvent.Abort()
                        abort.request_id = req_id
                        processed.append(abort)
        return processed

    def _dp_sync_and_check(self, forward_op) -> DpForwardMetadata:
        """Synchronize DP ranks with CPU-only metadata.

        All ranks call this before GPU forward work. The gathered metadata is
        used for eager token-aware collectives and for choosing a common padded
        CUDA graph shape during decode.
        """
        import torch.distributed as dist

        executes_model_forward = _forward_op_executes_model_forward(
            forward_op,
            is_disagg_decode=isinstance(self.kv_transfer, DisaggDecodeExecutor),
        )
        num_tokens = sum(forward_op.input_lengths) if executes_model_forward else 0
        batch_size = len(forward_op.request_ids) if executes_model_forward else 0
        if not executes_model_forward:
            forward_mode = ForwardMode.IDLE
        else:
            forward_mode = ForwardMode.from_num_extends(
                forward_op.num_extends(),
                batch_size,
            )

        self._dp_local_info[0, 0] = num_tokens
        self._dp_local_info[0, 1] = batch_size
        self._dp_local_info[0, 2] = int(forward_mode)
        dist.all_gather_into_tensor(
            self._dp_global_info,
            self._dp_local_info,
            group=self.world_cpu_group,
        )
        global_num_tokens = self._dp_global_info[:, 0].tolist()
        global_batch_size = self._dp_global_info[:, 1].tolist()
        global_forward_mode = self._dp_global_info[:, 2].tolist()
        any_rank_has_work = max(global_num_tokens) > 0
        need_idle_forward = num_tokens == 0 and any_rank_has_work
        all_decode_or_idle = all(
            mode
            in (
                int(ForwardMode.DECODE),
                int(ForwardMode.IDLE),
            )
            for mode in global_forward_mode
        )
        # Replicated prefill-graph gate (see PrefillGraph._select_bucket).
        all_extend = all(
            mode == int(ForwardMode.EXTEND) for mode in global_forward_mode
        )
        return DpForwardMetadata(
            global_num_tokens=global_num_tokens,
            global_batch_size=global_batch_size,
            global_forward_mode=global_forward_mode,
            all_decode_or_idle=all_decode_or_idle,
            all_extend=all_extend,
            need_idle_forward=need_idle_forward,
        )

    def _get_scheduler_stats(self):
        """Query scheduler for page usage and queue depth."""
        available = self.scheduler.available_kv_pages()
        active = self.scheduler.active_kv_pages()
        return {
            "num_active_pages": active,
            "num_cached_pages": (
                self._scheduler_cache_geometry.num_usable_pages - available
            ),
            "num_queue_reqs": self.scheduler.waiting_size(),
        }

    def _observe_load_snapshot(self, stats: dict) -> None:
        """Copy the already-sampled scheduler scalars into the publisher mailbox."""
        self.load_snapshot_publisher.observe(
            (
                len(self.output_processor.rid_to_state),
                stats["num_queue_reqs"],
                stats["num_active_pages"],
                stats["num_cached_pages"],
                self._scheduler_cache_geometry.num_usable_pages,
            )
        )

    def _record_scheduler_iteration_metrics(
        self, stats: dict, num_iteration_tokens: int
    ) -> None:
        self.metrics.record_scheduler_iteration(
            running=len(self.output_processor.rid_to_state),
            waiting=stats["num_queue_reqs"],
            num_active_pages=stats["num_active_pages"],
            num_total_pages=self._scheduler_cache_geometry.num_usable_pages,
            num_iteration_tokens=num_iteration_tokens,
        )

    # ------------------------------------------------------------------
    # Pause / resume helpers
    # ------------------------------------------------------------------

    def _reset_caches_for_release(self) -> bool:
        """Invalidate the prefix/single-table cache before KV is discarded on release.

        KV pages are re-mapped + zeroed on wake, so any retained prefix entry
        would be stale. The unsafe case (prefix caching on with no reset) is
        rejected up front in ``MemoryOccupationController.handle_release`` via
        ``kv_cache_release_allowed``, so by the time we get here either a clear
        exists or prefix caching is off (nothing to invalidate). Returns False
        while an asynchronous cache transfer still pins L1 so the release can
        remain pending and retry on the next event-loop iteration.
        """
        clear = getattr(self.scheduler, "clear_l1_cache", None)
        return not callable(clear) or clear()

    def _kv_pools(self) -> list:
        """All KV pools whose pages are tagged ``kv_cache`` — the target pool and
        the draft pool in speculative-decoding runs. Release/repair must walk the
        SAME set, so both derive it here rather than enumerating pools by hand."""
        pools = []
        for attr in ("token_to_kv_pool", "draft_token_to_kv_pool"):
            pool = getattr(self.model_executor, attr, None)
            if pool is not None:
                pools.append(pool)
        return pools

    def _kv_repair_after_wake(self) -> None:
        """Zero re-mapped KV buffers (garbage after re-map) for every KV pool,
        including the draft pool in spec-decode runs — its allocations are tagged
        ``kv_cache`` too, so a wake that skipped it would feed the draft model
        stale KV. FP8 KV scales ride with the weights region, so no scale reset
        is needed here."""
        for pool in self._kv_pools():
            if hasattr(pool, "clear_kv_buffers"):
                pool.clear_kv_buffers()

    def _paused_idle_step(self, prev_forward_op=None, prev_results=None) -> None:
        """Run one iteration under ``PAUSED_ALL`` (keep mode): no new forward
        work, but keep DP ranks in lockstep, service the drain check, and yield
        the CPU so the freeze does not busy-spin a core."""
        if prev_results is not None:
            request_changes = self._commit_forward_results(
                prev_forward_op, prev_results
            )
            advance_forward(self.scheduler, request_changes)
            self._publish_scheduler_kv_events()
            stats = self._get_scheduler_stats()
            self._observe_load_snapshot(stats)

        if self.has_dp:
            dp_metadata = self._dp_sync_and_check(None)
            # While memory is released the weights region is unmapped; an idle
            # forward runs the model and would read freed memory. All DP ranks
            # release together, so skipping the idle forward stays consistent
            # across ranks (the small DP sync above still runs to keep lockstep).
            if dp_metadata.need_idle_forward and not self._pause.released:
                self.model_executor.execute_idle_forward(
                    dp_metadata.global_num_tokens,
                    dp_metadata.global_batch_size,
                    dp_metadata.all_decode_or_idle,
                )

        self._pause.maybe_finish_drain(self.scheduler)
        time.sleep(_PAUSED_IDLE_SLEEP_S)

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def _shutdown_complete(self) -> bool:
        return self.shutdown_event.is_set()

    def event_loop(self):
        """Non-overlapping scheduler loop."""
        while not self._shutdown_complete():
            self._process_new_requests()
            # EPD prefill: admit requests whose async embedding receives completed
            # this cycle (rank-synced). Fixed position right after
            # _process_new_requests so the drain's TP collective ordering is
            # rank-identical every cycle.
            self._drain_ready_epd_embeddings()
            self._commit_cache_results()
            if self._pause.forward_blocked:
                self._paused_idle_step()
                continue
            execution_plan = self.scheduler.next_execution_plan()
            self._publish_scheduler_kv_events()
            cache_zero_event = self.model_executor.zero_cache_pages(
                execution_plan.pages_to_zero
            )
            self._submit_cache_ops(execution_plan)

            forward_op = self._get_forward_op(execution_plan)
            stats = self._get_scheduler_stats()
            self._observe_load_snapshot(stats)
            num_iter_tokens = (
                sum(forward_op.input_lengths) if forward_op is not None else 0
            )

            # DP sync: all ranks must participate even when idle.
            dp_metadata = None
            if self.has_dp:
                dp_metadata = self._dp_sync_and_check(forward_op)
                if dp_metadata.need_idle_forward:
                    self.model_executor.execute_idle_forward(
                        dp_metadata.global_num_tokens,
                        dp_metadata.global_batch_size,
                        dp_metadata.all_decode_or_idle,
                    )
                    self._record_scheduler_iteration_metrics(stats, num_iter_tokens)
                    continue

            request_changes = []

            if forward_op is not None:
                sampling_params_list = self._gather_sampling_params(forward_op)
                grammar_inputs = self._gather_grammar_state(forward_op)
                self._mark_stats_scheduled(forward_op)
                results, on_first_token = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    execution_plan,
                    dp_metadata=dp_metadata,
                    stats=stats,
                    grammar_inputs=grammar_inputs,
                    cache_zero_event=cache_zero_event,
                )
                if results is not None:
                    request_changes.extend(
                        self._commit_forward_results(
                            forward_op, results, on_first_token
                        )
                    )

            if self.kv_transfer is not None:
                kv_transfer_events = self.kv_transfer.generate_events()
                request_changes.extend(
                    self._process_kv_transfer_events(kv_transfer_events)
                )

            if request_changes:
                advance_forward(self.scheduler, request_changes)
                self._publish_scheduler_kv_events()

            # Resolve a deferred abort/wait pause reply once in-flight work drains.
            self._pause.maybe_finish_drain(self.scheduler)

            self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

    def _mark_stats_scheduled(self, forward_op) -> None:
        # Stamp the pre-forward "scheduled" time on each request's stats tracker
        # so the queue/prefill split is anchored before the forward (idempotent:
        # only the first forward a request appears in sets it). --enable-log-request-stats.
        if not self.server_args.enable_log_request_stats or forward_op is None:
            return
        now = time.time()
        rid_to_state = self.output_processor.rid_to_state
        for rid in forward_op.request_ids:
            st = rid_to_state.get(rid)
            if st is not None:
                st.stats.mark_scheduled(now)

    def _gather_sampling_params(self, forward_op) -> list[SamplingParams]:
        """Look up per-request SamplingParams from the output processor. The
        sampling backend does its own flip detection + RNG state management
        internally, so we only need the scalar params here."""
        return [
            self.output_processor.rid_to_state[rid].sampling_params
            for rid in forward_op.request_ids
        ]

    def _gather_grammar_state(self, forward_op) -> GrammarStepInputs | None:
        """Build ``GrammarStepInputs`` for the current batch, or ``None``.

        Returns ``None`` when no request in this batch has a grammar — the
        model_executor short-circuits then. Otherwise carries the grammars
        list + per-EXTEND-slot ``advance_mask`` (False on intermediate
        chunked-prefill chunks, since the sampled token is discarded by
        post_process and must not advance the matcher).
        """
        rid_to_state = self.output_processor.rid_to_state
        grammars = [rid_to_state[rid].grammar for rid in forward_op.request_ids]
        if not any(grammars):
            return None

        advance_mask = None
        num_extends = forward_op.num_extends()
        if num_extends > 0:
            bs = len(forward_op.request_ids)
            extend_prefix_lens = forward_op.extend_prefix_lens
            extend_input_lengths = forward_op.input_lengths[:num_extends]
            advance_mask = [True] * bs
            for i in range(num_extends):
                rid = forward_op.request_ids[i]
                # This chunk completes prefill iff it processes the final
                # token of the prompt; intermediate chunks don't.
                advance_mask[i] = (
                    extend_prefix_lens[i] + extend_input_lengths[i]
                    >= rid_to_state[rid].input_length
                )

        return GrammarStepInputs(grammars=grammars, advance_mask=advance_mask)

    def event_loop_overlap(self):
        """
        Overlapping scheduler loop: post-process the previous step's results
        while the current step's forward pass is in flight.
        """
        # EPD invariant: the async embedding drain (EpdPrefillAdmission.drain)
        # that admits EPD requests runs ONLY in event_loop(), never here. A
        # prefill node that receives encode embeddings must therefore run the
        # non-overlap loop -- should_use_overlap_schedule enforces this by forcing
        # prefill -> non-overlap. Assert it rather than trusting that external
        # coupling: if a prefill ever reached this loop, every EPD request would
        # stage into the admission controller and hang forever with no drain.
        assert self.epd_admission is None, (
            "EPD prefill must run the non-overlap event_loop(); the embedding "
            "drain is not wired into event_loop_overlap()"
        )
        prev_results: ModelExecutionResult = None
        prev_forward_op = None

        while not self._shutdown_complete():
            # Order this iter's default-stream writes (prefix_cache page-table
            # writes) after the prev iter's forward on execution_stream that
            # reads the same tensor. Non-blocking on host.
            torch.cuda.default_stream().wait_stream(
                self.model_executor.execution_stream
            )
            self._process_new_requests()
            self._commit_cache_results()
            if self._pause.forward_blocked:
                # Freeze: commit any in-flight (overlapped) step — a forward
                # already on the GPU can't be un-launched — then idle.
                self._paused_idle_step(prev_forward_op, prev_results)
                prev_results = None
                prev_forward_op = None
                continue
            execution_plan = self.scheduler.next_execution_plan()
            self._publish_scheduler_kv_events()

            cache_zero_event = self.model_executor.zero_cache_pages(
                execution_plan.pages_to_zero
            )
            self._submit_cache_ops(execution_plan)

            forward_op = self._get_forward_op(execution_plan)
            stats = self._get_scheduler_stats()
            self._observe_load_snapshot(stats)
            num_iter_tokens = (
                sum(forward_op.input_lengths) if forward_op is not None else 0
            )

            grammar_inputs = None
            if forward_op is not None:
                # Gather both sampling params and grammar state BEFORE the
                # prev_results commit below — that commit can finish requests
                # and pop them from output_processor.rid_to_state, which would
                # KeyError when we look up rids that are still in the current
                # forward_op.
                sampling_params_list = self._gather_sampling_params(forward_op)
                grammar_inputs = self._gather_grammar_state(forward_op)

            # DP sync: all ranks must participate even when idle.
            dp_metadata = None
            if self.has_dp:
                dp_metadata = self._dp_sync_and_check(forward_op)
                if dp_metadata.need_idle_forward:
                    if prev_results is not None:
                        request_changes = self._commit_forward_results(
                            prev_forward_op, prev_results
                        )
                        advance_forward(self.scheduler, request_changes)
                        self._publish_scheduler_kv_events()
                        prev_results = None
                        prev_forward_op = None
                    self.model_executor.execute_idle_forward(
                        dp_metadata.global_num_tokens,
                        dp_metadata.global_batch_size,
                        dp_metadata.all_decode_or_idle,
                    )
                    self._record_scheduler_iteration_metrics(stats, num_iter_tokens)
                    continue

            # ---- dispatch current forward first (async GPU launch) ----
            # Issue curr's forward before committing prev so the GPU runs curr
            # while the CPU syncs/post-processes prev. Committing prev first
            # would block the CPU on prev's copy_event and leave the GPU idle
            # until dispatch — visible as a gap between forwards in the trace.
            #
            # Eager grammar exception: setup_grammar_step reads each matcher's
            # current state to fill the bitmask. Under the overlap pattern the
            # matcher hasn't been advanced yet by prev's accept_token (commit
            # below), so the fill would use a one-step-stale state and let the
            # model sample a token the matcher then rejects. Capturable
            # grammar dodges this with an in-graph hostfunc that advances
            # before fill; eager has no equivalent, so we commit prev first
            # whenever this batch carries grammars. Costs the dispatch/commit
            # overlap for grammar batches but is correct.
            request_changes = []
            curr_has_grammar = grammar_inputs is not None
            eager_grammar_needs_advance = (
                curr_has_grammar
                and prev_results is not None
                and self.model_executor.eager_grammar_buffers is not None
            )
            if eager_grammar_needs_advance:
                request_changes.extend(
                    self._commit_forward_results(prev_forward_op, prev_results)
                )
                prev_results = None
                prev_forward_op = None

            curr_results = None
            if forward_op is not None:
                self._mark_stats_scheduled(forward_op)
                curr_results, _ = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    execution_plan,
                    dp_metadata=dp_metadata,
                    stats=stats,
                    grammar_inputs=grammar_inputs,
                    cache_zero_event=cache_zero_event,
                )

            # ---- post-process previous step (overlapped with current forward) ----
            if prev_results is not None:
                request_changes.extend(
                    self._commit_forward_results(prev_forward_op, prev_results)
                )

            # ---- collect KV transfer events ----
            if self.kv_transfer is not None:
                kv_transfer_events = self.kv_transfer.generate_events()
                request_changes.extend(
                    self._process_kv_transfer_events(kv_transfer_events)
                )

            if request_changes:
                advance_forward(self.scheduler, request_changes)
                self._publish_scheduler_kv_events()

            # Resolve a deferred abort/wait pause reply once in-flight work drains.
            self._pause.maybe_finish_drain(self.scheduler)

            self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

            prev_results = curr_results
            prev_forward_op = forward_op

    def close(self) -> None:
        self.load_snapshot_publisher.close()
        # Best-effort: tell an attached SMG frontend this engine is going away
        # (msgpack mode only; the pickle sender has no such helper) so the
        # worker is marked dead instead of staying healthy-idle.
        send_engine_dead = getattr(self.send_to_tokenizer, "send_engine_dead", None)
        if callable(send_engine_dead):
            send_engine_dead()
        close_transfer = getattr(self.kv_transfer, "close", None)
        if callable(close_transfer):
            close_transfer()


def run_event_loop(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    mapping = server_args.mapping
    gpu_id = mapping.rank % mapping.nprocs_per_node + server_args.base_gpu_id
    attn_tp_rank = mapping.attn.tp_rank
    dp_rank = mapping.attn.dp_rank
    global_rank = mapping.rank

    setproctitle.setproctitle(f"tokenspeed::scheduler_{dp_rank}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()
    register_usr_signal()

    prefix = f" ATTN TP RANK {attn_tp_rank}"
    configure_logger(server_args, prefix=prefix)

    event_loop = None
    shutdown_event = threading.Event()
    previous_sigterm_handler = None
    try:
        if server_args.disaggregation_mode == "encode":
            # The encode role is LM-free; run the lightweight vision-tower loop
            # instead of building the full EventLoop (KV/LM scheduler).
            from tokenspeed.runtime.epd.encode_loop import (
                run_encode_loop,
            )

            run_encode_loop(server_args, port_args, pipe_writer, gpu_id, global_rank)
            return

        # Convert SIGTERM into a loop-owned stop request so the current
        # scheduler iteration and ordinary runtime cleanup can finish.
        if threading.current_thread() is threading.main_thread():
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(
                signal.SIGTERM,
                lambda _signum, _frame: shutdown_event.set(),
            )

        event_loop = EventLoop(
            server_args,
            port_args,
            gpu_id,
            attn_tp_rank,
            dp_rank,
            global_rank,
            shutdown_event,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": event_loop.max_total_num_tokens,
                "max_req_input_len": event_loop.max_req_input_len,
                "max_single_request_tokens": event_loop.max_single_request_tokens,
                "max_num_seqs": server_args.max_num_seqs,
                "chunked_prefill_size": server_args.chunked_prefill_size,
                "max_model_len": event_loop.max_model_len,
                "multimodal_encoder_dtype": event_loop.multimodal_encoder_dtype,
                "cache_storage": getattr(event_loop, "cache_storage", None),
            }
        )

        if event_loop.has_dp:
            # All DP schedulers must finish initialization before any rank enters
            # the loop and starts the first DP metadata collective.
            dist.barrier(group=event_loop.world_cpu_group)

        if event_loop.use_overlap_schedule:
            event_loop.event_loop_overlap()
        else:
            event_loop.event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGUSR1)
    finally:
        if event_loop is not None:
            try:
                event_loop.close()
            except Exception:
                logger.error(
                    "Scheduler transport shutdown failed: %s",
                    get_exception_traceback(),
                )
                parent_process.send_signal(signal.SIGUSR1)
        if (
            previous_sigterm_handler is not None
            and threading.current_thread() is threading.main_thread()
        ):
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
