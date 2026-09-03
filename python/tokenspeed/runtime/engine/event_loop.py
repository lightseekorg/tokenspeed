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
from collections import deque

import psutil
import setproctitle
import torch
import torch.distributed as dist
import zmq
from tokenspeed_scheduler import Scheduler

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.batch_log import BatchLogger
from tokenspeed.runtime.engine.cache_hooks import L2CacheHooks
from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser
from tokenspeed.runtime.engine.io_struct import IpcReceiver, IpcSender, NullSender
from tokenspeed.runtime.engine.load_snapshot import create_load_reporter
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController, PauseHooks
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    advance_scheduler,
    make_config,
    resolve_dspark_prefix_replay_tokens,
    scheduler_cache_group_pages,
    should_use_overlap_schedule,
)
from tokenspeed.runtime.epd.prefill_hooks import EpdPrefillHooks
from tokenspeed.runtime.execution.device import (
    DeviceRole,
    build_device_side,
    maybe_control_plane_guard,
)
from tokenspeed.runtime.execution.distributed_initializer import (
    DistributedConfig,
    DistributedInitializer,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    PendingExecution,
    PlannedForward,
)
from tokenspeed.runtime.grammar.capturable_grammar import GrammarStepInputs
from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.multimodal.inputs import multimodal_context_for_forward
from tokenspeed.runtime.pd.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    NullEventPublisher,
    drain_scheduler_kv_events,
    scheduler_kv_events_to_wire_events,
)
from tokenspeed.runtime.pd.transfer_hooks import PdTransferHooks
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import (
    configure_logger,
    get_colorful_logger,
    get_zmq_socket,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.process import register_usr_signal
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


def maybe_warm_cupti_for_graph_capture() -> None:
    """Preload CUPTI before any CUDA graph is captured. Opt-in.

    The warm-up guards a hazard reported on older stacks: a profiler that
    first attaches AFTER capture invalidates the captured graphs -- every
    later replay dies with cudaErrorLaunchFailure -- which would forbid
    runtime ``/start_profile`` on graph-mode servers. One empty profiler
    session loads CUPTI ahead of every capture.

    It is off by default because it defeats its own purpose: the empty session
    leaves activity collection dead for the life of the process, so every
    later ``/start_profile`` returns a trace with ``cpu_op`` entries and zero
    ``"cat": "kernel"`` events, on every rank, in eager and graph mode alike.
    That was documented for ROCm's roctracer and is CUPTI's behaviour too --
    on CUDA 13.0 / torch 2.13, 2xGB300 TP8, a profiled decode yields 0 kernel
    events with the warm-up and 933k across 340 graph replays without it. The
    hazard did not reproduce there either: all 340 replays ran after the
    attach with no launch failure.

    ``TOKENSPEED_CUPTI_GRAPH_WARMUP=1`` restores it on a stack that still
    needs it, at the cost of GPU profiling. AMD ignores that request: on
    roctracer the warm-up has the same cost and no upside.
    """
    from tokenspeed_kernel.platform import current_platform

    if not envs.TOKENSPEED_CUPTI_GRAPH_WARMUP.get():
        return
    if not torch.cuda.is_available() or current_platform().is_amd:
        return

    from torch.profiler._utils import _init_for_cuda_graphs

    _init_for_cuda_graphs()


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

        self.use_overlap_schedule = should_use_overlap_schedule(
            disable_overlap_schedule=server_args.disable_overlap_schedule,
            disaggregation_mode=server_args.disaggregation_mode,
        )
        self.overlap_schedule_depth = int(self.use_overlap_schedule)
        # In-flight depth of the unified event loop: how many dispatched
        # forwards may await commit at once. 0 = commit in the same iteration
        # (classic non-overlap); 1 = the overlap schedule (CPU post-processes
        # step N-1 while the GPU runs step N); pp_size = the prefill chunk
        # pipeline. Distinct from overlap_schedule_depth: that one sizes
        # decode KV reservations in the C++ scheduler and recipes, this one
        # only queues commits.
        if server_args.mapping.has_pp:
            self.in_flight_depth = server_args.mapping.pp_size
        else:
            self.in_flight_depth = int(self.use_overlap_schedule)

        decode_input_tokens = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm is not None
            else 1
        )
        mapping = server_args.mapping
        # The C++ scheduler's req_pool_idx range is rank-local and 1-based:
        # real rows are 1..max_batch_size, row 0 is reserved.
        per_rank_max_batch = server_args.max_num_seqs // max(mapping.attn.dp_size, 1)
        # The entire device side is built in here: model runners, attention
        # backends, KV pools and the executor are locals of the builder and
        # never come back out. ``device`` is a local too — only ``.handle``
        # outlives this constructor, so the running loop can neither name a
        # device object nor call a startup hook.
        device = build_device_side(
            server_args=server_args,
            model_config=self.model_config,
            draft_model_config=draft_model_config,
            gpu_id=gpu_id,
            global_rank=global_rank,
            attn_tp_rank=attn_tp_rank,
            min_per_gpu_mem=min_per_gpu_mem,
            overlap_schedule_depth=self.overlap_schedule_depth,
            decode_input_tokens=decode_input_tokens,
            max_batch_size=per_rank_max_batch,
        )
        self._device = device.handle
        specs = device.specs
        self.multimodal_encoder_dtype = specs.multimodal_encoder_dtype
        self.cache_storage = specs.cache_storage
        self._scheduler_cache_geometry = specs.cache_geometry
        geometry = self._scheduler_cache_geometry
        # The contract is the one source of admitted capacity.
        self.max_total_num_tokens = geometry.token_capacity
        # Planning reads this every round; keep the value, not the handle.
        self._uses_eager_grammar = specs.uses_eager_grammar
        cache_groups = specs.cache_groups
        # The builder may have lowered this to the cache-group checkpoint grain.
        max_scheduled_tokens = server_args.chunked_prefill_size

        self.attn_tp_size = server_args.attn_tp_size or mapping.attn.tp_size
        self.world_size = server_args.world_size or mapping.world_size
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_cpu_group = pg_manager.get_process_group(
            "gloo", server_args.mapping.attn.tp_group
        )
        self.dp_rank = dp_rank
        self.dp_size = mapping.attn.dp_size
        self.has_dp = mapping.has_attn_dp
        if self.has_dp:
            self.world_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.world_group
            )
            self._dp_local_info = torch.zeros(1, 3, dtype=torch.int32)
            self._dp_global_info = torch.zeros(mapping.world_size, 3, dtype=torch.int32)
        num_host_pages = specs.num_host_pages
        # L2 cache-op accounting + rank-synced completion tracking (see
        # cache_hooks.py); a no-op shell when kvstore is disabled. The hooks
        # get the handle, not the L2 executor: polling goes through it.
        self._cache_hooks = L2CacheHooks(
            self._device if server_args.enable_kvstore else None,
            speculative_algorithm=server_args.speculative_algorithm,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=self.attn_tp_size,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            global_rank=global_rank,
        )

        self._kv_events_enabled = (
            EventPublisherFactory.is_enabled(server_args.kv_events_config)
            and attn_tp_rank == 0
        )

        # Encode nodes never build an EventLoop (they run the LM-free encode
        # loop), so here "disaggregation is on" means the cache-transfer PD
        # protocol: this engine is a P or D role.
        pd_enabled = server_args.disaggregation_mode != "null"
        if pd_enabled:
            if not specs.supports_disaggregation:
                raise RuntimeError(
                    "PD disaggregation requires a unified cache contract"
                )
            unsupported = []
            if server_args.enable_mixed_batch:
                unsupported.append("mixed prefill/decode batches")
            if (
                server_args.speculative_algorithm is not None
                and server_args.disaggregation_layerwise_interval > 0
                and not specs.supports_pd_layerwise_finalization
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
                    "Cache-transfer PD currently does not support: "
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
        logger.info(
            "Scheduler config: prefix_granularity=%s num_device_pages=%s "
            "max_scheduled_tokens=%s decode_input_tokens=%s "
            "overlap_schedule_depth=%s disable_l2_cache=%s "
            "enable_l3_storage=%s "
            "max_batch_size=%s (global max_num_seqs=%s, dp_size=%s) "
            "disable_prefix_cache=%s prefix_replay_tokens=%s "
            "cache_groups=%s",
            scheduler_cfg.prefix_granularity,
            scheduler_cfg.num_device_pages,
            scheduler_cfg.max_scheduled_tokens,
            scheduler_cfg.decode_input_tokens,
            scheduler_cfg.overlap_schedule_depth,
            scheduler_cfg.disable_l2_cache,
            scheduler_cfg.enable_l3_storage,
            scheduler_cfg.max_batch_size,
            server_args.max_num_seqs,
            self.dp_size,
            scheduler_cfg.disable_prefix_cache,
            scheduler_cfg.prefix_replay_tokens,
            [group.group_id for group in cache_groups],
        )
        self.scheduler = Scheduler(scheduler_cfg)
        # Per-round batch logging lives on the control plane: it reports
        # scheduler quantities (queue depth, page usage) that the loop already
        # samples, and its counters stay on this thread.
        self._batch_logger = BatchLogger(
            enabled=global_rank == 0,
            decode_log_interval=server_args.decode_log_interval,
            # Usable pages, the same total the load snapshot and the
            # Prometheus gauge publish, so the three never disagree.
            num_total_pages=geometry.num_usable_pages,
            spec_num_steps=specs.spec_num_steps,
            spec_num_tokens=specs.spec_num_tokens,
            cache_state_group_ids=specs.cache_state_group_ids,
            cache_group_pages=scheduler_cache_group_pages(self.scheduler),
        )
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
                "The cache cannot admit one request with the configured "
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
        if attn_tp_rank == 0:
            self.kv_event_publisher = EventPublisherFactory.create(
                server_args.kv_events_config,
                attn_dp_rank=dp_rank,
            )
        else:
            self.kv_event_publisher = NullEventPublisher(attn_dp_rank=dp_rank)

        self._init_interprocess_comm()
        self._init_load_reporter()

        # Pause/resume control state. Shared with the request handler, which
        # drives the control-request side; the event loop reads the gate.
        # PauseHooks is the loop-side integration (see pause.py) — the normal
        # scheduling paths below only carry single-line hooks into it.
        self._pause = PauseController(self.send_to_tokenizer)
        self._pause_hooks = PauseHooks(self, self._pause, self._device)

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
            reset_caches_fn=self._pause_hooks.reset_caches_for_release,
            kv_repair_fn=self._pause_hooks.kv_repair_after_wake,
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
            clear_cache_fn=self._clear_cache,
            architectures=self.model_config.hf_config.architectures,
            pause_controller=self._pause,
            memory_controller=self._memory,
            device=self._device,
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
            defer_to_device=self._device.run_multimodal_work,
        )
        # The peer's control face only — bootstrap register/abort, event
        # polling. Its execution face is inside the handle.
        self.kv_transfer = device.transfer
        epd_admission = None
        if server_args.disaggregation_mode != "null":
            # EPD: a multimodal prefill node is also the encode->prefill
            # embedding SINK (independent of kv_transfer, its P->D KV source)
            # -- it receives each image's embedding from encode workers over
            # Mooncake so the prefill skips the vision tower. The admission
            # controller owns the receive jobs, the rank-synced admission
            # drain, and the optional NCCL row-shard reassembly; None for
            # decode/encode/text-only nodes. EpdPrefillHooks is the loop-side
            # integration (see prefill_hooks.py).
            from tokenspeed.runtime.epd.prefill_admission import (
                make_epd_prefill_admission,
            )

            epd_admission = make_epd_prefill_admission(
                server_args,
                global_rank,
                model_config=self.model_config,
                encoder_model_facts=device.encoder_model_facts,
                mapping=mapping,
                attn_tp_rank=self.attn_tp_rank,
                attn_tp_size=self.attn_tp_size,
                attn_tp_cpu_group=self.attn_tp_cpu_group,
                pg_manager=pg_manager,
                run_device_work=self._device.run_multimodal_work,
            )
        self._epd_hooks = EpdPrefillHooks(self, epd_admission)
        # PD transfer-event integration (see pd/transfer_hooks.py); a no-op
        # when PD is disabled.
        self._pd_hooks = PdTransferHooks(self, self._device)

    def _publish_scheduler_kv_events(self) -> None:
        """Drain the KV events the C++ scheduler accumulated and publish them.

        Drain semantics: events queue up inside the scheduler across any
        number of mutations (advance / next_execution_plan), so one call at
        the event-loop tail — its only call site — publishes everything the
        round produced, in order, as a single batch.
        """
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

    def _submit_scheduler_requests(self, specs) -> None:
        self._register_l3_storage_hits(specs)
        self.scheduler.submit_requests(specs)

    def _clear_cache(self) -> bool:
        cleared = self.scheduler.clear_cache()
        if cleared:
            self._device.rotate_l3_namespace()
        return cleared

    def _register_l3_storage_hits(self, specs) -> None:
        """Tell the scheduler which prefix pages already live in L3.

        Cross-instance reuse cannot see Mooncake objects through the Host
        index. Probe them with the same content hashes the scheduler will
        use, then register only keys every TP rank agrees exist.
        """

        if not specs:
            return
        hashes = []
        seen: set[str] = set()
        for spec in specs:
            for content_hash in self.scheduler.prefix_hashes_for_tokens(
                list(spec.tokens)
            ):
                if content_hash in seen:
                    continue
                seen.add(content_hash)
                hashes.append(content_hash)
        if not hashes:
            return
        group_ids, content_hashes, page_offsets = self.scheduler.expand_prefix_keys(
            hashes
        )
        pages = [
            (int(group_id), 0, content_hash, int(page_offset))
            for group_id, content_hash, page_offset in zip(
                group_ids, content_hashes, page_offsets
            )
        ]
        exists = self._device.query_l3_storage(pages)
        if exists is None:
            return
        if len(exists) != len(group_ids):
            raise RuntimeError("L3 existence result is not aligned with cache keys")
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None and exists:
            flags = torch.tensor(
                [1 if present else 0 for present in exists], dtype=torch.int32
            )
            dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=self.attn_tp_cpu_group)
            exists = [bool(flag) for flag in flags.tolist()]
        hit_groups = []
        hit_hashes = []
        hit_offsets = []
        miss_groups = []
        miss_hashes = []
        miss_offsets = []
        for group_id, content_hash, page_offset, present in zip(
            group_ids, content_hashes, page_offsets, exists
        ):
            target = (
                (hit_groups, hit_hashes, hit_offsets)
                if present
                else (miss_groups, miss_hashes, miss_offsets)
            )
            target[0].append(int(group_id))
            target[1].append(content_hash)
            target[2].append(int(page_offset))
        if miss_groups:
            self.scheduler.unregister_storage_keys(
                miss_groups, miss_hashes, miss_offsets
            )
        if hit_groups:
            self.scheduler.register_storage_keys(hit_groups, hit_hashes, hit_offsets)

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
        # Chunk-pipeline: request I/O is owned by GLOBAL rank 0 only —
        # every stage's tp_rank-0 would otherwise try to open the one
        # frontend socket pair. recv_reqs broadcasts over the world group.
        owns_request_io = (
            self.server_args.mapping.rank == 0
            if self.server_args.mapping.has_pp
            else self.attn_tp_rank == 0
        )
        if owns_request_io:
            if self.server_args.zmq_msgpack:
                # SMG drives the scheduler directly: it binds the sockets and
                # this engine connects in over the msgpack wire; the handshake
                # (engine identity, ready response) lives in zmq_msgpack.
                from tokenspeed.runtime.engine import zmq_msgpack

                self.recv_from_tokenizer, self.send_to_tokenizer = (
                    zmq_msgpack.connect_msgpack_engine_for_loop(context, self)
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
            self.send_to_tokenizer = NullSender()

    def _init_load_reporter(self) -> None:
        reports_load = self.attn_tp_rank == 0
        self.load_reporter = create_load_reporter(
            enabled=reports_load,
            # Bound only in direct-ZMQ mode, and only where it is used: other
            # ranks send through a NullSender that has no such setter.
            direct_setter=(
                self.send_to_tokenizer.set_load_snapshot
                if reports_load and self.server_args.zmq_msgpack
                else None
            ),
            endpoint=self.port_args.metrics_ipc_name,
            dp_rank=self.dp_rank,
            heartbeat_interval=self.server_args.load_watch_interval,
            num_total_pages=self._scheduler_cache_geometry.num_usable_pages,
            sample_stats=self._get_scheduler_stats,
        )

    # ------------------------------------------------------------------
    # Shared step helpers
    # ------------------------------------------------------------------

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
        # Pause-state snapshot for withhold_admissions below: it must be
        # taken before process_requests, which may flip the state mid-batch.
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

        self._pause_hooks.apply_transitions(grammar_manager)

        # Partition new requests by grammar readiness. Compile-bound requests
        # are queued in GrammarManager and admitted in a later iteration when
        # their futures resolve (get_ready_grammar_requests below).
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

            if self.kv_transfer is not None and bootstrap is None:
                raise ValueError(
                    "Cache-transfer PD request is missing bootstrap information"
                )
            if self._device.role is DeviceRole.PD_DECODE:
                # The prompt was computed on the prefill node.
                state.computed_length = state.input_length
            self.output_processor.register(spec.request_id, state)
            # EPD prefill: an encode-routed request is staged OUT of the
            # scheduler until its embeddings arrive; its P->D sender
            # registration and submission are both deferred to the EPD
            # admission drain (see EpdPrefillHooks.try_stage for why).
            if self._epd_hooks.try_stage(spec, state, bootstrap):
                continue
            if self.kv_transfer is not None:
                self.kv_transfer.register(spec.request_id, bootstrap)
            admitted_specs.append(spec)

        if self._pause_hooks.withhold_admissions(admitted_specs, pause_blocked_before):
            return

        if admitted_specs:
            self._submit_scheduler_requests(admitted_specs)

    @nvtx_range("loop:commit", color="rapids")
    def _pp_broadcast_output_tokens(self, forward_op, results) -> None:
        """Align sampled tokens across pipeline stages before commit.

        Only the last stage samples; the other stages produced placeholder
        outputs. Every rank's C++ scheduler expects the REAL bootstrap
        payload in the final chunk's ExtendResult — the sampled first token
        (read back as LastToken) and the drafter candidates its remote
        decode will carry — so the last stage broadcasts (output_tokens,
        output_lengths, next_input_ids) over the PP gloo group and the
        others adopt them. Runs on the commit path (queue head), off the
        dispatch hot path.
        """
        mapping = self.server_args.mapping
        if not mapping.has_pp:
            return
        group = pg_manager.get_process_group("gloo", mapping.pp_group)
        src_global_rank = mapping.pp_group[-1]
        # Host tensors already: the result was synced before commit, and the
        # executor issues every output as a D2H copy.
        payload = [None]
        if mapping.is_last_pp_rank:
            payload = [
                (
                    results.output_tokens,
                    results.output_lengths,
                    results.next_input_ids,
                )
            ]
        dist.broadcast_object_list(payload, src=src_global_rank, group=group)
        if not mapping.is_last_pp_rank:
            tokens, lengths, next_ids = payload[0]
            results.output_tokens = tokens
            results.output_lengths = lengths
            results.next_input_ids = next_ids

    def _commit_forward_results(
        self,
        forward_op,
        pending: PendingExecution,
    ):
        # Where a dispatched round waits for the GPU: join the forward
        # thread's future (launches done) + the copy event (D2H landed).
        # Everything below reads host tensors.
        with nvtx_range("commit:sync", color="red"):
            results = pending.result()
        self.request_handler.forward_ct += 1
        forward_mode = ForwardMode.from_num_extends(
            forward_op.num_extends(),
            len(forward_op.request_ids),
        )
        self.request_handler._profile_batch_predicate(forward_mode)
        self._pp_broadcast_output_tokens(forward_op, results)

        is_prefill_instance = self._device.role is DeviceRole.PD_PREFILL
        request_changes = self.output_processor.post_process_forward_op(
            forward_op,
            results,
            is_prefill_instance=is_prefill_instance,
        )

        # Fold committed tokens into the decode throughput window (host-side
        # reads of the already-synced result; no GPU sync).
        if forward_op.num_extends() <= 0:
            bs = len(forward_op.request_ids)
            self._batch_logger.record_decode(results, bs)

        return request_changes

    def _get_forward_op(self, execution_plan):
        forward_ops = execution_plan.forward
        if len(forward_ops) == 0 or len(forward_ops[0].request_ids) == 0:
            return None
        return forward_ops[0]

    def _dp_sync_and_check(self, forward_op) -> DpForwardMetadata:
        """Synchronize DP ranks with CPU-only metadata.

        All ranks call this before GPU forward work. The gathered metadata is
        used for eager token-aware collectives and for choosing a common padded
        CUDA graph shape during decode.
        """
        # Whether forward_op will enter the model forward path. The
        # ForwardBatch now carries model work ONLY: the transfer peer's
        # remote prefills and remote decodes ride their own plan streams,
        # which ``execute`` submits on every round — a rank that reports
        # idle over one still sends it. So the answer is simply whether the
        # batch has tokens.
        executes_model_forward = (
            forward_op is not None and sum(forward_op.input_lengths) > 0
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

    def _num_running(self) -> int:
        return len(self.output_processor.rid_to_state)

    def _get_scheduler_stats(self):
        available = self.scheduler.available_kv_pages()
        active = self.scheduler.active_kv_pages()
        return {
            "num_active_pages": active,
            "num_cached_pages": (
                self._scheduler_cache_geometry.num_usable_pages - available
            ),
            "num_queue_reqs": self.scheduler.waiting_size(),
        }

    def _record_scheduler_iteration_metrics(
        self, stats: dict, num_iteration_tokens: int
    ) -> None:
        self.metrics.record_scheduler_iteration(
            running=self._num_running(),
            waiting=stats["num_queue_reqs"],
            num_active_pages=stats["num_active_pages"],
            num_total_pages=self._scheduler_cache_geometry.num_usable_pages,
            num_iteration_tokens=num_iteration_tokens,
        )

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def _shutdown_complete(self) -> bool:
        return self.shutdown_event.is_set()

    def _drain_in_flight(self, in_flight) -> list:
        """Commit every queued forward, oldest first; return their changes."""
        request_changes = []
        while in_flight:
            fo, res = in_flight.popleft()
            request_changes.extend(self._commit_forward_results(fo, res))
        return request_changes

    def _dispatch_depends_on_pending_commit(self, forward_op, grammar_inputs) -> bool:
        """Whether the upcoming dispatch reads state that only a pending
        commit produces, so the in-flight queue must drain first.

        The single registry of overlap-breaking dependencies — add new rules
        here, not in ``event_loop``. One rule today:

        - Eager grammar: ``setup_grammar_step`` reads each matcher's current
          state to fill the bitmask, and the matcher only advances at the
          pending step's commit (``accept_token``). Capturable grammar dodges
          this with an in-graph hostfunc; eager has no equivalent, so trade
          the overlap away for grammar batches.

        Prefer removing a dependency over registering one: the P-side remote
        decode was a rule here until the scheduler learned to hold it, and
        draining for it emptied the PP chunk pipeline on every finished
        prompt.
        """
        return grammar_inputs is not None and self._uses_eager_grammar

    def event_loop(self):
        """The one scheduler loop, parameterized by in-flight depth.

        ``in_flight_depth`` is how many dispatched forwards may await commit:

        - 0: commit in the same iteration (classic non-overlap behavior).
        - 1: dispatch the current forward before committing the previous one,
          so the CPU post-processes step N-1 while the GPU runs step N (the
          overlap schedule).
        - pp_size: the prefill chunk pipeline — consecutive chunks occupy
          different pipeline stages; committing the queue head (join the
          forward thread, then its copy event) is the backpressure.

        Correctness never depends on the depth: any dispatch whose inputs
        depend on a pending commit's side effects drains the queue first
        (``_dispatch_depends_on_pending_commit`` is the single registry of
        those rules), and rounds that run no real forward (pause/freeze,
        DP idle) drain it fully.

        Scheduler feedback is only ever an explicit ``advance_scheduler`` call
        in this loop body — helpers return events, never advance. Two calls:
        cache-op completions at the head of the round (so this round's plan
        sees them) and forward results at the tail (they only exist after
        dispatch); everything else funnels into ``request_changes``.
        """
        in_flight: deque = deque()
        depth = self.in_flight_depth
        with maybe_control_plane_guard():
            while not self._shutdown_complete():
                self._process_new_requests()

                # EPD prefill: admit requests whose async embedding receives completed
                # this cycle (rank-synced). Fixed position right after
                # _process_new_requests so the drain's TP collective ordering is
                # rank-identical every cycle. A no-op without an EPD admission
                # controller (every non-EPD deployment).
                self._epd_hooks.drain_ready_embeddings()
                cache_events = self._cache_hooks.poll_ready_events()
                if cache_events:
                    # Advanced at the HEAD of the round (not funneled into the
                    # tail advance) so completed cache ops are visible to this
                    # round's next_execution_plan — deferring them would delay
                    # cache-gated admissions by a full round.
                    advance_scheduler(self.scheduler, cache_events)

                # Every path in this round appends its committed results here;
                # they feed back into the scheduler through the single
                # advance_scheduler call at the tail.
                request_changes = []
                forward_op = None
                # An idle round (freeze or DP idle) runs no dispatch and no
                # kv-transfer event poll.
                idle_round = False

                if self._pause.forward_blocked:
                    # Freeze: dispatched forwards can't be un-launched; commit them
                    # all before idling.
                    request_changes.extend(self._drain_in_flight(in_flight))
                    self._pause_hooks.paused_idle_step()
                    idle_round = True
                else:
                    execution_plan = self.scheduler.next_execution_plan()
                    self._cache_hooks.count_plan_ops(execution_plan)

                    forward_op = self._get_forward_op(execution_plan)
                    stats = self._get_scheduler_stats()
                    self.load_reporter.observe(stats, self._num_running())
                    num_iter_tokens = (
                        sum(forward_op.input_lengths) if forward_op is not None else 0
                    )
                    # Record once per iteration, from the same pre-dispatch
                    # snapshot as ``stats`` (the running gauge counts requests
                    # admitted but not yet committed-finished this round —
                    # consistent with waiting/pages).
                    self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

                    # DP sync: all ranks must participate even when idle. Checked
                    # right after forward_op is derived so an idle round commits
                    # pending steps and skips the per-batch work below (the
                    # gathers are local and read-only, so ordering them after the
                    # collective is rank-safe).
                    dp_metadata = None
                    if self.has_dp:
                        dp_metadata = self._dp_sync_and_check(forward_op)
                        if dp_metadata.need_idle_forward:
                            request_changes.extend(self._drain_in_flight(in_flight))
                            idle_round = True

                    planned = None
                    if not idle_round and forward_op is not None:
                        # Gather sampling params and grammar state BEFORE any
                        # pending commit below — a commit can finish requests and
                        # pop them from output_processor.rid_to_state, which would
                        # KeyError on rids still present in the current forward_op.
                        sampling_params_list = self._gather_sampling_params(forward_op)
                        grammar_inputs = self._gather_grammar_state(forward_op)

                        if in_flight and self._dispatch_depends_on_pending_commit(
                            forward_op, grammar_inputs
                        ):
                            request_changes.extend(self._drain_in_flight(in_flight))

                        self._mark_stats_scheduled(forward_op)
                        self._batch_logger.log_dispatch(forward_op, stats)
                        planned = PlannedForward(
                            forward_op=forward_op,
                            sampling_params_list=sampling_params_list,
                            dp_metadata=dp_metadata,
                            grammar_inputs=grammar_inputs,
                            multimodal_context=(
                                multimodal_context_for_forward(
                                    forward_op, self.output_processor.rid_to_state
                                )
                                if self.model_config.is_multimodal_active
                                else None
                            ),
                        )
                        # EPD invariant: handshaked items were filled by the
                        # async admission drain before admission; none may
                        # reach the forward un-received. No-op outside EPD.
                        self._epd_hooks.assert_embeddings_received(
                            planned.multimodal_context
                        )

                    # One call per round: the plan's page zeroing and cache
                    # transfers ride the FIFO first, then the batch the role
                    # routes. ``planned`` is None on idle/empty rounds — the
                    # plan hygiene still runs.
                    pending = self._device.execute(execution_plan, planned)
                    if idle_round:
                        self._device.run_idle_forward(dp_metadata)
                    if pending is not None:
                        in_flight.append((forward_op, pending))

                if not idle_round:
                    # Commit from the head once the queue exceeds the depth
                    # (immediately at depth 0; one step behind at depth 1; a full
                    # pipeline behind under PP). A round with no new work drains
                    # fully so results never wait on future traffic.
                    effective_depth = depth if forward_op is not None else 0
                    while len(in_flight) > effective_depth:
                        fo, res = in_flight.popleft()
                        request_changes.extend(self._commit_forward_results(fo, res))

                    request_changes.extend(self._pd_hooks.poll_transfer_events())

                # The forward-result feedback point: everything this round
                # committed reaches the scheduler here, before the next round
                # plans. (Cache-op completions advance at the head instead — see
                # _cache_hooks.poll_ready_events — but through the same advance_scheduler,
                # the only caller of scheduler.advance.)
                if request_changes:
                    advance_scheduler(self.scheduler, request_changes)

                self._publish_scheduler_kv_events()

                if self._pause.forward_blocked:
                    # Frozen rounds take no planning sample of their own; the
                    # idle sleep bounds this to one sample per millisecond.
                    self.load_reporter.sample_and_observe(self._num_running())

                # Resolve a deferred abort/wait pause reply once in-flight work drains.
                self._pause.maybe_finish_drain(self.scheduler)

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
        forward short-circuits then. Otherwise carries the grammars
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

    def close(self) -> None:
        self.load_reporter.close()
        # Best-effort: tell an attached SMG frontend this engine is going away
        # (msgpack mode only; the pickle sender has no such helper) so the
        # worker is marked dead instead of staying healthy-idle.
        send_engine_dead = getattr(self.send_to_tokenizer, "send_engine_dead", None)
        if callable(send_engine_dead):
            send_engine_dead()
        close_transfer = getattr(self.kv_transfer, "close", None)
        if callable(close_transfer):
            close_transfer()
        self._device.shutdown_cache()


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
    # Re-assert the NVSHMEM IB traffic class in every inference process:
    # NVSHMEM reads it from the process environment at bootstrap, and worker
    # processes may be spawned without inheriting the launcher's setting.
    if envs.NVSHMEM_IB_TRAFFIC_CLASS.is_set():
        envs.NVSHMEM_IB_TRAFFIC_CLASS.set(envs.NVSHMEM_IB_TRAFFIC_CLASS.get())
        logger.info("NVSHMEM_IB_TRAFFIC_CLASS=%d", envs.NVSHMEM_IB_TRAFFIC_CLASS.get())
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

        maybe_warm_cupti_for_graph_capture()

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

        event_loop.event_loop()

    except Exception:  # noqa: BLE001 - process boundary; report and signal parent
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGUSR1)
    finally:
        if event_loop is not None:
            try:
                event_loop.close()
            except Exception:  # noqa: BLE001 - best-effort teardown; signal parent
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
