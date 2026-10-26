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

"""The Python half of the Rust engine host (RFC 0002).

``ts-host`` (``tokenspeed-engine/crates/ts-host``) runs one rank: its control
thread schedules, admits requests and emits outputs in Rust, and the
interpreter it embeds runs the data plane. This module is everything that
interpreter runs, plus the launcher's side:

* ``start_rust_host`` starts the binary for one rank; every launch path
  starts its ranks through it.
* ``bootstrap`` builds the rank inside the host: the model config,
  distributed init, ``build_device_side`` and the scheduler. The Rust side
  takes the scheduler and ``HostConfig``.
* ``HostDataPlane`` is what the host runs the data plane through. Its
  forward thread runs each operation the control thread queued (a plan
  becomes the ``PlannedForward`` that ``DeviceHandle`` launches); its control
  thread runs the completion queries itself (a launched forward's landed
  results as plain lists, the cache ops' acknowledgments). The L3 tier runs
  in the host (``ts-l3``), from the ``L3Settings`` bootstrap resolves.

The host covers TP, attention-DP and pipeline ranks and both PD roles (its
own Rust transfer peer, started from ``pd/host_peer.py``'s settings);
``rust_host_refusals`` and the checks in ``bootstrap`` refuse what it does not
serve instead of serving it partially. An EPD encode rank runs the host's
encode mode, whose Python half is ``epd/encode_host.py``.
"""

from __future__ import annotations

import faulthandler
import functools
import os
import signal
import subprocess
import sys
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import Pipe
from multiprocessing.connection import Connection

import msgspec
import torch
import torch.distributed as dist
from tokenspeed_kernel.compile_monitor import compile_stats
from tokenspeed_scheduler import Scheduler

from tokenspeed.runtime.cache.l3.config import L3Settings
from tokenspeed.runtime.cache.l3.factory import l3_settings
from tokenspeed.runtime.distributed.nccl_suspend import nccl_suspend_supported
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.gc_hooks import (
    FULL_COLLECTION_INTERVAL,
    GcHooks,
    freeze_startup_heap,
)
from tokenspeed.runtime.engine.logprobs import LogprobServability
from tokenspeed.runtime.engine.memory_occupation import memory_regions
from tokenspeed.runtime.engine.profiler import Profiler
from tokenspeed.runtime.engine.rank_setup import (
    init_distributed,
    load_model_config,
    maybe_warm_cupti_for_graph_capture,
    rank_process_title,
    request_length_limits,
    scheduler_config_from_args,
)
from tokenspeed.runtime.engine.scheduler_utils import (
    cache_event_to_payload,
    cache_ops_from_plan,
    engram_context_len,
    resolve_dspark_prefix_replay_tokens,
    should_use_overlap_schedule,
)
from tokenspeed.runtime.epd.encode_host import EncodeBootstrap
from tokenspeed.runtime.epd.prefill_receive import (
    assert_embeddings_received,
    build_epd_receive,
)
from tokenspeed.runtime.execution.device import (
    arm_data_plane_sync_debug,
    build_device_side,
    maybe_control_plane_guard,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    InputLogprobPlan,
    ModelExecutionResult,
    NGramInputs,
    PlannedForward,
    RequestHistorySeeds,
    StopCriteria,
)
from tokenspeed.runtime.grammar.capturable_grammar import (
    GrammarStepInputs,
    resolve_grammar_completion,
)
from tokenspeed.runtime.grammar.host_grammar import HostGrammar, HostGrammarEntries
from tokenspeed.runtime.metrics.config import metrics_enabled
from tokenspeed.runtime.moe.expert_rebalance import ExpertRebalanceSpecs, SlotMoves
from tokenspeed.runtime.multimodal.inputs import (
    MultimodalInputs,
    multimodal_context_for_forward,
)
from tokenspeed.runtime.multimodal.shm_transport import prepare_shm_features
from tokenspeed.runtime.pd.host_peer import build_host_peer
from tokenspeed.runtime.pd.kv_events import KVEventsConfig, kv_events_enabled
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import configure_logger, get_colorful_logger
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer
from tokenspeed.runtime.utils.jit_compile_check import (
    install_jit_compile_check,
    mark_jit_compile_serving,
)
from tokenspeed.runtime.utils.network import get_ip, join_host_port
from tokenspeed.runtime.utils.process import set_parent_death_signal
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.startup_timing import startup_phase
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


def resolve_stop_token_ids(
    sampling_params: SamplingParams, eos_token_ids, tokenizer
) -> frozenset[int]:
    """The tokens that end a request: its own ``stop_token_ids``, the
    model's EOS ids and the tokenizer's EOS and additional stop ids; none
    under ``ignore_eos``.

    The host hands them to the Rust control plane as the model's EOS ids,
    which its finish check and the device's verify cut (``StopCriteria``)
    both read, so the two always end a request on the same token.
    """
    if sampling_params.ignore_eos:
        return frozenset()
    ids = set(sampling_params.stop_token_ids or ())
    ids.update(eos_token_ids or ())
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids.add(eos_id)
    ids.update(getattr(tokenizer, "additional_stop_token_ids", None) or ())
    return frozenset(int(token_id) for token_id in ids)


# ----------------------------------------------------------------------
# The launcher's side
# ----------------------------------------------------------------------


def host_binary() -> str:
    """The ``ts-host`` executable a rank runs: ``TOKENSPEED_HOST_BINARY``, or
    the one installed with the tokenspeed-engine wheel."""
    override = envs.TOKENSPEED_HOST_BINARY.get()
    if override is not None:
        return override
    import tokenspeed_engine

    return str(tokenspeed_engine.host_binary())


def rust_host_refusals(server_args: ServerArgs) -> list[str]:
    """What in ``server_args`` the Rust host cannot serve yet, empty when
    it can serve all of it."""
    refusals = []
    # An EPD encode server starts its ranks itself (``_start_encode_ranks``):
    # they take encode requests, not the engine wire.
    if server_args.disaggregation_mode not in ("null", "prefill", "decode"):
        refusals.append(f"--disaggregation-mode {server_args.disaggregation_mode}")
    if kv_events_enabled(server_args.kv_events_config):
        publisher = KVEventsConfig.from_cli(server_args.kv_events_config).publisher
        if publisher not in (None, "null", "zmq"):
            refusals.append(f"the KV event publisher {publisher!r}")
    return refusals


class HostProcess:
    """A running host, with the ``multiprocessing.Process`` surface the
    headless launcher supervises its ranks through."""

    def __init__(self, popen: subprocess.Popen, sentinel: int) -> None:
        self._popen = popen
        #: Readable once the process exits, which is what
        #: ``multiprocessing.connection.wait`` needs of a sentinel: the read
        #: end of a pipe whose only write end the process holds.
        self.sentinel = sentinel

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def exitcode(self) -> int | None:
        # Negative for a signal, as ``multiprocessing`` reports it.
        return self._popen.poll()

    def is_alive(self) -> bool:
        return self._popen.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self._popen.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self) -> None:
        if self.is_alive():
            self._popen.terminate()

    def kill(self) -> None:
        if self.is_alive():
            self._popen.kill()


def start_rust_host(
    binary: str,
    server_args: ServerArgs,
    port_args: PortArgs,
    start_context: AbstractContextManager,
    metrics_relay_endpoint: str | None,
    parent_death_signal: bool,
) -> tuple[HostProcess, Connection]:
    """Start ``binary`` for the rank ``server_args.mapping.rank`` names; the
    caller has checked ``rust_host_refusals``.

    The host reads its arguments from, and reports readiness on, one end of
    a socket pair; the returned end raises ``EOFError`` if the host dies
    first. Its command line starts with the rank's process title
    (``rank_process_title``; ``tokenspeed::encode_tp{rank}`` for an EPD
    encode rank), which ``ps`` and stale-process cleanup match.

    Args:
        binary: The ``ts-host`` executable.
        server_args: The rank's server arguments.
        port_args: The engine's ports.
        start_context: Entered around starting the process.
        metrics_relay_endpoint: The node's metrics relay
            (``metrics/relay.py``), which runs the output plugins; None
            when no plugin is selected.
        parent_death_signal: Kill the host with SIGKILL when this process
            dies, even when it is SIGKILLed (Linux only; see
            ``utils.process.set_parent_death_signal``). For a node that is
            the only owner of its ranks.

    Returns:
        The process and the launcher's end of the pipe.
    """
    parent, child = Pipe(duplex=True)
    sentinel, held_until_exit = os.pipe()
    preexec_fn = None
    if parent_death_signal:
        preexec_fn = functools.partial(set_parent_death_signal, os.getpid())
    with start_context:
        popen = subprocess.Popen(
            [
                (
                    f"tokenspeed::encode_tp{server_args.mapping.attn.tp_rank}"
                    if server_args.disaggregation_mode == "encode"
                    else rank_process_title(server_args.mapping)
                ),
                "--python",
                sys.executable,
                "--control-fd",
                str(child.fileno()),
            ],
            executable=binary,
            pass_fds=[child.fileno(), held_until_exit],
            preexec_fn=preexec_fn,
        )
    child.close()
    os.close(held_until_exit)
    parent.send((server_args, port_args, metrics_relay_endpoint))
    return HostProcess(popen, sentinel), parent


# ----------------------------------------------------------------------
# The host's side
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GrammarSettings:
    """Grammar-constrained decoding's settings (``--grammar-backend
    xgrammar``), for the host's grammar manager.

    Attributes:
        tokenizer_info: The vocabulary as xgrammar sees it
            (``TokenizerInfo.serialize_json``).
        any_whitespace: JSON schemas allow any whitespace.
        compile_timeout_s: How long a request may wait for its grammar.
        max_retries: Timeouts a grammar is retried after.
    """

    tokenizer_info: str
    any_whitespace: bool
    compile_timeout_s: float
    max_retries: int


def grammar_tokenizer_info(tokenizer, vocab_size: int, eos_token_ids) -> str:
    """The vocabulary the grammar compiler works over, serialized: as
    xgrammar's Hugging Face integration builds it, ending a grammar on the model's
    EOS ids and the tokenizer's own and extra stop ids."""
    from xgrammar import TokenizerInfo

    stop_token_ids = None
    if eos_token_ids:
        stop_token_ids = set(eos_token_ids)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)
        stop_token_ids.update(
            getattr(tokenizer, "additional_stop_token_ids", None) or ()
        )
        stop_token_ids = sorted(stop_token_ids)
    return TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=stop_token_ids
    ).serialize_json()


@dataclass(frozen=True)
class EplbSettings:
    """The online expert rebalance's settings (``--enable-eplb``).

    Attributes:
        specs: The placement geometry the device side reports.
        algorithm: ``--eplb-algorithm``.
        rebalance_num_iterations: Forwards between snapshots.
        layers_per_chunk: Layers moved per round.
        ep_ranks: This rank's EP group, in EP-rank order.
    """

    specs: ExpertRebalanceSpecs
    algorithm: str
    rebalance_num_iterations: int
    layers_per_chunk: int
    ep_ranks: tuple[int, ...]


@dataclass(frozen=True)
class HostConfig:
    """What the host's control plane is configured from.

    Attributes:
        handshake_address: The frontend's handshake endpoint.
        engine_index: The engine identity this rank registers as.
        ready_response: The msgpack-encoded ``WireEngineCoreReadyResponse``.
        in_flight_depth: Dispatched forwards that may await commit.
        vocab_size: Token ids in the vocabulary.
        eos_token_ids: The model's EOS ids.
        max_model_len: The longest request, prompt and output.
        max_req_input_len: The longest prompt.
        logprob_servability: Which logprobs the engine returns; its
            ingress refuses the others.
        lm_head_tp_under_dp: LM-head TP under attention DP, which cannot
            compute prompt logprobs.
        stream_interval: The server's streaming interval.
        spec_num_tokens: The verify width with speculative decoding, else
            None.
        ngram_context_len: Predecessors Engram reads per token, 0 for a
            model without Engram.
        request_token_history: The model reads each request's committed
            token history (``ModelConfig.requires_request_token_history``).
        physical_context_len: Tokens a request's buffers hold.
        kv_total_pages: Usable cache pages, reported with the load.
        rank: This rank.
        replica_ranks: The ranks that mirror this rank's scheduler: its
            attention-TP group, across every pipeline stage. The first owns
            the frontend.
        pp_ranks: With pipeline parallelism, this rank's pipeline, in stage
            order; the last stage samples. None without.
        dp_ranks: With attention DP, every rank: their forwards run in step.
            None without.
        collective_bind_host: Where the rank listens for the other ranks'
            collective connections.
        multimodal: The model runs its vision tower: requests may carry
            multimodal inputs.
        memory_saver: ``--enable-memory-saver``: memory can be released.
        memory_regions: The regions this engine can release.
        metrics_prometheus_dir: ``PROMETHEUS_MULTIPROC_DIR``, where the host
            writes its Prometheus series, when the Prometheus reporter is
            on; else None.
        metrics_relay_endpoint: The node's metrics relay, when an output
            plugin is selected; else None.
        engine_metrics: This rank reports the scheduler loop's metrics
            (the ``EngineMetrics`` set): the attention-TP group's first rank,
            with ``--enable-metrics``.
        transfer_metrics: This rank reports PD transfer metrics (the
            ``KVTransferMetrics`` set): every rank, with ``--enable-metrics``.
        metrics_model_name: The ``model_name`` label.
        metrics_app_key: The ``app_key`` label.
        dp_rank: This rank's attention-DP rank, the ``dp_rank`` label.
        batch_log: This rank writes the batch lines (``BatchLogger``): the
            attention-TP group's first rank, on the first pipeline stage.
        decode_log_interval: Decode rounds between two decode lines.
        spec_num_steps: Draft steps per verify, 0 without speculation.
        cache_state_group_ids: The state-family cache groups.
        log_spec_accept_lengths: ``TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS``.
        finish_log: This rank logs each request's finish line: the
            attention-TP group's first rank.
        log_request_stats: ``--enable-log-request-stats``: the finish line is
            the request's timing summary.
        kv_events: The scheduler records KV cache events
            (``--kv-events-config`` with ``enable_kv_cache_events``).
        kv_events_publisher: Where this rank publishes them (the ZMQ
            publisher's fields), on the attention-TP group's first rank; else
            None.
        l3: With the L3 storage tier, what the host starts its executor
            from; else None.
        l3_metrics: This rank reports the L3 tier's metrics: with the tier
            and ``--enable-metrics``.
        attn_tp_rank: This rank's attention-TP rank, an L3 metrics label.
        engine_role: ``--disaggregation-mode``, an L3 metrics label.
        eplb: With ``--enable-eplb``, the rebalance's settings; else None.
        epd_prefill: An EPD prefill node: requests whose images an encode
            node embeds wait for the embeddings before they are scheduled;
            its first stage receives them (``HostBootstrap.epd_receive``).
        grammar: Grammar-constrained decoding's settings; None with
            ``--grammar-backend none``.
        eager_grammar: The forward fills grammar bitmasks from the
            matchers' committed state (``DeviceSpecs.uses_eager_grammar``).
    """

    handshake_address: str
    engine_index: int
    ready_response: bytes
    in_flight_depth: int
    vocab_size: int
    eos_token_ids: tuple[int, ...]
    max_model_len: int
    max_req_input_len: int
    logprob_servability: LogprobServability
    lm_head_tp_under_dp: bool
    stream_interval: int
    spec_num_tokens: int | None
    ngram_context_len: int
    request_token_history: bool
    physical_context_len: int
    kv_total_pages: int
    rank: int
    replica_ranks: tuple[int, ...]
    pp_ranks: tuple[int, ...] | None
    dp_ranks: tuple[int, ...] | None
    collective_bind_host: str
    multimodal: bool
    memory_saver: bool
    memory_regions: tuple[str, ...]
    metrics_prometheus_dir: str | None
    metrics_relay_endpoint: str | None
    engine_metrics: bool
    transfer_metrics: bool
    metrics_model_name: str
    metrics_app_key: str
    dp_rank: int
    batch_log: bool
    decode_log_interval: int
    spec_num_steps: int
    cache_state_group_ids: tuple[str, ...]
    log_spec_accept_lengths: bool
    finish_log: bool
    log_request_stats: bool
    kv_events: bool
    kv_events_publisher: KVEventsConfig | None
    l3: L3Settings | None
    l3_metrics: bool
    attn_tp_rank: int
    engine_role: str
    eplb: EplbSettings | None
    epd_prefill: bool
    grammar: GrammarSettings | None
    eager_grammar: bool


_SAMPLING_PARAMS = msgspec.msgpack.Decoder(SamplingParams)

# The Host pools an L3 executor addresses, kept for the process.
_L3_HOST_POOLS: list = []


@dataclass
class _AdmittedRequest:
    """A request as the data plane holds it from admission to release."""

    sampling_params: SamplingParams
    stop_criteria: StopCriteria
    # Read by ``multimodal_context_for_forward``; None for a text request.
    multimodal_inputs: MultimodalInputs | None
    # Its grammar's matcher in the host, once compiled; None without one.
    grammar: HostGrammar | None = None


@dataclass(frozen=True)
class _Dispatched:
    """A launched forward as the host holds it until its commit."""

    results: ModelExecutionResult
    forward_mode: ForwardMode


def _logprob_lists(results) -> tuple | None:
    """A forward's prompt-row logprobs and top-k alternatives as lists, None
    when it computed none."""

    def top(values, ids):
        return (values.tolist(), ids.tolist()) if values is not None else None

    if (
        results.input_token_logprobs is None
        and results.input_top_logprobs_val is None
        and results.output_top_logprobs_val is None
    ):
        return None
    return (
        (
            results.input_token_logprobs.tolist()
            if results.input_token_logprobs is not None
            else None
        ),
        top(results.input_top_logprobs_val, results.input_top_logprobs_idx),
        top(results.output_top_logprobs_val, results.output_top_logprobs_idx),
    )


class HostDataPlane:
    """The data plane as the host drives it.

    Two threads call in. The host's forward thread runs every operation but
    the completion queries, one at a time in the order the control thread
    queued them, so a request is admitted before any plan names it and
    released only after the last forward that does was launched. The control
    thread runs the completion queries -- ``complete`` and
    ``poll_cache_events`` -- under its dispatch guard: they read what landed
    and touch no stream.
    """

    def __init__(
        self,
        handle,
        profiler: Profiler,
        regions,
        report_compiles: bool,
        max_encoder_tokens: int | None,
        attn_tp_cpu_group,
        epd_publisher,
    ) -> None:
        self._device = handle
        # The control plane exports the serving-time compilations.
        self._report_compiles = report_compiles
        self._profiler = profiler
        # Tag -> MemoryRegion (``memory_occupation.memory_regions``).
        self._regions = regions
        # Per request: what every forward it runs in needs, fixed at
        # admission.
        self._requests: dict[str, _AdmittedRequest] = {}
        # Caps each multimodal encoder call (the forward token bound); None
        # for a model without a vision tower.
        self._max_encoder_tokens = max_encoder_tokens
        # The ranks a request's shared-memory features are exchanged over.
        self._attn_tp_cpu_group = attn_tp_cpu_group
        # EPD, a multimodal prefill node's first stage: publishes the
        # embeddings the host's receiver landed (``EmbeddingPublisher``); None
        # elsewhere.
        self._epd = epd_publisher
        # The host's matcher entries (``set_grammar_entries``), with grammar
        # on.
        self._grammar_entries: HostGrammarEntries | None = None
        self._gc_hooks = GcHooks(full_collection_interval=FULL_COLLECTION_INTERVAL)

    @staticmethod
    def control_thread_guard() -> AbstractContextManager:
        """What the host's control thread runs under while it serves: it runs
        the completion queries only, and device work crosses only through the
        forward thread (``maybe_control_plane_guard``)."""
        return maybe_control_plane_guard()

    def enter_forward_thread(self) -> None:
        """Called once by the host's forward thread, before its first
        operation."""
        threading.current_thread().name = "tokenspeed::forward"
        self._device.bind_forward_thread()

    def exit_forward_thread(self) -> None:
        """The forward thread's last call, once nothing more is queued:
        finish a drafter's pending round before the caches close."""
        self._device.settle()

    def admit(
        self,
        requests: list[tuple[str, bytes, list[int], int, list | None]],
    ) -> None:
        """Register requests: ``(rid, sampling params, stop token ids, token
        limit, multimodal frames)`` each. The control plane already resolved
        the parameters (``is_normalized`` is set), so decoding them changes
        nothing. The multimodal frames are a generate request on the wire
        carrying only the request's multimodal inputs, whose features this
        side owns from here on."""
        carriers = []
        for rid, encoded, stop_token_ids, token_limit, multimodal in requests:
            carrier = (
                zmq_wire.decode_request_frames(multimodal)[0]
                if multimodal is not None
                else None
            )
            if carrier is not None:
                carriers.append(carrier)
            self._requests[rid] = _AdmittedRequest(
                sampling_params=_SAMPLING_PARAMS.decode(encoded),
                stop_criteria=StopCriteria(
                    stop_token_ids=tuple(stop_token_ids), token_limit=token_limit
                ),
                multimodal_inputs=(
                    carrier.multimodal_inputs if carrier is not None else None
                ),
            )
        if carriers:
            # Every attention-TP rank admits the same requests in the same
            # order, so the shared-memory reachability exchange pairs up.
            prepare_shm_features(carriers, self._attn_tp_cpu_group)

    def execute(
        self,
        plan,
        dp: tuple[list[int], list[int], list[int], bool] | None,
        ngram: tuple[list[list[int]], list[int]] | None,
        grammar_advance: list[bool] | None,
        logprobs: tuple[tuple[list[int], list[int], list[int], int] | None, int] | None,
        history: tuple[list[int], list[int], list[list[int]]] | None,
    ) -> _Dispatched | None:
        """Run one round's plan; the forward to complete later, if one ran.

        Args:
            plan: The round's plan.
            dp: With attention DP, every rank's batch: tokens, requests and
                prefill rows each, and whether this rank runs an idle forward
                in place of its batch. None without attention DP.
            ngram: Engram's input windows and positions for the batch, None
                for a model without Engram.
            grammar_advance: With a grammar row in the batch, whether each
                row's sampled token advances its matcher; None without one.
            logprobs: With a batch, the logprobs it computes beyond the
                sampled tokens': the prompt rows (``InputLogprobPlan``'s
                row starts, counts, positions and top-k), None when no row
                needs one, and the prefill rows' sampled-token top-k.
            history: For a model that reads request token history, the
                committed prefixes the batch seeds (``RequestHistorySeeds``'
                slots, prefix lengths and tokens); None when no row needs
                one.
        """
        dp_metadata = None
        idle_forward = False
        if dp is not None:
            num_tokens, batch_sizes, num_extends, idle_forward = dp
            dp_metadata = DpForwardMetadata.from_ranks(
                global_num_tokens=num_tokens,
                global_batch_size=batch_sizes,
                global_num_extends=num_extends,
                need_idle_forward=idle_forward,
            )
        forward_ops = plan.forward
        planned = None
        # The control plane commits a forward exactly when this holds.
        if (
            not idle_forward
            and len(forward_ops) > 0
            and len(forward_ops[0].request_ids) > 0
        ):
            forward_op = forward_ops[0]
            requests = [self._requests[rid] for rid in forward_op.request_ids]
            input_rows, sampled_top_k = logprobs
            planned = PlannedForward(
                forward_op=forward_op,
                sampling_params_list=[request.sampling_params for request in requests],
                dp_metadata=dp_metadata,
                grammar_inputs=(
                    GrammarStepInputs(
                        grammars=[request.grammar for request in requests],
                        advance_mask=grammar_advance,
                    )
                    if grammar_advance is not None
                    else None
                ),
                multimodal_context=(
                    multimodal_context_for_forward(
                        forward_op, self._requests, self._max_encoder_tokens
                    )
                    if self._max_encoder_tokens is not None
                    else None
                ),
                ngram_inputs=(
                    NGramInputs(
                        tokens=tuple(tuple(window) for window in ngram[0]),
                        positions=tuple(ngram[1]),
                    )
                    if ngram is not None
                    else None
                ),
                request_history_seeds=(
                    RequestHistorySeeds(
                        slots=tuple(history[0]),
                        prefix_lengths=tuple(history[1]),
                        tokens=tuple(tuple(prefix) for prefix in history[2]),
                    )
                    if history is not None
                    else None
                ),
                input_logprob_plan=(
                    InputLogprobPlan(
                        row_starts=tuple(input_rows[0]),
                        counts=tuple(input_rows[1]),
                        position_starts=tuple(input_rows[2]),
                        top_k=input_rows[3],
                    )
                    if input_rows is not None
                    else None
                ),
                sampled_top_k=sampled_top_k,
                stop_criteria=tuple(request.stop_criteria for request in requests),
            )
            # Every handshaked image this forward scatters was received at
            # admission; one that was not is a bug, not a cue to encode.
            assert_embeddings_received(self._epd, planned.multimodal_context)
        # Each op once, validated before any owner runs one: the Host copies
        # go to the device; the host's L3 executor took the storage
        # operations.
        cache_ops = cache_ops_from_plan(plan)
        results = self._device.execute(plan, planned, cache_ops=cache_ops.device)
        if idle_forward:
            self._device.run_idle_forward(dp_metadata)
        self._gc_hooks.note_round(forwarded=results is not None or idle_forward)
        if results is None:
            return None
        return _Dispatched(
            results,
            ForwardMode.from_num_extends(
                planned.forward_op.num_extends(), len(planned.forward_op.request_ids)
            ),
        )

    def complete(self, dispatched: _Dispatched) -> tuple[
        list[int],
        list[int],
        list[float] | None,
        list[int] | None,
        list[list[int]] | None,
        tuple[int, float] | None,
        list[int] | None,
        tuple | None,
    ]:
        """Wait for a forward's host copies to land (the control thread: the
        event wait releases the interpreter lock), and return its sampled
        tokens, kept lengths, logprobs, NaN flags and, on the P role, each
        row's bootstrap payload (``next_input_ids``), as the commit reads
        them; then, when the host exports metrics, the serving-time kernel
        compilations and their seconds so far; where this side advanced the
        grammar matchers, each row's index of the token that ended its
        grammar; and, when the forward computed any, its planned prompt
        rows' logprobs, their top-k alternatives and the prefill rows'
        sampled-token alternatives (values and ids each), None for those it
        did not."""
        results = dispatched.results
        results.sync()
        compiles = compile_stats() if self._report_compiles else None
        return (
            results.output_tokens.tolist(),
            results.output_lengths.tolist(),
            (
                results.output_logprobs.tolist()
                if results.output_logprobs is not None
                else None
            ),
            (
                results.output_nan_flags.tolist()
                if results.output_nan_flags is not None
                else None
            ),
            (
                [list(row) for row in results.next_input_ids.tolist()]
                if results.next_input_ids is not None
                else None
            ),
            (
                (compiles.serving_compiles, compiles.serving_seconds)
                if compiles is not None
                else None
            ),
            resolve_grammar_completion(results),
            _logprob_lists(results),
        )

    def note_committed(self, dispatched: _Dispatched) -> None:
        """A committed forward counts toward a profile's steps. The forward
        thread runs this, after the commit's ``complete``: the profiler
        starts and stops on the thread whose launches it traces."""
        self._profiler.note_forward(dispatched.forward_mode)

    def poll_cache_events(self) -> list[tuple[str, int]]:
        """The cache ops acknowledged since the last poll, in the order they
        landed: ``(kind, op_id)`` each, as ``cache_event_to_payload`` names
        them."""
        return [
            (payload["kind"], payload["op_id"])
            for payload in map(
                cache_event_to_payload, self._device.poll_cache_results()
            )
        ]

    def release(self, rids: list[str]) -> None:
        """Forget requests that ended. Their grammar matchers and
        shared-memory features go behind every forward launched before this,
        the ones that may read them."""
        requests = [(rid, self._requests.pop(rid)) for rid in rids]
        if any(
            request.grammar is not None
            or (
                request.multimodal_inputs is not None
                and request.multimodal_inputs.has_pending_shm_features()
            )
            for _, request in requests
        ):
            # A release is a data-plane task: it runs behind a drafter's
            # pending round.
            self._device.settle()
        for rid, request in requests:
            if request.grammar is not None:
                request.grammar.release()
            features = request.multimodal_inputs
            if features is not None and features.has_pending_shm_features():
                try:
                    features.release_shm_features()
                except Exception:
                    # Nothing waits on the release: log the leak rather than
                    # stop serving.
                    logger.exception(
                        f"releasing the shared-memory features of {rid} failed"
                    )

    def land(
        self,
        spec_candidates: tuple[int, list[int]] | None,
        cache_slot: int | None,
    ) -> None:
        """D role: a remote prefill landed; write its drafter candidates
        ``(slot, ids)`` and make ``cache_slot`` ready for the first decode
        (``run_remote_prefill_landing``)."""
        self._device.run_remote_prefill_landing(spec_candidates, cache_slot)

    def idle_forward(self, dp: tuple[list[int], list[int], list[int]]) -> None:
        """A frozen round's idle forward: every rank's tokens, requests and
        prefill rows."""
        num_tokens, batch_sizes, num_extends = dp
        self._device.run_idle_forward(
            DpForwardMetadata.from_ranks(
                global_num_tokens=num_tokens,
                global_batch_size=batch_sizes,
                global_num_extends=num_extends,
                need_idle_forward=True,
            )
        )

    def update_weights(self, payload: bytes) -> tuple[bool, str]:
        """Apply one weight operation (a msgpack control struct), ordered
        behind the forwards dispatched before it."""
        ok, message = self._device.update_weights(
            zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_CONTROL, payload])[0]
        )
        return bool(ok), str(message)

    def profile(self, payload: bytes) -> tuple[bool, str]:
        """Start or stop the profiler (a msgpack ``ProfileReq``)."""
        request = zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_CONTROL, payload])[
            0
        ]
        output = self._profiler.profile(request)
        return bool(output.success), str(output.message)

    def release_memory(self, tags: list[str]) -> None:
        """Free memory regions, in order; the scheduler has drained."""
        for tag in tags:
            self._regions[tag].release()

    def resume_memory(self, tags: list[str]) -> None:
        """Restore memory regions, in order."""
        for tag in tags:
            self._regions[tag].resume()

    def set_grammar_entries(self, entries: list[tuple[str, int]]) -> None:
        """The host's matcher entries, before serving starts."""
        self._grammar_entries = HostGrammarEntries(dict(entries))

    def attach_grammars(self, grammars: list[tuple[str, int, list[int]]]) -> None:
        """Give requests their grammar matchers ``(rid, handle, the tokens
        that end the grammar)``, for the forwards they run in from now on."""
        for rid, handle, stop_token_ids in grammars:
            self._requests[rid].grammar = HostGrammar(
                self._grammar_entries, handle, frozenset(stop_token_ids)
            )

    def epd_publish(
        self,
        landed: list[tuple[str, list[tuple[int, int | None, int, int, int, bool]]]],
    ) -> None:
        """EPD: put the embeddings the host's receiver landed onto their
        requests' items ``(rid, landed items)``, before any forward of
        theirs; with row sharding the replica's ranks rebuild each image's
        other rows. Like every data-plane task, behind a drafter's pending
        round."""
        self._device.settle()
        self._epd.publish(
            [
                (self._requests[rid].multimodal_inputs.mm_items, items)
                for rid, items in landed
            ]
        )

    def epd_released(self) -> list[int]:
        """EPD: the receive slots whose rows ``epd_publish`` finished copying
        out; a completion query, on the control thread."""
        return self._epd.released()

    def snapshot_expert_load(self) -> tuple[list[list[int]], list[list[int]]]:
        """This rank's route counters since the previous snapshot and the
        placement they were counted under, as ``[layers, physical]`` rows."""
        snapshot = self._device.snapshot_expert_load()
        return (
            snapshot.physical_count.tolist(),
            snapshot.physical_to_logical_map.tolist(),
        )

    def apply_expert_placement(
        self,
        layer_ids: list[int],
        rows: list[list[int]],
        moves: list[tuple[list, list, list, list]],
    ) -> None:
        """Move one chunk of layers' experts: their new placement rows and
        this rank's ``(recv, send, local_copy, free_rider)`` per layer."""
        self._device.apply_expert_placement(
            layer_ids,
            torch.tensor(rows, dtype=torch.int32),
            {
                layer_id: SlotMoves(
                    recv=tuple(map(tuple, recv)),
                    send=tuple(map(tuple, send)),
                    local_copy=tuple(map(tuple, local_copy)),
                    free_rider=tuple(map(tuple, free_rider)),
                )
                for layer_id, (recv, send, local_copy, free_rider) in zip(
                    layer_ids, moves
                )
            },
        )


class HostBootstrap:
    """One rank, built: the scheduler and config for the control thread, and
    the data plane the forward and control threads call."""

    #: Which host mode this rank runs (``EncodeBootstrap`` is an EPD encode
    #: rank's).
    mode = "serve"

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        connection: Connection,
        pd_outbox,
        metrics_relay_endpoint: str | None,
    ) -> None:
        self._server_args = server_args
        self._connection = connection
        mapping = server_args.mapping
        gpu_id = mapping.rank % mapping.nprocs_per_node + server_args.base_gpu_id
        attn_tp_rank = mapping.attn.tp_rank
        dp_rank = mapping.attn.dp_rank
        global_rank = mapping.rank

        faulthandler.enable()
        # NVSHMEM reads its IB traffic class from the process environment at
        # bootstrap; re-assert the launcher's setting in this process.
        if envs.NVSHMEM_IB_TRAFFIC_CLASS.is_set():
            envs.NVSHMEM_IB_TRAFFIC_CLASS.set(envs.NVSHMEM_IB_TRAFFIC_CLASS.get())
            logger.info(
                f"NVSHMEM_IB_TRAFFIC_CLASS={envs.NVSHMEM_IB_TRAFFIC_CLASS.get():d}"
            )
        configure_logger(server_args, prefix=f" ATTN TP RANK {attn_tp_rank}")
        install_jit_compile_check()
        with startup_phase(
            "scheduler.init", rank=global_rank, role=server_args.disaggregation_mode
        ):
            maybe_warm_cupti_for_graph_capture()

            model_config = load_model_config(
                server_args, server_args.model, draft_of=None
            )
            if server_args.speculative_draft_model_path is not None:
                draft_model_config = load_model_config(
                    server_args,
                    server_args.speculative_draft_model_path,
                    draft_of=model_config,
                )
            else:
                draft_model_config = None
            ngram_context_len = engram_context_len(model_config.hf_text_config)
            if ngram_context_len and mapping.has_pp:
                raise NotImplementedError(
                    "Engram input history requires PP=1 and in-flight depth <= 1"
                )

            min_per_gpu_mem = init_distributed(
                server_args,
                port_args,
                gpu_id=gpu_id,
                global_rank=global_rank,
                hidden_size=model_config.hidden_size,
            )
            overlap_schedule_depth = int(
                should_use_overlap_schedule(
                    disable_overlap_schedule=server_args.disable_overlap_schedule,
                    disaggregation_mode=server_args.disaggregation_mode,
                )
            )
            spec_algorithm = server_args.speculative_algorithm
            # Verify width per decode step.
            decode_input_tokens = (
                server_args.speculative_num_draft_tokens
                if spec_algorithm is not None
                else 1
            )
            kv_events = kv_events_enabled(server_args.kv_events_config)
            kv_events_config = (
                KVEventsConfig.from_cli(server_args.kv_events_config)
                if kv_events
                else None
            )
            # The scheduler's request slots are per attention-DP rank.
            max_batch_size = server_args.max_num_seqs // max(mapping.attn.dp_size, 1)
            device = build_device_side(
                server_args=server_args,
                model_config=model_config,
                draft_model_config=draft_model_config,
                gpu_id=gpu_id,
                global_rank=global_rank,
                attn_tp_rank=attn_tp_rank,
                min_per_gpu_mem=min_per_gpu_mem,
                overlap_schedule_depth=overlap_schedule_depth,
                decode_input_tokens=decode_input_tokens,
                max_batch_size=max_batch_size,
                kv_transfer_builder=functools.partial(
                    build_host_peer, outbox=pd_outbox
                ),
            )
            # The L3 tier the host runs over the shared Host pool; None
            # without a storage backend.
            l3 = l3_settings(
                server_args,
                model_config,
                draft_model_config,
                host_cache=device.host_cache,
                world_cpu_group=(
                    pg_manager.get_process_group("gloo", mapping.world_group)
                    if mapping.world_size > 1
                    else None
                ),
            )
            if l3 is not None:
                # The executor addresses the Host pool by pointer, and native
                # I/O that cannot prove its retirement keeps it past close.
                _L3_HOST_POOLS.append(device.host_cache.storage)
            storage_config = l3.runtime.scheduler if l3 is not None else None
            specs = device.specs
            geometry = specs.cache_geometry
            scheduler_config = scheduler_config_from_args(
                server_args,
                specs,
                storage_config=storage_config,
                max_scheduled_tokens=server_args.chunked_prefill_size,
                max_batch_size=max_batch_size,
                decode_input_tokens=decode_input_tokens,
                overlap_schedule_depth=overlap_schedule_depth,
                enable_kv_cache_events=kv_events,
                prefix_replay_tokens=resolve_dspark_prefix_replay_tokens(
                    speculative_algorithm=spec_algorithm,
                    enable_prefix_caching=server_args.enable_prefix_caching,
                    enable_kvstore=server_args.enable_kvstore,
                    disaggregation_mode=server_args.disaggregation_mode,
                    draft_model_path_use_base=server_args.draft_model_path_use_base,
                    draft_model_config=draft_model_config,
                ),
            )
            #: Driven by the control thread only, once the host takes it.
            self.scheduler = Scheduler(scheduler_config)
            self._max_single_request_tokens = self.scheduler.max_single_request_tokens()
            max_model_len, max_req_input_len = request_length_limits(
                context_len=model_config.context_len,
                max_single_request_tokens=self._max_single_request_tokens,
                decode_input_tokens=decode_input_tokens,
            )
            self._max_total_num_tokens = geometry.token_capacity
            self._max_model_len = max_model_len
            self._max_req_input_len = max_req_input_len
            self._multimodal_encoder_dtype = specs.multimodal_encoder_dtype
            self._cache_storage = specs.cache_storage
            self._supports_prompt_logprobs = specs.supports_prompt_logprobs
            ready_response = zmq_wire.engine_ready_response(
                server_args=server_args,
                model_config=model_config,
                geometry=geometry,
                multimodal_encoder_dtype=specs.multimodal_encoder_dtype,
                world_size=server_args.world_size or mapping.world_size,
                dp_size=mapping.attn.dp_size,
                attn_tp_size=server_args.attn_tp_size or mapping.attn.tp_size,
                dp_rank=dp_rank,
                kv_cache_size_tokens=geometry.token_capacity,
            )
            if (specs.expert_rebalance is None) == server_args.enable_eplb:
                raise RuntimeError(
                    "--enable-eplb and the device side's expert placement updater "
                    "disagree: the server needs both or neither"
                )
            # The tokens that end every request not ignoring EOS: the model's EOS
            # ids and the tokenizer's, as the request handler resolves them.
            tokenizer = get_tokenizer(
                server_args.tokenizer,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                architectures=model_config.hf_config.architectures,
                **model_config.tokenizer_kwargs,
            )
            eos_token_ids = resolve_stop_token_ids(
                SamplingParams(), model_config.hf_eos_token_id, tokenizer
            )
            regions = memory_regions(
                device.handle,
                TorchMemorySaverAdapter.create(enable=server_args.enable_memory_saver),
                nccl_suspendable=nccl_suspend_supported(),
            )
            # EPD: a multimodal prefill node's first stage receives image
            # embeddings from encode nodes; None for every other rank.
            epd_receive = build_epd_receive(
                server_args,
                model_config=model_config,
                encoder_model_facts=device.encoder_model_facts,
                mapping=mapping,
                attn_tp_rank=attn_tp_rank,
                attn_tp_size=server_args.attn_tp_size or mapping.attn.tp_size,
                attn_tp_cpu_group=pg_manager.get_process_group(
                    "gloo", mapping.attn.tp_group
                ),
                pg_manager=pg_manager,
            )
            self.config = HostConfig(
                handshake_address=server_args.zmq_handshake_endpoint(),
                engine_index=server_args.zmq_engine_index + dp_rank,
                ready_response=zmq_wire.encode(ready_response),
                # The chunk pipeline keeps one forward per stage in flight.
                in_flight_depth=(
                    mapping.pp_size if mapping.has_pp else overlap_schedule_depth
                ),
                vocab_size=model_config.vocab_size,
                eos_token_ids=tuple(sorted(eos_token_ids)),
                max_model_len=max_model_len,
                max_req_input_len=max_req_input_len,
                logprob_servability=LogprobServability.of(
                    server_args,
                    supports_prompt_logprobs=specs.supports_prompt_logprobs,
                    vocab_size=model_config.vocab_size,
                ),
                lm_head_tp_under_dp=mapping.attn.has_dp and mapping.lm_head.has_tp,
                stream_interval=server_args.stream_interval,
                spec_num_tokens=(
                    decode_input_tokens if spec_algorithm is not None else None
                ),
                ngram_context_len=ngram_context_len,
                request_token_history=model_config.requires_request_token_history,
                physical_context_len=(
                    model_config.context_len + server_args.spec_context_pad
                ),
                kv_total_pages=geometry.num_usable_pages,
                rank=global_rank,
                # Every stage runs the same plan, so with a pipeline the replica
                # is the world (ServerArgs refuses attention DP with one). Under
                # rank emulation the other ranks do not run: this rank's
                # collectives are its own, as its process groups are
                # (``init_emulated_rank_zero``).
                replica_ranks=(
                    (global_rank,)
                    if server_args.emulate_rank_zero
                    else (
                        tuple(mapping.world_group)
                        if mapping.has_pp
                        else tuple(mapping.attn.tp_group)
                    )
                ),
                pp_ranks=tuple(mapping.pp_group) if mapping.has_pp else None,
                dp_ranks=tuple(mapping.world_group) if mapping.has_attn_dp else None,
                # One node's ranks reach each other on loopback.
                collective_bind_host="127.0.0.1" if mapping.nnodes == 1 else "0.0.0.0",
                multimodal=model_config.is_multimodal_active,
                memory_saver=server_args.enable_memory_saver,
                memory_regions=tuple(regions),
                metrics_prometheus_dir=(
                    os.environ["PROMETHEUS_MULTIPROC_DIR"]
                    if metrics_enabled(server_args, "prometheus")
                    else None
                ),
                metrics_relay_endpoint=metrics_relay_endpoint,
                engine_metrics=server_args.enable_metrics and attn_tp_rank == 0,
                transfer_metrics=server_args.enable_metrics,
                metrics_model_name=server_args.served_model_name,
                metrics_app_key=server_args.app_key or "",
                dp_rank=dp_rank,
                batch_log=attn_tp_rank == 0
                and (not mapping.has_pp or mapping.pp_rank == 0),
                decode_log_interval=server_args.decode_log_interval,
                spec_num_steps=specs.spec_num_steps,
                cache_state_group_ids=tuple(specs.cache_state_group_ids),
                log_spec_accept_lengths=envs.TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS.get(),
                finish_log=attn_tp_rank == 0,
                log_request_stats=server_args.enable_log_request_stats,
                kv_events=kv_events,
                kv_events_publisher=(
                    kv_events_config
                    if kv_events
                    and attn_tp_rank == 0
                    and kv_events_config.publisher in (None, "zmq")
                    else None
                ),
                epd_prefill=(
                    server_args.disaggregation_mode == "prefill"
                    and model_config.is_multimodal_active
                ),
                grammar=(
                    GrammarSettings(
                        tokenizer_info=grammar_tokenizer_info(
                            tokenizer,
                            model_config.vocab_size,
                            model_config.hf_eos_token_id,
                        ),
                        any_whitespace=not server_args.disable_any_whitespace,
                        compile_timeout_s=float(
                            server_args.grammar_compile_timeout_secs
                        ),
                        max_retries=int(server_args.grammar_compile_max_retries),
                    )
                    if server_args.grammar_backend == "xgrammar"
                    else None
                ),
                eager_grammar=specs.uses_eager_grammar,
                eplb=(
                    EplbSettings(
                        specs=specs.expert_rebalance,
                        algorithm=server_args.eplb_algorithm,
                        rebalance_num_iterations=server_args.eplb_rebalance_num_iterations,
                        layers_per_chunk=server_args.eplb_rebalance_layers_per_chunk,
                        ep_ranks=tuple(mapping.moe.ep_group),
                    )
                    if server_args.enable_eplb
                    else None
                ),
                l3=l3,
                l3_metrics=l3 is not None and server_args.enable_metrics,
                attn_tp_rank=attn_tp_rank,
                engine_role=server_args.disaggregation_mode,
            )
            self._world_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.world_group
            )
            self._collective_host = "127.0.0.1" if mapping.nnodes == 1 else get_ip()
            #: The PD peer the host starts its own from, None outside
            #: disaggregation.
            self.transfer_peer = device.transfer
            #: EPD: the settings the host starts its embedding receiver from,
            #: on a multimodal prefill node's first stage; else None.
            self.epd_receive = epd_receive.settings if epd_receive is not None else None
            self.data_plane = HostDataPlane(
                device.handle,
                Profiler(
                    server_args=server_args,
                    device=device.handle,
                    attn_mapping=mapping.attn,
                    # With a pipeline every stage rendezvouses around a trace.
                    sync_size=(
                        mapping.world_size if mapping.has_pp else mapping.attn.tp_size
                    ),
                    sync_group=pg_manager.get_process_group(
                        "gloo",
                        (
                            mapping.world_group
                            if mapping.has_pp
                            else mapping.attn.tp_group
                        ),
                    ),
                ),
                regions,
                report_compiles=self.config.engine_metrics,
                max_encoder_tokens=(
                    server_args.chunked_prefill_size
                    if model_config.is_multimodal_active
                    else None
                ),
                attn_tp_cpu_group=pg_manager.get_process_group(
                    "gloo", mapping.attn.tp_group
                ),
                epd_publisher=(
                    epd_receive.publisher if epd_receive is not None else None
                ),
            )
            self._device = device.handle

    def collective_addresses(self, port: int) -> list[str]:
        """Every rank's collective address, in rank order, given this rank's
        port; a collective of the whole world, through the rendezvous the
        data plane already has."""
        gathered: list = [None] * self._world_cpu_group.size()
        dist.all_gather_object(
            gathered,
            join_host_port(self._collective_host, port),
            group=self._world_cpu_group,
        )
        return gathered

    def report_ready(self) -> None:
        """Tell the launcher the rank is serving, once the frontend
        handshake is done."""
        server_args = self._server_args
        self._connection.send(
            {
                "status": "ready",
                "max_total_num_tokens": self._max_total_num_tokens,
                "max_req_input_len": self._max_req_input_len,
                "max_single_request_tokens": self._max_single_request_tokens,
                "max_num_seqs": server_args.max_num_seqs,
                "chunked_prefill_size": server_args.chunked_prefill_size,
                "max_model_len": self._max_model_len,
                "multimodal_encoder_dtype": self._multimodal_encoder_dtype,
                "cache_storage": self._cache_storage,
                "supports_prompt_logprobs": self._supports_prompt_logprobs,
            }
        )

    def enter_serving(self) -> None:
        """End startup: from here on a host synchronization or a kernel
        compilation on the data plane is a stall."""
        arm_data_plane_sync_debug(self._server_args.device)
        mark_jit_compile_serving()
        freeze_startup_heap()

    def close(self) -> None:
        """Release the device side's caches, once the forward thread and the
        L3 executor have stopped."""
        self._device.shutdown_cache()


def bootstrap(control_fd: int, pd_outbox) -> HostBootstrap | EncodeBootstrap:
    """Build this host's rank from the arguments the launcher sends on
    ``control_fd``; with PD, its transfer peer reaches the host through
    ``pd_outbox`` (see ``pd/host_peer.py``). An EPD encode rank runs the
    host's encode mode instead (``epd/encode_host.py``)."""
    connection = Connection(control_fd)
    server_args, port_args, metrics_relay_endpoint = connection.recv()
    if server_args.disaggregation_mode == "encode":
        return EncodeBootstrap(server_args, port_args, connection)
    return HostBootstrap(
        server_args, port_args, connection, pd_outbox, metrics_relay_endpoint
    )


def notify_launcher_of_failure() -> None:
    """Tell the launcher this rank failed: the launcher then stops the other
    ranks."""
    os.kill(os.getppid(), signal.SIGUSR1)
