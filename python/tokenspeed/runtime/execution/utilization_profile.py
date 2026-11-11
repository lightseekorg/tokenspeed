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

"""Bounded utilization observation of the runtime's actual eager/graph path."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from tokenspeed_kernel.op_profile import (
    _PROFILE_CONFIG_DEFAULTS,
    CostRegistry,
    ProfileSession,
    _profile_config_for_observation,
    _profile_expand_timeline_config,
)
from tokenspeed_kernel.profile_costs import (
    register_collective_costs,
    register_compute_costs,
)
from tokenspeed_kernel.profile_graph import (
    GraphCapture,
    GraphNodes,
    GraphReplayCollector,
    ProfileStep,
    ProfileWindow,
)
from tokenspeed_kernel.profile_nvtx import NvtxTimeline, NvtxUnavailableError
from tokenspeed_kernel.profile_utilization import HardwareProfile

from tokenspeed.runtime.utils.host_sync import allow_host_sync


@runtime_checkable
class _CostProvider(Protocol):
    def configure_utilization_profile(
        self, costs: CostRegistry, *, hbm_policy: str
    ) -> dict: ...


@dataclass(frozen=True)
class UtilizationProfileConfig:
    run_id: str
    output_dir: str
    hardware: str
    layers: str
    start_step: int
    num_steps: int
    max_scopes: int
    hbm_policy: str
    cuda_runtime_library: str | None = _PROFILE_CONFIG_DEFAULTS["cuda_runtime_library"]
    cupti_library: str | None = _PROFILE_CONFIG_DEFAULTS["cupti_library"]
    max_metadata_bytes: int = _PROFILE_CONFIG_DEFAULTS["max_metadata_bytes"]
    metadata_slots: int = _PROFILE_CONFIG_DEFAULTS["metadata_slots"]
    ranks: list[int] | None = _PROFILE_CONFIG_DEFAULTS["ranks"]
    startup_observation: dict | None = _PROFILE_CONFIG_DEFAULTS["startup_observation"]

    def __post_init__(self):
        for name in (
            "start_step",
            "num_steps",
            "max_scopes",
            "max_metadata_bytes",
            "metadata_slots",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.run_id, str)
            or not self.run_id
            or not self.output_dir
            or self.hbm_policy not in {"logical_io_once", "unavailable"}
        ):
            raise ValueError("Explicit run/output/HBM policy required")
        if self.ranks is not None and (
            not self.ranks
            or any(type(rank) is not int or rank < 0 for rank in self.ranks)
            or len(set(self.ranks)) != len(self.ranks)
        ):
            raise ValueError("ranks must contain distinct nonnegative integers")


class RuntimeUtilizationProfile:
    """One configured rank; preserves computation and exposes exact result IDs."""

    def __init__(
        self,
        config,
        *,
        model,
        model_id: str,
        rank: int,
        graph_enabled: bool = False,
        mapping=None,
    ):
        self.config, self.model, self.model_id, self.rank = (
            config,
            model,
            model_id,
            rank,
        )
        self.graph_enabled = graph_enabled
        self.hardware = HardwareProfile.from_file(config.hardware)
        self.output = Path(config.output_dir) / f"rank-{rank}.metrics.json"
        self.costs = CostRegistry()
        register_compute_costs(self.costs, hbm_policy=config.hbm_policy)
        register_collective_costs(self.costs, hbm_policy=config.hbm_policy)
        layers = model.get_submodule(config.layers)
        if not isinstance(layers, torch.nn.ModuleList) or not len(layers):
            raise ValueError("layers must name a nonempty ModuleList of decoder layers")
        self.modules = []
        for path, module in model.named_modules():
            layer_id = None
            prefix = config.layers + "."
            if path.startswith(prefix):
                piece = path[len(prefix) :].split(".", 1)[0]
                if piece.isdigit():
                    layer_id = int(piece)
            self.modules.append((module, layer_id, path or type(model).__name__))
        self.source_context = dict(
            run_id=config.run_id,
            rank=rank,
            profile_config_sha256=hashlib.sha256(
                json.dumps(
                    asdict(config),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
            startup_observation=config.startup_observation,
        )
        self.static_metadata = dict(
            profile_config_sha256=self.source_context["profile_config_sha256"],
            startup_observation_sha256=(config.startup_observation or {}).get("sha256"),
            attention_tp_size=None,
            dcp_size=None,
            expert_tp_size=None,
            expert_ep_size=None,
        )
        if mapping is not None:
            self.static_metadata.update(
                attention_tp_size=mapping.attn.tp_size,
                dcp_size=mapping.attn.kvp_size,
                expert_tp_size=mapping.moe.tp_size,
                expert_ep_size=mapping.moe.ep_size,
            )
        if isinstance(model, _CostProvider):
            self.static_metadata.update(
                model.configure_utilization_profile(
                    self.costs, hbm_policy=config.hbm_policy
                )
            )
        self._captures: dict[tuple, GraphCapture] = {}
        self._capture_sequence = 0
        self._nodes: GraphNodes | None = None
        self._replays: GraphReplayCollector | None = None
        self.current_step: ProfileStep | None = None
        self._window = ProfileWindow(
            start_step=config.start_step, num_steps=config.num_steps
        )
        self.session = self._session("eager")
        self._timeline: NvtxTimeline | None = None
        self._eager_hooks: ExitStack | None = None
        self._finished = False
        self._capture_collection_started = False
        directory = Path(config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(
            asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        snapshot = dict(
            schema="tokenspeed.profile_config/v1",
            run_id=config.run_id,
            rank=rank,
            config=asdict(config),
            contents=contents,
            sha256=hashlib.sha256(contents.encode()).hexdigest(),
            hardware=asdict(self.hardware),
            graph_enabled=graph_enabled,
            static_metadata=self.static_metadata,
        )
        with (directory / f"rank-{rank}.config.json").open("x") as stream:
            json.dump(snapshot, stream, allow_nan=False)

    def _session(self, suffix):
        return ProfileSession(
            run_id=(
                self.config.run_id
                if suffix == "eager"
                else f"{self.config.run_id}:{suffix}"
            ),
            model_id=self.model_id,
            rank=self.rank,
            backend="none" if self.graph_enabled else "nvtx",
        )

    def _collector(self):
        if self._replays is None:
            self._replays = GraphReplayCollector(
                output=Path(self.config.output_dir) / f"rank-{self.rank}.replays.jsonl",
                metadata_slots=self.config.metadata_slots,
                max_metadata_bytes=self.config.max_metadata_bytes,
                max_steps=self.config.num_steps,
                source_context=self.source_context,
            )
        return self._replays

    @classmethod
    def from_env(
        cls,
        *,
        model,
        model_id,
        rank,
        device,
        enforce_eager,
        disable_prefill_graph,
        speculative,
        mapping=None,
    ):
        path = os.environ.get("TOKENSPEED_TIMELINE_CONFIG")
        if not path:
            return None
        if torch.device(device).type != "cuda" or torch.version.hip is not None:
            raise ValueError("Native utilization timelines require NVIDIA CUDA")
        actual_config = _profile_expand_timeline_config(
            json.loads(Path(path).read_text())
        )
        observation = actual_config["startup_observation"]
        if (
            observation is not None
            and actual_config
            != _profile_config_for_observation(
                observation, run_id=actual_config["run_id"]
            )
        ):
            raise ValueError(
                "Actual timeline config differs from the before-Engine observation"
            )
        config = UtilizationProfileConfig(**actual_config)
        # Standalone eager profiling needs no process group. Distributed ranks
        # still gather before the filter, including uncollected peers.
        if torch.distributed.is_initialized():
            from tokenspeed_kernel.profile_communication import (
                bind_communication_profile,
            )

            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            world_ranks = (
                tuple(mapping.world_group)
                if mapping is not None
                else tuple(range(torch.distributed.get_world_size()))
            )
            bind_communication_profile(
                run_id=config.run_id,
                config=asdict(config),
                rank=rank,
                device=device,
                cpu_group=pg_manager.get_process_group("gloo", world_ranks),
            )
        if config.ranks is not None and rank not in config.ranks:
            return None
        expected = os.environ.get("TOKENSPEED_PROFILE_RUN_ID")
        if expected is not None and expected != config.run_id:
            raise ValueError("Profile run_id differs from service trace run_id")
        if not enforce_eager and (
            not config.cuda_runtime_library or not config.cupti_library
        ):
            raise ValueError(
                "Graph profiles require explicit CUDA runtime and loaded profiler CUPTI libraries"
            )
        # These flags describe the real path; observation never changes them.
        return cls(
            config,
            model=model,
            model_id=model_id,
            rank=rank,
            graph_enabled=not enforce_eager,
            mapping=mapping,
        )

    def start_capture_collection(self, *, graph_key, batch_size):
        """Start once before the capture rendezvous; keep tracing through rebind."""
        if (
            not self.graph_enabled
            or batch_size != 1
            or self._capture_collection_started
        ):
            return
        torch.cuda.profiler.start()
        self._capture_collection_started = True
        # This is actual startup evidence, not a claim of native node coverage.
        receipt = dict(
            schema="tokenspeed.profile_collection/v1",
            run_id=self.config.run_id,
            rank=self.rank,
            process_id=os.getpid(),
            event="cuda_profiler_started_before_capture_rendezvous",
            graph_key=graph_key,
            next_capture_sequence=self._capture_sequence + 1,
        )
        target = Path(self.config.output_dir) / f"rank-{self.rank}.collection.json"
        with target.open("x") as stream:
            json.dump(receipt, stream, allow_nan=False)

    def arm(self) -> bool:
        """Arm the logical window and report whether CUDA collection is running."""
        collector = self._collector()
        self._window.arm()
        try:
            collector.emit(
                dict(
                    event="window_started",
                    run_id=self.config.run_id,
                    rank=self.rank,
                    start_step=self.config.start_step,
                    num_steps=self.config.num_steps,
                )
            )
        except Exception:
            self._window.stop()
            raise
        return self._capture_collection_started

    def stop(self):
        self._window.stop()
        state = self._collector().flush_ready()
        report = dict(
            schema="tokenspeed.profile_window/v1",
            run_id=self.config.run_id,
            rank=self.rank,
            start_step=self.config.start_step,
            num_steps=self.config.num_steps,
            observed_serving_steps=self._window.observed,
            **state,
        )
        target = Path(self.config.output_dir) / f"rank-{self.rank}.window.json"
        with target.open("x") as stream:
            json.dump(report, stream, allow_nan=False)
        return report

    def begin_step(self, *, step, request_ids, pool_indices):
        if not self._window.select():
            return None
        return ProfileStep(
            owner=self._collector(),
            run_id=self.config.run_id,
            rank=self.rank,
            serving_step=step,
            request_ids=tuple(request_ids),
            pool_indices=tuple(pool_indices),
        )

    @contextmanager
    def capture(self, *, graph_key, batch_size, query_width):
        if not self.graph_enabled or batch_size != 1:
            yield
            return
        if self._nodes is None:
            self._nodes = GraphNodes(
                cuda_runtime_library=self.config.cuda_runtime_library,
                cupti_library=self.config.cupti_library,
            )
        self._collector()
        self.hardware = self.hardware.validate_cuda_device(torch.cuda.current_device())
        self._capture_sequence += 1
        capture_id = f"{self.config.run_id}:r{self.rank}:c{self._capture_sequence}"
        capture = GraphCapture(
            session=self._session(capture_id),
            costs=self.costs,
            hardware=self.hardware,
            capture_id=capture_id,
            graph_key=graph_key,
            nodes=self._nodes,
            max_metadata_bytes=self.config.max_metadata_bytes,
            max_scopes=self.config.max_scopes,
            metadata={
                **self.static_metadata,
                "run_id": self.config.run_id,
                "logical_batch": batch_size,
                "query_width": query_width,
            },
            output=Path(self.config.output_dir)
            / f"rank-{self.rank}.capture-{self._capture_sequence}.json",
            modules=self.modules,
        )
        with capture:
            yield
        self._captures[graph_key] = capture

    def replay(
        self, *, graph_key, batch_size, padded_batch_size, query_width, query_shard
    ):
        step = self.current_step
        if step is None:
            return nullcontext()
        geometry = dict(
            batch_size=batch_size,
            padded_batch_size=padded_batch_size,
            query_width=query_width,
            query_shard=query_shard is not None,
        )
        return self._collector().replay(
            step, self._captures.get(graph_key), geometry=geometry
        )

    def release_graphs(self):
        self._captures.clear()

    @contextmanager
    def eager(self, *, batch_size, input_tokens, decode_query_width, query_sharded):
        if not self.graph_enabled or self.current_step is None:
            yield
            return
        step = self.current_step
        session = self._session(f"eager:{step.serving_step}")
        session.backend = "nvtx"
        # Packed prefill rows can have unequal per-request widths. Preserve the
        # actual live token count without inventing a rectangular batch.
        metadata = {**self.static_metadata, "logical_batch": batch_size}
        if not query_sharded:
            metadata["logical_tokens"] = input_tokens
            if batch_size == 1:
                metadata["query_width"] = input_tokens
            elif (
                decode_query_width is not None
                and input_tokens == batch_size * decode_query_width
            ):
                metadata["query_width"] = decode_query_width
        with step.owner.reserve_eager_metadata(step) as reserved:
            if not reserved:
                yield
                return
            with ExitStack() as stack:
                timeline = NvtxTimeline(
                    session,
                    costs=self.costs,
                    hardware=self.hardware,
                    stream=torch.cuda.current_stream(),
                    max_scopes=self.config.max_scopes,
                    output=Path(self.config.output_dir)
                    / f"rank-{self.rank}.step-{step.serving_step}.eager.json",
                    max_metadata_bytes=self.config.max_metadata_bytes,
                    eager_step={
                        **step.identity(),
                        "profile_config_sha256": self.source_context[
                            "profile_config_sha256"
                        ],
                    },
                )
                try:
                    stack.enter_context(timeline)
                except NvtxUnavailableError as error:
                    step.unavailable(f"eager_native_nvtx_unavailable: {error}")
                    yield
                    return
                for module, layer_id, name in self.modules:
                    stack.enter_context(
                        session.watch_module(module, layer_id=layer_id, name=name)
                    )
                with session.iteration(iter_id=step.serving_step, metadata=metadata):
                    yield
            step.bind_eager_timeline(timeline)

    @contextmanager
    def iteration(self, *, step: int, identity: ProfileStep | None = None):
        previous = self.current_step
        self.current_step = identity
        if identity is None:
            try:
                yield
            finally:
                self.current_step = previous
            return
        if self.graph_enabled:
            try:
                yield
            finally:
                self.current_step = previous
            return
        # Preserve the established eager timeline entry while adding standard
        # module scopes. Graph mode never enters this synchronizing finalizer.
        if self._finished:
            yield
            self.current_step = previous
            return
        if self._timeline is None:
            hooks = ExitStack()
            try:
                for module, layer_id, name in self.modules:
                    hooks.enter_context(
                        self.session.watch_module(module, layer_id=layer_id, name=name)
                    )
                timeline = NvtxTimeline(
                    self.session,
                    costs=self.costs,
                    hardware=self.hardware,
                    stream=torch.cuda.current_stream(),
                    max_scopes=self.config.max_scopes,
                    output=self.output,
                    max_metadata_bytes=self.config.max_metadata_bytes,
                )
                timeline.__enter__()
            except BaseException:
                hooks.close()
                self.current_step = previous
                raise
            self._eager_hooks = hooks
            self._timeline = timeline
        error = None
        try:
            with self.session.iteration(iter_id=step, metadata=self.static_metadata):
                yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            self.current_step = previous
            if error is not None or self._window.selected >= self.config.num_steps:
                self._finished = True
                with allow_host_sync(
                    "finalize eager utilization metadata after its last batch"
                ):
                    self._timeline.__exit__(
                        type(error) if error is not None else None,
                        error,
                        error.__traceback__ if error is not None else None,
                    )
                self._eager_hooks.close()
