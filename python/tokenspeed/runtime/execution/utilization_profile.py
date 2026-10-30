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

"""Bounded, eager utilization capture on the model's forward thread."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenspeed_kernel.op_profile import CostRegistry, ProfileSession
from tokenspeed_kernel.profile_costs import register_compute_costs
from tokenspeed_kernel.profile_nvtx import NvtxTimeline
from tokenspeed_kernel.profile_utilization import HardwareProfile

from tokenspeed.runtime.utils.host_sync import allow_host_sync

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UtilizationProfileConfig:
    """Explicit capture settings, loaded from TOKENSPEED_TIMELINE_CONFIG.

    Steps are one-based serving batches; startup warmup does not count.
    layers is a model submodule path to a ModuleList with global layer indices.
    hardware and library are paths to the hardware JSON and prebuilt NVTX bridge.
    output_dir receives rank-N.metrics.json; Nsight owns the .nsys-rep file.
    hbm_policy is 'unavailable' or the estimate 'logical_io_once'.
    """

    run_id: str
    output_dir: str
    hardware: str
    library: str
    layers: str
    start_step: int
    num_steps: int
    max_scopes: int
    hbm_policy: str

    def __post_init__(self) -> None:
        for name, value in (
            ("start_step", self.start_step),
            ("num_steps", self.num_steps),
            ("max_scopes", self.max_scopes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.run_id or not self.output_dir:
            raise ValueError("run_id and output_dir must be nonempty")


class RuntimeUtilizationProfile:
    """Collect opted-in ops and layers across a finite number of serving batches.

    Construction reads configuration and binds identities, without GPU work.
    iteration() must run on the execution stream and the same forward thread.
    The final selected batch writes metrics and submits native NVTX ranges before
    returning; no shutdown handler or post-runtime trace rewrite is required.
    Metrics cover registered calls only, not the model's uninstrumented work.
    """

    def __init__(
        self,
        config: UtilizationProfileConfig,
        *,
        model: torch.nn.Module,
        model_id: str,
        rank: int,
    ) -> None:
        self.config = config
        self.hardware = HardwareProfile.from_json(config.hardware)
        self.library = Path(config.library).resolve(strict=True)
        self.output = Path(config.output_dir) / f"rank-{rank}.metrics.json"
        self.session = ProfileSession(
            run_id=config.run_id, model_id=model_id, rank=rank, backend="nvtx"
        )
        layers = model.get_submodule(config.layers)
        if not isinstance(layers, torch.nn.ModuleList) or not len(layers):
            raise ValueError("layers must name a nonempty ModuleList of decoder layers")
        for layer_id, layer in enumerate(layers):
            self.session.bind_layer(layer, layer_id=layer_id)
        self.costs = CostRegistry()
        register_compute_costs(self.costs, hbm_policy=config.hbm_policy)
        self._timeline: NvtxTimeline | None = None
        self._finished = False

    @classmethod
    def from_env(
        cls,
        *,
        model: torch.nn.Module,
        model_id: str,
        rank: int,
        device: str,
        enforce_eager: bool,
        disable_prefill_graph: bool,
        speculative: bool,
    ) -> RuntimeUtilizationProfile | None:
        """Return a configured collector, or None when the environment is unset.

        Reject graph replay and draft execution before recording, because their
        call identities are not covered by this initial runtime integration.
        All paths in the JSON are relative to the process working directory.
        """
        path = os.environ.get("TOKENSPEED_TIMELINE_CONFIG")
        if not path:
            return None
        if torch.device(device).type != "cuda" or torch.version.hip is not None:
            raise ValueError("Native utilization timelines require NVIDIA CUDA")
        if not enforce_eager or not disable_prefill_graph:
            raise ValueError(
                "Native utilization timelines require --enforce-eager "
                "and --disable-prefill-graph"
            )
        if speculative:
            raise ValueError(
                "Native utilization timelines do not yet support draft models"
            )
        config = UtilizationProfileConfig(**json.loads(Path(path).read_text()))
        return cls(config, model=model, model_id=model_id, rank=rank)

    @contextmanager
    def iteration(self, *, step: int) -> Iterator[None]:
        """Record a selected serving batch and finalize on its last step or error."""
        if self._finished or step < self.config.start_step:
            yield
            return
        if self._timeline is None:
            if step != self.config.start_step:
                raise RuntimeError("The configured timeline start step was skipped")
            timeline = NvtxTimeline(
                self.session,
                costs=self.costs,
                hardware=self.hardware,
                stream=torch.cuda.current_stream(),
                max_scopes=self.config.max_scopes,
                library=self.library,
                output=self.output,
            )
            # Event preparation and final readback deliberately synchronize only
            # at the boundaries of this opt-in diagnostic window.
            with allow_host_sync("prepare utilization timeline CUDA events"):
                timeline.__enter__()
            self._timeline = timeline
            logger.info(f"Utilization timeline starts at serving step {step}")
        error = None
        try:
            with self.session.iteration(iter_id=step):
                yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            last_step = self.config.start_step + self.config.num_steps - 1
            if error is not None or step >= last_step:
                self._finished = True
                with allow_host_sync(
                    "finalize utilization timeline after its last batch"
                ):
                    self._timeline.__exit__(
                        type(error) if error is not None else None,
                        error,
                        error.__traceback__ if error is not None else None,
                    )
                if error is None:
                    logger.info(
                        f"Utilization timeline saved to {self.output}; "
                        "cost coverage is limited to opted-in registered calls"
                    )
