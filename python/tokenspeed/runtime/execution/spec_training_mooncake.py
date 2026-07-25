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

"""TorchSpec-compatible hidden-state export for offline TokenSpeed prefill."""

from __future__ import annotations

import os
import socket
import time

import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _parse_size(value: str, default: int) -> int:
    if not value:
        return default
    normalized = value.strip().upper()
    multipliers = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    for suffix, multiplier in multipliers.items():
        if normalized.endswith(suffix):
            return int(float(normalized[: -len(suffix)]) * multiplier)
    return int(normalized)


def _tensor_bytes(tensor: torch.Tensor, dtype: torch.dtype) -> bytes:
    cpu = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


class SpecTrainingMooncakeExporter:
    """Synchronously publish one whole-prompt prefill batch to Mooncake."""

    def __init__(self) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise RuntimeError(
                "--enable-spec-training-mooncake requires the mooncake Python package"
            ) from exc

        master = os.getenv("MOONCAKE_MASTER_SERVER")
        metadata = os.getenv("MOONCAKE_METADATA_SERVER")
        if not master or not metadata:
            raise RuntimeError(
                "--enable-spec-training-mooncake requires "
                "MOONCAKE_MASTER_SERVER and MOONCAKE_METADATA_SERVER"
            )

        protocol = os.getenv("MOONCAKE_PROTOCOL", "tcp")
        if protocol.lower() == "tcp":
            os.environ.setdefault("MC_STORE_MEMCPY", "0")

        self._store = MooncakeDistributedStore()
        result = self._store.setup(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", socket.gethostname()),
            metadata_server=metadata,
            global_segment_size=_parse_size(
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", ""),
                4 * 1024**3,
            ),
            local_buffer_size=_parse_size(
                os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", ""),
                512 * 1024**2,
            ),
            protocol=protocol,
            rdma_devices=os.getenv("MOONCAKE_DEVICE_NAME", ""),
            master_server_addr=master,
        )
        if result not in (None, 0):
            raise RuntimeError(f"Mooncake setup failed with error code {result}")

        self._store_full_codes = {
            int(code)
            for code in os.getenv("MOONCAKE_STORE_FULL_ERROR_CODES", "-200").split(",")
            if code.strip()
        }
        self._retry_wait = float(os.getenv("MOONCAKE_STORE_FULL_WAIT_SECONDS", "0.5"))
        self._max_wait = float(os.getenv("MOONCAKE_STORE_FULL_MAX_WAIT_SECONDS", "0"))
        logger.info("Initialized TorchSpec-compatible Mooncake hidden-state exporter")

    def export(self, ctx: ForwardContext, output: LogitsProcessorOutput) -> None:
        if not ctx.forward_mode.is_extend() or ctx.num_extends != ctx.bs:
            raise RuntimeError(
                "Spec-training Mooncake export only supports pure prefill batches"
            )
        if not ctx.request_ids or len(ctx.request_ids) != ctx.bs:
            raise RuntimeError("Missing request IDs for spec-training export")
        if not ctx.input_lengths_cpu or len(ctx.input_lengths_cpu) != ctx.bs:
            raise RuntimeError("Missing input lengths for spec-training export")
        if any(ctx.extend_prefix_lens_cpu or []):
            raise RuntimeError(
                "Spec-training Mooncake export does not support cached prefixes"
            )
        if output.hidden_states is None or output.last_hidden_states is None:
            raise RuntimeError(
                "Target model did not return auxiliary and final hidden states"
            )

        total_tokens = sum(ctx.input_lengths_cpu)
        if len(ctx.input_ids_cpu or []) != total_tokens:
            raise RuntimeError(
                "Prompt ID count does not match the captured prefill token count"
            )
        if output.hidden_states.shape[0] != total_tokens:
            raise RuntimeError(
                "Auxiliary hidden-state rows do not match the prompt token count"
            )
        if output.last_hidden_states.shape[0] != total_tokens:
            raise RuntimeError(
                "Final hidden-state rows do not match the prompt token count"
            )

        offset = 0
        for rid, length in zip(ctx.request_ids, ctx.input_lengths_cpu):
            end = offset + length
            ids = torch.tensor(
                ctx.input_ids_cpu[offset:end],
                dtype=torch.int64,
            )
            keys = [f"{rid}_hs", f"{rid}_ids", f"{rid}_lhs"]
            values = [
                _tensor_bytes(output.hidden_states[offset:end], torch.bfloat16),
                _tensor_bytes(ids, torch.int64),
                _tensor_bytes(output.last_hidden_states[offset:end], torch.bfloat16),
            ]
            self._put_batch(keys, values)
            offset = end

    def _put_batch(self, keys: list[str], values: list[bytes]) -> None:
        started = time.monotonic()
        while True:
            result = self._store.put_batch(keys, values)
            if result in (None, 0):
                return
            try:
                self._store.batch_remove(keys, force=True)
            except RuntimeError:
                logger.warning("Failed to clean up partial Mooncake export: %s", keys)
            if result not in self._store_full_codes:
                raise RuntimeError(
                    f"Mooncake put_batch failed for {keys} with error code {result}"
                )
            if self._max_wait > 0 and time.monotonic() - started >= self._max_wait:
                raise RuntimeError(f"Mooncake remained full while publishing {keys}")
            time.sleep(self._retry_wait)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
