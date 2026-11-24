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

"""Explicit fixed-capacity buffers shared across serial model layers."""

import logging
import os

import torch
import torch.distributed as dist

_BUFFERS = {}
_WIRE1_TRANSPORTS = {}
# Existing Decode configurations retain their original allocation.
CAPACITY = int(os.environ.get("TOKENSPEED_SM120_DEEPEP_CAPACITY", "128"))
if CAPACITY <= 0:
    raise ValueError("TOKENSPEED_SM120_DEEPEP_CAPACITY must be positive")


def nic_for_visible_device(nic_list: str, device_index: int, device_count: int) -> str:
    """Select one NIC from a list ordered like this process's visible GPUs.

    A supplied list must name every visible device, including peers whose
    transport may be created later. CUDA device ordinals are process-local.
    """
    nics = [item.strip() for item in nic_list.split(",")]
    if device_count < 1 or len(nics) != device_count or any(not item for item in nics):
        raise ValueError(
            "SM120 DeepEP NIC list must name every visible GPU with a nonempty entry "
            f"(got {len(nics)} entries for {device_count} devices)"
        )
    if not 0 <= device_index < device_count:
        raise ValueError("SM120 DeepEP NIC selection requires a visible CUDA device")
    return nics[device_index]


def configure_device_nic(wire: str) -> str | None:
    """Apply an explicit per-device NIC map, or retain vendor auto-selection.

    The old wire1-only variable remains an alias on that transport; the common
    variable wins when both are present. Do this before creating vendor state.
    """
    nic_list = os.environ.get("TOKENSPEED_SM120_DEEPEP_NICS")
    if nic_list is None and wire == "nvfp4_fp8":
        nic_list = os.environ.get("TOKENSPEED_WIRE1_NICS")
    if nic_list is None:
        return None
    nic = nic_for_visible_device(
        nic_list, torch.cuda.current_device(), torch.cuda.device_count()
    )
    os.environ["EP_NIC_NAME"] = nic
    os.environ["NVSHMEM_HCA_LIST"] = f"{nic}:1"
    return nic


def get_dispatch_capacity(buffer, max_tokens, offset, *, prefill):
    """Keep wire1 metadata layout fixed across a decode phase's chunks.

    The caller still slices actual rows at CAPACITY. All ranks supply the
    same phase bound and offset, including ranks with no source rows.
    Prefill and Elastic retain their existing chunk-capacity rules.
    """
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.low_latency_wire1 import (
        LowLatencyWire1Buffer,
    )

    if not 0 <= offset < max_tokens:
        raise ValueError("Dispatch offset must name a nonempty logical chunk")
    capacity = min(CAPACITY, max_tokens - offset)
    if isinstance(buffer, LowLatencyWire1Buffer) and not prefill:
        capacity = min(buffer.capacity, max_tokens)
    if prefill:
        capacity = min(CAPACITY, 1 << (capacity - 1).bit_length())
    return capacity


def get_buffer(group, hidden, topk, experts):
    wire = os.environ.get("TOKENSPEED_SM120_DEEPEP_WIRE", "elastic_fp8_bf16")
    if wire not in ("elastic_fp8_bf16", "nvfp4_fp8"):
        raise ValueError(f"Unknown DeepEP wire mode: {wire}")
    configure_device_nic(wire)
    key = (id(group), hidden, topk, experts, wire)
    if wire == "nvfp4_fp8":
        from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.low_latency_wire1 import (
            LowLatencyWire1Buffer,
        )

        # Target and draft may own distinct ProcessGroup objects with identical
        # ordered members. Their serial execution shares one legacy RDMA team.
        key = (
            tuple(dist.get_process_group_ranks(group)),
            dist.get_backend(group),
            torch.cuda.current_device(),
            hidden,
            topk,
            experts,
            wire,
        )
        if key not in _BUFFERS:
            transport_key = key[:3]
            owner = _WIRE1_TRANSPORTS.get(transport_key)
            _BUFFERS[key] = LowLatencyWire1Buffer(
                group, hidden, experts, CAPACITY, shared_owner=owner
            )
            _WIRE1_TRANSPORTS.setdefault(transport_key, _BUFFERS[key])
            logging.getLogger(__name__).warning(
                "SM120 DeepEP wire1 NVFP4 dispatch / FP8 combine: experts=%s capacity=%s",
                experts,
                CAPACITY,
            )
        return _BUFFERS[key]
    if key not in _BUFFERS:
        from deep_ep import ElasticBuffer

        _BUFFERS[key] = ElasticBuffer(
            group,
            num_max_tokens_per_rank=CAPACITY,
            hidden=hidden,
            num_topk=topk,
            use_fp8_dispatch=True,
            allow_hybrid_mode=False,
            num_allocated_qps=8,
            sl_idx=1,
            num_cpu_timeout_secs=120,
            num_gpu_timeout_secs=60,
            explicitly_destroy=True,
        )
        logging.getLogger(__name__).warning(
            "SM120 DeepEP experimental buffer: experts=%s topk=%s capacity=%s",
            experts,
            topk,
            CAPACITY,
        )
    return _BUFFERS[key]
