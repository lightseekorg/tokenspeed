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

"""Optional K3 TP8 NVFP4 down-projection input preparation.

A prepared op owns communication storage; each call returns a borrowed
Nvfp4Activation for immediate, same-stream MoE consumption.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe.activation import Nvfp4Activation
from tokenspeed_kernel.ops.moe.latent_down import KimiK3LatentDownOp
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

logger = logging.getLogger(__name__)


def _mailbox_geometry(m):
    if m <= 4:
        return 8, 128
    if m <= 8:
        return 16, 128
    if m <= 32:
        return 64, 128
    if m <= 64:
        return 128, 128
    return 608, 256


@register_kernel(
    "moe",
    "quantize_latent_input",
    name="cute_dsl_nvfp4_latent_input",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    priority=Priority.SPECIALIZED,
)
def _quantize_latent_input(source, data, scales, scale, signals, **kwargs):
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import launch

    launch(source, data, scales, scale, signals, **kwargs)


@register_kernel(
    "moe",
    "quantize_latent_input_cooperative",
    name="cute_dsl_nvfp4_latent_input_cooperative",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    priority=Priority.SPECIALIZED,
)
def _quantize_latent_input_cooperative(source, data, scales, scale, **kwargs):
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input_cooperative import (
        launch,
    )

    launch(source, data, scales, scale, **kwargs)


def _auto_geometry(m):
    """Fixed launch bands, not a per-width benchmark lookup table."""
    if not 1 <= m <= 1280:
        raise ValueError("automatic NVFP4 preparation supports M=1..1280")
    if m <= 8:
        return False, 8, 256, 128
    if m <= 32:
        return True, 2, 608, 128
    if m <= 64:
        return True, 4, 608, 128
    if m <= 128:
        return True, 8, 608, 128
    return False, 16, 608, 256


def fusion_mode() -> str:
    """Read the startup-only experimental setting (default: no behaviour change)."""
    mode = os.environ.get("TOKENSPEED_K3_DOWN_NVFP4_FUSION", "off")
    if mode not in ("off", "mailbox", "all", "auto"):
        raise ValueError(
            "TOKENSPEED_K3_DOWN_NVFP4_FUSION must be off, mailbox, all or auto"
        )
    return mode


@dataclass
class _Workspace:
    data: torch.Tensor
    scales: torch.Tensor
    signals: torch.Tensor
    handles: tuple
    data_mc: int
    scale_mc: int


class KimiK3Nvfp4DownOp:
    _workspaces: dict[tuple, _Workspace] = {}

    def __init__(self, mailbox, scale, workspace, max_m, mode, group, ctas):
        self.mailbox = mailbox
        self.scale = scale
        self.workspace = workspace
        self.max_m = max_m
        self.mode = mode
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.hidden = mailbox.shard_dim * self.world
        self.ctas = ctas
        self.kernel = select_kernel(
            "moe",
            "quantize_latent_input",
            format_signature(x=dense_tensor_format(torch.bfloat16)),
            solution="cute_dsl",
        )
        self.cooperative_kernel = select_kernel(
            "moe",
            "quantize_latent_input_cooperative",
            format_signature(x=dense_tensor_format(torch.bfloat16)),
            solution="cute_dsl",
        )
        self.use_pdl = pdl_enabled()

    @classmethod
    def initialize(
        cls,
        mailbox: KimiK3LatentDownOp | None,
        scale: torch.Tensor,
        *,
        group: dist.ProcessGroup,
        max_m: int,
        mode: str,
    ) -> KimiK3Nvfp4DownOp | None:
        """Prepare and agree a TP8 op before graph capture or forward execution.

        Args:
            mailbox: Existing #1383 producer/mailbox bundle, or None for fallback.
            scale: Receiver's scalar FP32 encoding multiplier (already processed).
            group: The TP group whose eight ranks own the latent columns.
            max_m: Largest live row count supported by the experiment.
            mode: Frozen startup setting, off/mailbox/all/auto. Auto uses new
                separate at M<=8, new fused at M=9..128, and old separate at
                M=129..1280; larger widths retain the original path.

        Returns:
            Prepared op, or None when the group cannot safely use the fusion.
        """
        if mode == "off":
            return None
        if mode not in ("mailbox", "all", "auto"):
            raise ValueError("unknown NVFP4 down fusion mode")
        if not current_platform().is_nvidia or dist.get_world_size(group) != 8:
            return None
        configuration = torch.tensor(
            [{"mailbox": 1, "all": 2, "auto": 3}[mode], max_m],
            dtype=torch.int64,
            device=scale.device,
        )
        configurations = [torch.empty_like(configuration) for _ in range(8)]
        dist.all_gather(configurations, configuration, group=group)
        if any(not torch.equal(configuration, other) for other in configurations):
            raise ValueError(
                "NVFP4 fusion mode and capacity must agree across TP ranks"
            )
        special_math = any(
            os.environ.get(name, "0") not in ("0", "", "false", "False")
            for name in (
                "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
                "FLASHINFER_NVFP4_4OVER6",
            )
        )
        eligible = mailbox is not None and not special_math and max_m > 0
        vote = torch.tensor(int(eligible), dtype=torch.int32, device=scale.device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        if not vote.item():
            logger.info(
                "K3 down NVFP4 fusion declined: mailbox or quantization recipe unsupported"
            )
            return None
        scale_values = [torch.empty_like(scale) for _ in range(8)]
        dist.all_gather(scale_values, scale, group=group)
        stacked = torch.stack(scale_values)
        if not bool(torch.isfinite(stacked).all() and (stacked > 0).all()):
            raise ValueError("NVFP4 encoding multipliers must be finite and positive")
        auto_dispatch = mode == "auto"
        if auto_dispatch:
            # Auto keeps the existing BF16 producer and mailbox capacity.
            # Forced "all" retains the separate large multicast experiment.
            mode = "mailbox"
        if mode == "all" and not bool((stacked == stacked[0]).all()):
            logger.info(
                "K3 down NVFP4 fusion: unequal TP multipliers, using mailbox only"
            )
            mode = "mailbox"
        max_m = max(max_m, mailbox.max_m) if mode == "all" else mailbox.max_m
        hidden = mailbox.shard_dim * 8
        if mailbox.shard_dim % 64:
            return None
        ctas = min(
            152, torch.cuda.get_device_properties(scale.device).multi_processor_count
        )
        key = (group.group_name, scale.device.index, id(mailbox._slot), max_m, mode)
        if key not in cls._workspaces:
            cls._workspaces[key] = cls._build_workspace(
                group, scale.device, max_m, hidden, mode, ctas
            )
        op = cls(mailbox, scale, cls._workspaces[key], max_m, mode, group, ctas)
        if auto_dispatch:
            op.mode = "auto"
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import (
            MAILBOX,
            MULTICAST,
            PLAIN,
            compile_kernel,
        )

        if auto_dispatch:
            from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input_cooperative import (
                compile_kernel as compile_cooperative,
            )

            for rows in (1, 9, 33, 65):
                fused, values, grid, threads = _auto_geometry(rows)
                compile_cooperative(
                    hidden,
                    values,
                    grid,
                    threads,
                    fused,
                    scale.device.index,
                    op.use_pdl,
                )
            compile_kernel(
                hidden,
                PLAIN,
                0,
                1,
                608,
                256,
                scale.device.index,
                op.use_pdl,
            )
        for rows in (() if auto_dispatch else (1, 5, 9, 33, 65)):
            mailbox_ctas, mailbox_threads = _mailbox_geometry(rows)
            compile_kernel(
                hidden,
                MAILBOX,
                0,
                1,
                mailbox_ctas,
                mailbox_threads,
                scale.device.index,
                pdl_enabled(),
            )
        if mode == "all":
            compile_kernel(
                hidden,
                MULTICAST,
                op.rank,
                8,
                ctas,
                512,
                scale.device.index,
                pdl_enabled(),
            )
        logger.info(
            "K3 down NVFP4 fusion prepared: mode=%s rank=%d max_m=%d",
            op.mode,
            op.rank,
            max_m,
        )
        return op

    @staticmethod
    def _build_workspace(group, device, max_m, hidden, mode, ctas):
        from torch.distributed import _symmetric_memory as symm_mem

        # Linear SF's backing allocation retains FlashInfer's 16-row padding.
        padded_m = (max_m + 15) // 16 * 16
        with torch.inference_mode(False), torch.no_grad():
            if mode == "all":
                data = symm_mem.empty(
                    (max_m, hidden // 2), dtype=torch.uint8, device=device
                )
                scales = symm_mem.empty(
                    (padded_m, hidden // 16), dtype=torch.uint8, device=device
                )
                flags = symm_mem.empty((ctas * 8,), dtype=torch.int32, device=device)
                flags.zero_()
                data_handle = symm_mem.rendezvous(data, group)
                scale_handle = symm_mem.rendezvous(scales, group)
                flag_handle = symm_mem.rendezvous(flags, group)
                if not data_handle.multicast_ptr or not scale_handle.multicast_ptr:
                    raise RuntimeError("NVFP4 output multicast rendezvous failed")
                signals = torch.tensor(
                    flag_handle.buffer_ptrs, dtype=torch.int64, device=device
                )
                dist.barrier(group=group)
                return _Workspace(
                    data,
                    scales,
                    signals,
                    (data_handle, scale_handle, flag_handle, flags),
                    data_handle.multicast_ptr,
                    scale_handle.multicast_ptr,
                )
            data = torch.empty((max_m, hidden // 2), dtype=torch.uint8, device=device)
            scales = torch.empty(
                (padded_m, hidden // 16), dtype=torch.uint8, device=device
            )
            return _Workspace(
                data,
                scales,
                torch.empty((1,), dtype=torch.int64, device=device),
                (),
                0,
                0,
            )

    def handles(self, m: int) -> bool:
        """Whether the next forward uses a validated-capacity fusion path."""
        if not 1 <= m <= self.max_m:
            return False
        if self.mode == "auto":
            return m <= 1280
        return True

    def _prepare_auto(self, hidden_states, weight):
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import (
            PLAIN,
        )

        m = hidden_states.shape[0]
        fused, values, ctas, threads = _auto_geometry(m)
        if fused:
            slot = self.mailbox._slot
            slot.gemm_by_m[m](hidden_states, weight, slot.mailbox, slot.multicast_ptr)
            source = slot.mailbox
        else:
            source = self.mailbox(hidden_states, weight)
        data = self.workspace.data[:m]
        scales = self.workspace.scales[:m]
        if values == 16:
            self.kernel(
                source,
                data,
                scales,
                self.scale,
                self.workspace.signals[:1],
                hidden=self.hidden,
                m=m,
                mode=PLAIN,
                rank=0,
                world=1,
                data_mc=0,
                scale_mc=0,
                ctas=ctas,
                threads=threads,
            )
        else:
            self.cooperative_kernel(
                source,
                data,
                scales,
                self.scale,
                hidden=self.hidden,
                m=m,
                values=values,
                ctas=ctas,
                threads=threads,
                mailbox=fused,
                use_pdl=self.use_pdl,
            )
        return Nvfp4Activation(
            data, scales.view(torch.float8_e4m3fn), (m, self.hidden), self.scale
        )

    def __call__(
        self, hidden_states: torch.Tensor, weight: torch.Tensor
    ) -> Nvfp4Activation:
        """Project a TP8 weight shard and prepare its quantized MoE input.

        Args:
            hidden_states: Replicated BF16 [M, hidden] source.
            weight: This rank's BF16 [448, hidden] projection rows.

        Returns:
            Borrowed packed activation, valid until this workspace is reused.
        """
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import (
            MAILBOX,
            MULTICAST,
        )

        m = hidden_states.shape[0]
        if not self.handles(m):
            raise ValueError("NVFP4 down input is outside the prepared fusion widths")
        if self.mode == "auto":
            return self._prepare_auto(hidden_states, weight)
        if self.mailbox.handles(m):
            slot = self.mailbox._slot
            slot.gemm_by_m[m](hidden_states, weight, slot.mailbox, slot.multicast_ptr)
            source, mode, rank, world = slot.mailbox, MAILBOX, 0, 1
            ctas, threads = _mailbox_geometry(m)
        else:
            source = torch.mm(hidden_states, weight.t())
            mode, rank, world = MULTICAST, self.rank, self.world
            ctas, threads = self.ctas, 512
        ws = self.workspace
        data, scales = ws.data[:m], ws.scales[:m]
        signals = ws.signals if mode == MULTICAST else ws.signals[:1]
        self.kernel(
            source,
            data,
            scales,
            self.scale,
            signals,
            hidden=self.hidden,
            m=m,
            mode=mode,
            rank=rank,
            world=world,
            data_mc=ws.data_mc,
            scale_mc=ws.scale_mc,
            ctas=ctas,
            threads=threads,
        )
        return Nvfp4Activation(
            data, scales.view(torch.float8_e4m3fn), (m, self.hidden), self.scale
        )


__all__ = ["KimiK3Nvfp4DownOp", "fusion_mode"]
