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

"""Registers the AMD Gluon Kimi-K3 MoE layer tail (gfx950)."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.kimi_k3 import moe_tail as _moe_tail
except ImportError:
    _moe_tail = None


if _moe_tail is not None:
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    # The sizes the kernel's tiles and work split are written for: Kimi-K3 at
    # TP8. ``rows`` is a compile-time row count, one request's decode or
    # verify.
    _TRAITS = {
        "rows": frozenset(range(1, _moe_tail.MAX_ROWS + 1)),
        "hidden_size": frozenset({_moe_tail.HIDDEN}),
        "tp_size": frozenset({_moe_tail.TP}),
        "num_heads": frozenset({96}),
        "head_dim": frozenset({128}),
        "num_experts": frozenset({896}),
        "top_k": frozenset({16}),
        "latent_size": frozenset({_moe_tail.LATENT}),
        "intermediate_size": frozenset({3072}),
        "shared_intermediate_size": frozenset({6144}),
        "weight_dtype": frozenset({"mxfp4"}),
        # The routing and activation the kernel computes.
        "num_expert_groups": frozenset({1}),
        "topk_groups": frozenset({1}),
        "scoring_func": frozenset({"sigmoid"}),
        "routed_scaling_factor": frozenset({1.0}),
        "renormalize": frozenset({True}),
        "situ_beta": frozenset({4.0}),
        "situ_linear_beta": frozenset({25.0}),
        # Each of the 256 workgroups needs a compute unit to itself, all at
        # once (a partitioned GPU has fewer).
        "compute_units": frozenset(range(256, 1025)),
    }
    _SIGNATURES = format_signatures("hidden_states", "dense", {torch.bfloat16})

    # How long one wait inside a launch may stall before the launch stops
    # with an error, in seconds.
    _WAIT_DEADLINE_S = 60.0
    # The waits' clock (s_memrealtime) runs at a constant 100 MHz; a wait
    # compares times in units of 256 ticks as 32-bit values, which bounds the
    # deadline.
    _CLOCK_HZ = 100_000_000
    _MAX_DEADLINE_TICKS = (1 << 31 << 8) - 1
    # The step words: word 0 counts steps modulo _STEP_PERIOD, words 1-2 hold
    # the wait deadline in clock ticks, words 3-5 the sticky errors a failed
    # wait sets.
    _STEP_WORDS = 8
    _DEADLINE_WORD = 1
    # The step count wraps at this period, so it never overflows int32. The
    # kernel reads the count only modulo 2**22, for the handoff tag
    # (moe_tail/_stages/common.py, step_tag), and modulo 2, for the peer-buffer
    # half. A period that is a multiple of 2**22 changes neither.
    _STEP_PERIOD = 1 << 22
    # A layer's handoffs are tagged with its slot, apart from other layers'.
    _SLOT_BASE = 128

    class _Workspace:
        """This rank's buffers that every layer's launch shares."""

        def __init__(self, *, rank: int, cpu_group: Any, device: torch.device) -> None:
            self.rank = rank
            self.cpu_group = cpu_group
            # The layer slots whose launch every rank has compiled.
            self.compiled: set[int] = set()
            # Uncached, so a consumer reads a handoff at its home memory, not
            # through the producer's L2. Launches run one after another, so
            # one scratch serves every layer.
            scratch_bytes = -(-_moe_tail.scratch_alloc_bytes() // 256) * 256
            self._scratch = _moe_tail.PeerBuffer(scratch_bytes, None, rank, 1, device)
            self.scratch = self._scratch.bytes
            self.scratch.zero_()
            # Every handoff's tag derives from the step count, so no buffer is
            # cleared between steps.
            self.steps = torch.zeros(_STEP_WORDS, dtype=torch.int32, device=device)
            ticks = int(_WAIT_DEADLINE_S * _CLOCK_HZ)
            assert 0 < ticks <= _MAX_DEADLINE_TICKS
            low = ticks & 0xFFFFFFFF
            self.steps[_DEADLINE_WORD] = low - (1 << 32) if low >= 1 << 31 else low
            self.steps[_DEADLINE_WORD + 1] = ticks >> 32
            # The ranks reduce through every peer's memory: one uncached buffer
            # a rank, every rank holding all the group's addresses.
            self._peer = _moe_tail.PeerBuffer(
                _moe_tail.peer_alloc_bytes(), cpu_group, rank, _moe_tail.TP, device
            )
            _moe_tail.init_peer_bytes(self._peer.bytes)
            self.peers = self._peer.addresses
            # The residual prefix inside a launch; layers run in sequence.
            self.prefix = torch.empty(
                _moe_tail.MAX_ROWS,
                _moe_tail.HIDDEN,
                dtype=torch.bfloat16,
                device=device,
            )
            # Every rank's buffers hold their start values before any peer's
            # first launch writes into them.
            torch.cuda.synchronize(device)
            dist.barrier(group=cpu_group)

    @dataclass(frozen=True)
    class _LayerOptions:
        build: Any
        slot: int
        starts_step: bool

    @register_kernel(
        "kimi_k3",
        "moe_tail_prepare",
        name="gluon_kimi_k3_moe_tail_prepare_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        traits=_TRAITS,
        priority=Priority.SPECIALIZED,
    )
    def gluon_kimi_k3_moe_tail_prepare_gfx950(
        *, shape: Any, rank: int, cpu_group: Any, device: torch.device
    ) -> _Workspace:
        del shape
        return _Workspace(rank=rank, cpu_group=cpu_group, device=device)

    # The weights by role, as the kernel names them, with their dtypes and
    # this rank's shapes.
    _bf16, _fp32, _u8 = torch.bfloat16, torch.float32, torch.uint8
    _H, _L, _TP = _moe_tail.HIDDEN, _moe_tail.LATENT, _moe_tail.TP
    _WEIGHTS = {
        "o_proj": ("w_o", _bf16, (_H, 1536)),
        "mlp_res_norm": ("ares_nw", _bf16, (_H,)),
        "mlp_res_proj": ("ares_qk", _bf16, (_H,)),
        "post_attention_norm": ("in_nw", _bf16, (_H,)),
        "router": ("w_gate", _bf16, (896, _H)),
        "router_bias": ("bias", _fp32, (896,)),
        "latent_down": ("w_ld", _bf16, (_L, _H)),
        "latent_norm": ("ln_w", _bf16, (_L,)),
        # Replicated; the tail reads this rank's rows of it.
        "latent_up": ("w_up", _bf16, (_H, _L)),
        "shared_gate_up": ("w_sgu", _bf16, (2 * 768, _H)),
        "shared_down": ("w_sd", _bf16, (_H, 768)),
        # The routed experts' MXFP4 banks in the N16 layout: two FP4 values a
        # byte and one E8M0 scale per 32 values, the down projection's 12
        # scale columns padded to 16.
        "w13": ("w13", _u8, (896, 2 * 384, _L // 2)),
        "w13_scale": ("w13s", _u8, (896 * 2 * 384, _L // 32)),
        "w2": ("w2", _u8, (896, _L, 384 // 2)),
        "w2_scale": ("w2s", _u8, (896 * _L, 16)),
    }
    # The MoE layer stores the same N16 bytes in cells
    # (``n16_mxfp4_shape``); the tail reads them through views of its shapes.
    _N16_CELLS = {
        "w13": (896, 384 // 8, _L // 128, 4, 16, 16),
        "w13_scale": (896, 384 // 16, _L // 256, 4, 16, 4),
        "w2": (896, _L // 16, 8 * 384),
        "w2_scale": (896, _L // 32, 2, 4, 16, 4),
    }

    @register_kernel(
        "kimi_k3",
        "moe_tail_pack",
        name="gluon_kimi_k3_moe_tail_pack_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        traits=_TRAITS,
        priority=Priority.SPECIALIZED,
    )
    def gluon_kimi_k3_moe_tail_pack_gfx950(
        *,
        workspace: _Workspace,
        shape: Any,
        layer_index: int,
        starts_step: bool,
        num_blocks: int,
        restarts_prefix: bool,
        eps: dict[str, float],
        tensors: dict[str, torch.Tensor],
        writes_next_mix: bool,
    ) -> tuple[dict[str, torch.Tensor], _LayerOptions]:
        if set(tensors) != set(_WEIGHTS):
            raise ValueError(
                f"the tail takes the weights {sorted(_WEIGHTS)}, not {sorted(tensors)}"
            )
        weights, problems = {}, []
        for role, (name, dtype, shape_) in _WEIGHTS.items():
            tensor = tensors[role].detach()
            shapes = {shape_, _N16_CELLS.get(role, shape_)}
            if (
                tensor.dtype != dtype
                or tuple(tensor.shape) not in shapes
                or not tensor.is_contiguous()
            ):
                problems.append(
                    f"{role} is {tensor.dtype} {tuple(tensor.shape)}"
                    f"{'' if tensor.is_contiguous() else ' strided'}, not "
                    f"contiguous {dtype} {' or '.join(map(str, sorted(shapes)))}"
                )
            else:
                weights[name] = tensor.view(shape_)
        if problems:
            raise ValueError("; ".join(problems))
        # This rank's rows of the replicated up projection: a view, since
        # narrowing the first dimension of a contiguous weight keeps it so.
        rows_per_rank = _H // _TP
        weights["w_up"] = weights["w_up"].narrow(
            0, workspace.rank * rows_per_rank, rows_per_rank
        )
        build = _moe_tail.KimiK3MoeTailBuild(
            tokens=shape.rows,
            nblocks=num_blocks,
            reset=restarts_prefix,
            eps=eps["mlp_res"],
            out_eps=eps["post_attention"],
            ln_eps=eps["latent"],
            next_mix=writes_next_mix,
            mix_eps=eps["next_mix"] if writes_next_mix else None,
        )
        slot = _SLOT_BASE + layer_index
        if not _SLOT_BASE <= slot < 2 * _SLOT_BASE:
            raise ValueError(f"layer {layer_index} has no tail slot")
        return weights, _LayerOptions(build=build, slot=slot, starts_step=starts_step)

    def _check_rows(tensor: torch.Tensor, rows: int, width: int, what: str) -> None:
        if (
            tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) != (rows, width)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{what} must be a contiguous BF16 [{rows}, {width}] tensor, not "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )

    @register_kernel(
        "kimi_k3",
        "moe_tail",
        name="gluon_kimi_k3_moe_tail_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        traits={**_TRAITS, "expert_layout": frozenset({_moe_tail.EXPERT_LAYOUT})},
        priority=Priority.SPECIALIZED,
    )
    def gluon_kimi_k3_moe_tail_gfx950(
        workspace: _Workspace,
        weights: dict[str, torch.Tensor],
        options: _LayerOptions,
        *,
        attn_output: torch.Tensor,
        stream: torch.Tensor | None,
        block_residual: torch.Tensor,
        out: torch.Tensor,
        next_mix: Any,
    ) -> None:
        build = options.build
        rows = build.tokens
        # Every tensor reaches the kernel as an address, so a tensor of the
        # wrong dtype or layout would be read as if it were right.
        _check_rows(attn_output, rows, 1536, "the attention output")
        _check_rows(out, rows, _H, "the output")
        if (stream is None) != build.reset:
            raise ValueError(
                "a layer that restarts the prefix takes no incoming stream, and "
                "every other layer takes one"
            )
        if stream is not None:
            _check_rows(stream, rows, _H, "the incoming stream")
        if (
            block_residual.dtype != torch.bfloat16
            or block_residual.dim() != 3
            or block_residual.shape[0] < build.nblocks
            or tuple(block_residual.shape[1:]) != (rows, _H)
            or block_residual.stride(2) != 1
        ):
            raise ValueError(
                f"the snapshots must be BF16 [>= {build.nblocks}, {rows}, {_H}] with "
                f"the hidden dimension contiguous, not {block_residual.dtype} "
                f"{tuple(block_residual.shape)}"
            )
        if (next_mix is not None) != build.next_mix:
            raise ValueError(
                "the next layer's input is written exactly when the layer was "
                "packed to write it"
            )
        if next_mix is not None:
            for tensor, what in (
                (next_mix.score_weight, "the next mix's score weight"),
                (next_mix.input_norm, "the next layer's input norm weight"),
            ):
                if (
                    tensor.dtype != torch.bfloat16
                    or tuple(tensor.shape) != (_H,)
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{what} must be a contiguous BF16 [{_H}] tensor, not "
                        f"{tensor.dtype} {tuple(tensor.shape)}"
                    )
            _check_rows(next_mix.out, rows, _H, "the next layer's input")
            mix_wp, mix_nw, mixed = (
                next_mix.score_weight,
                next_mix.input_norm,
                next_mix.out,
            )
            mix_m, mix_s, mix_acc = next_mix.partial
        else:
            mix_wp = mix_nw = mixed = mix_m = mix_s = mix_acc = None
        if options.starts_step:
            # Inside the CUDA graph, before the step's first launch. The wrap
            # is safe: see _STEP_PERIOD.
            workspace.steps[:1].add_(1).bitwise_and_(_STEP_PERIOD - 1)
        prefix = workspace.prefix[:rows]
        launch = functools.partial(
            _moe_tail.launch_gluon_kimi_k3_moe_tail_gfx950,
            build,
            core=attn_output,
            prefix=prefix,
            # The incoming stream is read in place; a layer that restarts the
            # prefix reads the launch's own prefix.
            prefix_in=prefix if stream is None else stream,
            # The kernel reads token t of block j at
            # ``base + t * stride(0) + j * stride(1)``: the token-major view of
            # the block-major snapshots, the same storage.
            blocks=block_residual.transpose(0, 1),
            output=out,
            scratch=workspace.scratch,
            peers=workspace.peers,
            epoch=workspace.steps,
            rank=workspace.rank,
            slot=options.slot,
            mix_wp=mix_wp,
            mix_nw=mix_nw,
            mix_m=mix_m,
            mix_s=mix_s,
            mix_acc=mix_acc,
            mixed=mixed,
            **{name: weights[name] for name, _, _ in _WEIGHTS.values()},
        )
        if options.slot not in workspace.compiled:
            # A launch waits about a minute for its peers' values, and a first
            # launch compiles for longer on some ranks than on others: every
            # rank compiles, then the ranks meet, then they launch.
            launch(compile_only=True)
            dist.barrier(group=workspace.cpu_group)
            workspace.compiled.add(options.slot)
        launch(compile_only=False)
