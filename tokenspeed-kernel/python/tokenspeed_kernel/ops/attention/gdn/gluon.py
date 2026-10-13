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

"""Registration shims for the AMD Gluon GDN decode / MTP-verify kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.gdn.decode import (
        launch_gluon_gdn_decode_gfx950,
    )

    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="gluon_gdn_decode_step_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "strided_qkv": frozenset({False, True}),
        },
    )
    def gluon_gdn_decode_step_gfx950(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        scale: float | None,
        output_state_indices: torch.Tensor | None,
        use_qk_l2norm: bool,
    ) -> torch.Tensor:
        """gfx950 ``gdn_decode_step``: the state goes to ``output_state_indices``
        when given, else back to ``initial_state_indices``."""
        return launch_gluon_gdn_decode_gfx950(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state=initial_state,
            read_indices=initial_state_indices,
            final_indices=(
                initial_state_indices
                if output_state_indices is None
                else output_state_indices
            ),
            step_states=None,
            step_indices=None,
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            use_qk_l2norm=use_qk_l2norm,
        )

    @register_kernel(
        "attention",
        "gdn_decode_mtp",
        name="gluon_gdn_decode_mtp_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "separate_output_state": frozenset({False, True}),
            "strided_qkv": frozenset({False, True}),
        },
    )
    def gluon_gdn_decode_mtp_gfx950(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        scale: float | None,
        disable_state_update: bool,
        use_qk_l2norm: bool,
        intermediate_states_buffer: torch.Tensor | None,
        output_state_indices: torch.Tensor | None,
        output_state: torch.Tensor | None,
        tree_ancestors: torch.Tensor | None,
        replay_payload: torch.Tensor | None,
    ) -> torch.Tensor:
        """gfx950 chain ``gdn_decode_mtp`` (draft trees run the Triton kernel)."""
        if tree_ancestors is not None or replay_payload is not None:
            raise NotImplementedError("the gfx950 GDN verify kernel follows a chain")
        # Per-token rows (in output_state, default the pool itself) or the
        # dense intermediate buffer; the read row takes the final state only
        # when neither applies and the update is enabled.
        step_states = None
        if output_state_indices is not None:
            step_states = initial_state if output_state is None else output_state
        elif intermediate_states_buffer is not None:
            step_states = intermediate_states_buffer.view(-1, *initial_state.shape[1:])
        write_back = not disable_state_update and output_state_indices is None
        return launch_gluon_gdn_decode_gfx950(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state=initial_state,
            read_indices=initial_state_indices,
            final_indices=initial_state_indices if write_back else None,
            step_states=step_states,
            step_indices=output_state_indices,
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            use_qk_l2norm=use_qk_l2norm,
        )
