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

"""FlashInfer Rubin decode with TokenSpeed's negative-index and PDL contract.

The installed upstream body and launcher remain authoritative. Only a uniform
CTA guard and dependency synchronization surround the body; no state storage
or copy of the third-party implementation is introduced.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cutlass
import cutlass.cute as cute
from tokenspeed_kernel.thirdparty.flashinfer.gdn_compat import compatibility_error
from tokenspeed_kernel.thirdparty.flashinfer.gdn_pdl import (
    _adapt_module,
    _merge_arguments,
    _PdlKernel,
    _PdlLaunch,
)

try:
    from flashinfer.gdn_kernels import gdn_decode_bf16_state_rubin as _upstream
except ImportError:
    _upstream = None


def is_available() -> bool:
    """Whether the installed FlashInfer includes the PR6604 decode entry."""
    return compatibility_error("decode") is None


@cute.kernel
def _gdn_masked_rubin_decode(
    body: cutlass.Constexpr,
    dynamic_args,
    constant_args: cutlass.Constexpr,
    constant_positions: cutlass.Constexpr,
    index_positions: cutlass.Constexpr,
    enable_pdl: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(enable_pdl):
        # Padding CTAs must signal too. The guard reads producer-owned indices.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    args = _merge_arguments(dynamic_args, constant_args, constant_positions)
    read_rows = args[index_positions[0]]
    write_rows = args[index_positions[1]]
    value_heads = args[index_positions[2]]
    value_tiles = args[index_positions[3]]
    row = cute.arch.block_idx()[0] // (value_heads * value_tiles)
    # PR6604 clamps -1 to row 0, but TokenSpeed's null state must stay zero.
    # All threads in this CTA take the same branch, including upstream barriers.
    if read_rows[row] >= 0 and write_rows[row] >= 0:
        body(*args)


class _MaskedRubinKernel(_PdlKernel):
    def __init__(self, kernel, *, enable_pdl: bool):
        super().__init__(kernel)
        self.enable_pdl = enable_pdl
        names = tuple(self.signature.parameters)
        self.index_positions = tuple(
            names.index(name)
            for name in ("h0_indices", "h0_out_indices", "HV", "num_v_tiles")
        )

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.kwargs:
            raise TypeError("Rubin decode requires a positional device-kernel ABI")
        dynamic = tuple(
            value
            for i, value in enumerate(bound.args)
            if i not in self.constant_positions
        )
        constants = tuple(bound.args[i] for i in self.constant_positions)
        launch = _gdn_masked_rubin_decode(
            self.body,
            dynamic,
            constants,
            self.constant_positions,
            self.index_positions,
            self.enable_pdl,
        )
        return _PdlLaunch(launch) if self.enable_pdl else launch


def _build_masked_kernel(
    module_name, kernel_name, compile_fn, *, extra_key_files, enable_pdl: bool
):
    return _upstream.build_and_load_cute_dsl_kernel(
        f"tokenspeed_masked_pdl{int(enable_pdl)}_{module_name}",
        kernel_name,
        compile_fn,
        extra_key_files=(
            *extra_key_files,
            __file__,
            str(Path(__file__).with_name("gdn_pdl.py")),
        ),
    )


@functools.cache
def _runner(enable_pdl: bool):
    reason = compatibility_error("decode")
    if reason is not None:
        raise RuntimeError(reason)
    namespace = _adapt_module(
        _upstream,
        kernels=(),
        launchers=("_run_t1_rubin",),
        entrypoints=("_build_t1_rubin", "gated_delta_rule_t1_rubin"),
        caches=("_compiled_t1_rubin", "_launch_t1_rubin", "_default_indices"),
        overrides={
            "gdn_t1_rubin_kernel": _MaskedRubinKernel(
                _upstream.gdn_t1_rubin_kernel, enable_pdl=enable_pdl
            ),
            "build_and_load_cute_dsl_kernel": functools.partial(
                _build_masked_kernel, enable_pdl=enable_pdl
            ),
        },
    )
    return namespace["gated_delta_rule_t1_rubin"]


def gated_delta_rule_t1_rubin(*, enable_pdl: bool, **kwargs):
    """Forward upstream T=1 inputs; skip negative read/write rows.

    Args:
        enable_pdl: Capture-time dependent-launch selection.
        **kwargs: Upstream Q/K/V, gates, K-last state pool, read/write indices,
            normalization and scale arguments, forwarded without changes.

    Returns:
        BF16 output ``[B, 1, HV, 128]``. Padding output is undefined; every
        nonnegative destination belonging to a live row is updated in place.
    """
    return _runner(enable_pdl)(**kwargs)
