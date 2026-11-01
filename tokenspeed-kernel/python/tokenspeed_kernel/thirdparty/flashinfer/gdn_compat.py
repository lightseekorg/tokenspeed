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

"""Checked upstream source contract for the Rubin GDN adapters.

The negative-row guard depends on CTA indexing, not just Python signatures.
Only audited release sources may enter the adapted launchers. Check once before
kernel selection, never hash files on the forward path.
"""

import functools
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Literal

_VERSION = "0.7.2.post1"
# SHA256 of the official PyPI wheel sources. Update with ABI review and GPU
# validation; a version string alone does not identify locally patched code.
_SOURCES = {
    "prefill": {
        "gdn_prefill": "9a3e122b2dc24c1f834f7d9b7df6e54aa7c23b6bdb1cfdf29987684c7e6ae17d",
        "api_logging": "8bda6aba04be7eb5891a7fd07c88346d2282ecf2d2bf2f68787033f0e231b884",
        "gdn_kernels.blackwell.gdn_prefill": "3a3e4158ec04c8fc144998975f72c7a5828c0cffa8c44e911af8e69f44259266",
        "gdn_kernels.blackwell.gated_delta_net_chunked": "7970d5e755b38acd558d2041f9f1d3764f42976532d7209110bb8984fdb81e28",
        "gdn_kernels.blackwell.gated_delta_net_chunked_sm107": "bb62ffeb6d1f428ee5df03dabc0da609d74d0597e5ed214081b0e015d5955e1e",
    },
    "decode": {
        "gdn_kernels.gdn_decode_bf16_state_rubin": "e1ae8dec1cb5636b40d3c6bd075f16538a6cff816932a9f2f19c6711a0780b49",
    },
}


@functools.cache
def compatibility_error(operation: Literal["prefill", "decode"]) -> str | None:
    """Return why Rubin GDN cannot run, or None for the audited upstream ABI.

    Args:
        operation: The prefill or T=1 decode adapter being selected.

    Returns:
        A diagnostic for missing/changed upstream code. An unavailable adapter
        must not be registered, allowing automatic portable-kernel selection.
    """
    sources = _SOURCES[operation]
    prefix = f"FlashInfer Rubin GDN {operation}"
    try:
        import flashinfer

        if flashinfer.__version__ != _VERSION:
            return (
                f"{prefix} requires FlashInfer {_VERSION}; got {flashinfer.__version__}"
            )
        modules = {}
        for name, expected in sources.items():
            module = importlib.import_module(f"flashinfer.{name}")
            if (
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                != expected
            ):
                return f"{prefix}: unsupported upstream source flashinfer.{name}"
            modules[name] = module
        if operation == "decode":
            module = modules["gdn_kernels.gdn_decode_bf16_state_rubin"]
            for name in (
                "gdn_t1_rubin_kernel",
                "_run_t1_rubin",
                "_build_t1_rubin",
                "_compiled_t1_rubin",
                "_launch_t1_rubin",
                "_default_indices",
            ):
                vars(module)[name]
            entry = module.gated_delta_rule_t1_rubin
            required = {
                "q",
                "k",
                "v",
                "A_log",
                "a",
                "dt_bias",
                "b",
                "initial_state_source",
                "initial_state_indices",
                "output_state_indices",
                "scale",
                "use_qk_l2norm_in_kernel",
            }
        else:
            module = modules["gdn_kernels.blackwell.gdn_prefill"]
            module._PREFILL_KERNEL_CLASSES["sm107"]
            module._get_compiled_cache
            module.chunk_gated_delta_rule_sm100
            entry = modules["gdn_prefill"].chunk_gated_delta_rule
            required = {
                "q",
                "k",
                "v",
                "g",
                "beta",
                "initial_state",
                "output_final_state",
                "cu_seqlens",
                "scale",
                "use_qk_l2norm_in_kernel",
                "use_cp",
            }
        missing = required - inspect.signature(entry).parameters.keys()
        if missing:
            return f"{prefix}: missing API parameters {sorted(missing)}"
    except (
        ImportError,
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        return f"{prefix}: incompatible upstream API ({error})"
    return None
