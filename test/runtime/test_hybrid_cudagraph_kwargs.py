"""Hybrid cuda-graph kwargs filtering for user-selectable sub-backends.

The hybrid backend's full-attention sub-backend is user-selectable and may
have a narrow ``init_cuda_graph_state`` signature (e.g. TRTLLM MHA takes only
``(max_bs, seq_lens_buf)``); the shared ``init_backend_cuda_graph_state``
helper must drop unaccepted extras before forwarding, hand the full extras to
``**kwargs`` backends, and never swallow TypeErrors raised inside a backend
body.
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class _NarrowBackend:
    """Positional-only signature, like TRTLLMMHAAttnBackend."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def init_cuda_graph_state(self, max_bs, seq_lens_buf):
        self.calls.append((max_bs, seq_lens_buf))


class _VarKwBackend:
    """Extras-consuming signature, like MHAAttnBackend."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def init_cuda_graph_state(self, max_bs, seq_lens_buf, **kwargs):
        self.calls.append((max_bs, seq_lens_buf, kwargs))


class _NamedExtraBackend:
    """Accepts one named extra only."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def init_cuda_graph_state(self, max_bs, seq_lens_buf, paged_cache_group_specs=None):
        self.calls.append((max_bs, seq_lens_buf, paged_cache_group_specs))


class _RaisingBackend:
    """VAR_KEYWORD signature whose body raises TypeError: the probe-based
    helper must propagate it (a retry-based one would silently drop specs)."""

    device = "cpu"

    def init_cuda_graph_state(self, max_bs, seq_lens_buf, **kwargs):
        raise TypeError("from inside the backend body")


_EXTRAS = {
    "paged_cache_group_specs": ("full_attention", "linear_attention"),
    "max_tokens_per_req": 2,
    "overlap_schedule_depth": 1,
}


class InitBackendCudaGraphStateHelperTest(unittest.TestCase):
    """Direct contract tests for the shared signature-filter helper."""

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.backends.base import (
                init_backend_cuda_graph_state,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.helper = init_backend_cuda_graph_state

    def test_narrow_backend_receives_positional_only(self):
        backend = _NarrowBackend()
        buf = object()
        self.helper(backend, 4, buf, **_EXTRAS)
        self.assertEqual(backend.calls, [(4, buf)])

    def test_var_kw_backend_receives_all_extras(self):
        backend = _VarKwBackend()
        buf = object()
        self.helper(backend, 4, buf, **_EXTRAS)
        self.assertEqual(backend.calls, [(4, buf, _EXTRAS)])

    def test_named_extra_backend_receives_only_matching_kwarg(self):
        backend = _NamedExtraBackend()
        buf = object()
        self.helper(backend, 4, buf, **_EXTRAS)
        self.assertEqual(
            backend.calls,
            [(4, buf, _EXTRAS["paged_cache_group_specs"])],
        )

    def test_type_error_from_backend_body_propagates(self):
        with self.assertRaisesRegex(TypeError, "from inside the backend body"):
            self.helper(_RaisingBackend(), 4, object(), **_EXTRAS)


class HybridInitCudaGraphStateForwardingTest(unittest.TestCase):
    """The hybrid wrapper must apply the same filtering when forwarding to
    both its user-selectable full backend and its mamba backend."""

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                HybridLinearAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.hybrid_cls = HybridLinearAttnBackend

    def _hybrid(self, full_backend, linear_backend):
        return self.hybrid_cls(full_backend, linear_backend, full_attn_layers=[0])

    def test_narrow_full_backend_with_extras_succeeds(self):
        full = _NarrowBackend()
        mamba = _NarrowBackend()
        buf = object()
        self._hybrid(full, mamba).init_cuda_graph_state(4, buf, **_EXTRAS)
        self.assertEqual(full.calls, [(4, buf)])
        self.assertEqual(mamba.calls, [(4, buf)])

    def test_var_kw_full_backend_receives_extras(self):
        full = _VarKwBackend()
        mamba = _NarrowBackend()
        buf = object()
        self._hybrid(full, mamba).init_cuda_graph_state(4, buf, **_EXTRAS)
        self.assertEqual(full.calls, [(4, buf, _EXTRAS)])
        # The narrow mamba sub-backend is covered by the same filter.
        self.assertEqual(mamba.calls, [(4, buf)])

    def test_var_kw_mamba_backend_receives_extras(self):
        full = _NarrowBackend()
        mamba = _VarKwBackend()
        buf = object()
        self._hybrid(full, mamba).init_cuda_graph_state(4, buf, **_EXTRAS)
        self.assertEqual(full.calls, [(4, buf)])
        self.assertEqual(mamba.calls, [(4, buf, _EXTRAS)])


if __name__ == "__main__":
    unittest.main()
