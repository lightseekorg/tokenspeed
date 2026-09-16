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

"""Static AMD kernel checks that skip when optional packages are unavailable."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenspeed_triton")
pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

from tokenspeed_kernel_amd.ops.gfx950.attention import dsv4 as dsv4_pkg  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (  # noqa: E402
    prefill,
    sparse_prefill,
)


def _shape(*dims: int) -> SimpleNamespace:
    return SimpleNamespace(shape=dims)


def test_sparse_prefill_is_exported_from_amd_dsv4_package():
    assert (
        dsv4_pkg.gluon_dsv4_sparse_prefill_gfx950
        is sparse_prefill.gluon_dsv4_sparse_prefill_gfx950
    )


def test_registered_prefill_wrapper_routes_target_shapes_to_sparse_helper():
    source = inspect.getsource(prefill.gluon_dsv4_prefill_gfx950)

    assert "gluon_dsv4_sparse_prefill_gfx950" in source
    assert "_use_sparse_prefill(q, indices)" in source
    assert prefill._use_sparse_prefill(_shape(1, 64, 512), _shape(1, 128))
    assert prefill._use_sparse_prefill(_shape(1, 128, 512), _shape(1, 640))
    assert not prefill._use_sparse_prefill(_shape(1, 16, 512), _shape(1, 640))
    assert not prefill._use_sparse_prefill(_shape(1, 64, 512), _shape(1, 64))


def test_sparse_prefill_masks_invalid_selected_rows_before_kv_loads():
    source = " ".join(Path(sparse_prefill.__file__).read_text().split())

    assert "num_queries, num_kv_rows, num_iters" in source
    assert "assert topk3.size(2) % block_k == 0" in source
    assert "s, kv3.shape[1], topk3.size(2) // block_k" in source
    assert "ASSUME_COMPACT_INDICES" in source
    assert "HAS_LENS" not in source
    assert "< num_heads" not in source
    assert "mask=valid0[None, :]" in source
    assert "mask=next_valid[None, :]" in source
    assert "mask=final_load_valid[None, :]" in source
    assert "index0.to(tl.int64) < num_kv_rows" in source
    assert "index0_mfma.to(tl.int64) < num_kv_rows" in source
    assert "next_index.to(tl.int64) < num_kv_rows" in source
    assert "next_index_mfma.to(tl.int64) < num_kv_rows" in source
    assert "final_index.to(tl.int64) < num_kv_rows" in source
    assert "penultimate_index.to(tl.int64) < num_kv_rows" in source
    assert "final_mfma_index.to(tl.int64) < num_kv_rows" in source
