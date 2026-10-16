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

"""Compile GFX950 QSA scoring without requiring an AMD GPU."""

import pytest
from tokenspeed_kernel._triton import gluon, triton


@pytest.mark.parametrize("metadata_type", ["i32", "i64"])
@pytest.mark.parametrize("block_q, head_dim", [(4, 16), (4, 128), (16, 128)])
def test_gluon_qsa_scores_compile_with_integer_metadata(
    metadata_type, block_q, head_dim
):
    indexer = pytest.importorskip(
        "tokenspeed_kernel_amd.ops.gfx950.attention.qsa.indexer"
    )
    fn = indexer.gluon_qsa_block_scores_gfx950
    constants = {
        "NUM_HEADS": 4,
        "HEADS_PAD": 4,
        "HEAD_DIM": head_dim,
        "HEAD_DIM_PAD": max(32, head_dim),
        "BLOCK_Q": block_q,
        "BLOCK_N": 128,
        "num_warps": 4,
    }
    signature = {name: "i32" for name in fn.arg_names if name not in constants}
    signature.update(
        query="*bf16",
        key_cache="*bf16",
        page_table=f"*{metadata_type}",
        request_indices=f"*{metadata_type}",
        complete_blocks=f"*{metadata_type}",
        logits="*fp32",
    )
    source = gluon._runtime.GluonASTSource(fn, signature, constexprs=constants)
    compiled = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco"]
