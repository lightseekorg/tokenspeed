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

"""A layer states its prologue steps once; PagedAttention hands them to the
kernel entry."""

import ast
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attention.prologue import (  # noqa: E402
    HeadKVCache,
    MRope,
    RopeStyle,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers import paged_attention  # noqa: E402
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm, RMSNorm  # noqa: E402
from tokenspeed.runtime.layers.rotary_embedding import (  # noqa: E402
    MRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
)

# Draft models inject target-context rows into their own cache outside any attention forward.
_CONTEXT_KV_WRITERS = {
    "models/dflash.py:DFlashAttention.apply_k_norm",
    "models/dflash.py:DFlashDraftModel.write_context_kv",
    "models/dflash2.py:DFlash2DraftModel.write_context_kv",
    "models/kimi_k3_dspark.py:K3DSparkAttention.apply_latent_rope",
    "models/kimi_k3_dspark.py:K3DSparkModel.write_context_kv",
    "execution/drafter/dflash.py:DFlash._write_native_cache_fused",
    "execution/drafter/dflash.py:DFlash._write_native_cache_fused_mla",
    "models/deepseek_v41_dspark.py:DeepseekV41DSparkModel._main_kv",
}
# Keys the prologue does not own: sparse indexers' and DeepSeek-V4's own attention.
_OTHER_KEY_OWNERS = {
    "models/deepseek_v4.py:DeepseekV4Attention._project_q_kv",
    "models/glm5.py:GlmDsaIndexer.forward",
    "models/glm53_flash.py:Glm53FlashIndexer.forward",
}
# The norm, RoPE, quantize and KV-write steps the prologue owns: modules and kernel entries.
_PROLOGUE_STEPS = {
    "apply_k_rope",
    "apply_rope",
    "apply_rope_mla",
    "fp8_quantize",
    "fused_fp8_set_kv_buffer",
    "k_norm",
    "mla_latent_norm_rope_scatter",
    "q_norm",
    "qk_rmsnorm",
    "quantize_store_kv_mxfp8",
    "rotary_emb",
    "set_kv_buffer",
    "set_mla_kv_buffer",
    "set_mla_kv_buffer_triton",
    "store_kv_cache",
    "store_latent_per_token_head",
    "write_kv",
}


def _prologue(monkeypatch, *, qk_norm, mode, rows, slots):
    """Run ``PagedAttention.prologue`` and return what it handed the kernel entry."""
    handed = {}
    monkeypatch.setattr(
        paged_attention,
        "gqa_prologue",
        lambda q, k, v, **kw: handed.update(kw, q=q, k=k, v=v),
    )
    layer = paged_attention.PagedAttention(
        4, 64, 1.0, num_kv_heads=2, layer_id=0, rotary_emb=None, qk_norm=qk_norm
    )
    cache = torch.zeros(8, 2, 64, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        forward_mode=mode,
        attn_backend=SimpleNamespace(forward_write_locations=lambda layer, m: slots),
        token_to_kv_pool=SimpleNamespace(
            kv_write_target=lambda layer_id, s: HeadKVCache(cache, cache, None, s)
        ),
    )
    kv = torch.zeros(rows, 2 * 64, dtype=torch.bfloat16)
    layer.prologue(
        torch.zeros(rows, 4 * 64, dtype=torch.bfloat16), kv, kv, torch.arange(rows), ctx
    )
    return handed


def _forward(monkeypatch, *, capturing: bool) -> tuple[dict, list]:
    """Run ``PagedAttention.forward`` for an extend; return what the kernel entry
    got and the norm-and-RoPE calls made before the break."""
    handed, before_break = {}, []
    monkeypatch.setattr(
        paged_attention, "is_breakable_capture_active", lambda: capturing
    )
    monkeypatch.setattr(
        paged_attention,
        "qk_norm_rope",
        lambda q, k, **kw: before_break.append(kw) or (q, k),
    )
    monkeypatch.setattr(
        paged_attention,
        "gqa_prologue",
        lambda q, k, v, **kw: handed.update(kw, q=q, k=k, v=v)
        or SimpleNamespace(q=q, k=k, v=v),
    )
    monkeypatch.setattr(
        paged_attention, "write_kv", lambda k, v, *, cache: handed.update(written=cache)
    )
    q_norm, k_norm = RMSNorm(64, eps=1e-5), RMSNorm(64, eps=1e-5)
    layer = paged_attention.PagedAttention(
        4,
        64,
        1.0,
        num_kv_heads=2,
        layer_id=0,
        rotary_emb=None,
        qk_norm=(q_norm, k_norm),
    )
    cache = torch.zeros(8, 2, 64, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        bs=1,
        attn_backend=SimpleNamespace(
            forward_write_locations=lambda layer, m: torch.arange(3),
            forward=lambda q, *a, **kw: q,
        ),
        token_to_kv_pool=SimpleNamespace(
            kv_write_target=lambda layer_id, s: HeadKVCache(cache, cache, None, s)
        ),
    )
    kv = torch.zeros(3, 2 * 64, dtype=torch.bfloat16)
    layer.forward(
        torch.zeros(3, 4 * 64, dtype=torch.bfloat16), kv, kv, torch.arange(3), ctx
    )
    return handed, before_break


def test_a_breakable_capture_norms_and_rotates_before_the_break(monkeypatch):
    """The captured segment prepares q and k; the break only stores."""
    handed, before_break = _forward(monkeypatch, capturing=True)
    assert len(before_break) == 1 and before_break[0]["norm"] is not None
    assert "norm" not in handed and handed["written"].slots.numel() == 3


def test_outside_a_capture_the_prologue_is_one_call(monkeypatch):
    handed, before_break = _forward(monkeypatch, capturing=False)
    assert before_break == [] and "written" not in handed
    assert handed["norm"] is not None and handed["cache"].slots.numel() == 3


@pytest.mark.parametrize("norm_cls", [RMSNorm, GemmaRMSNorm])
def test_the_prologue_gets_the_stored_norm_weight(monkeypatch, norm_cls):
    """Gemma's ``1 + w`` is formed in fp32 by the kernel, never in the weight dtype."""
    q_norm, k_norm = norm_cls(64, eps=1e-5), norm_cls(64, eps=1e-5)
    handed = _prologue(
        monkeypatch,
        qk_norm=(q_norm, k_norm),
        mode=ForwardMode.EXTEND,
        rows=3,
        slots=torch.arange(3),
    )
    norm = handed["norm"]
    assert norm.q_weight is q_norm.weight and norm.k_weight is k_norm.weight
    assert norm.weight_offset == norm_cls.weight_offset and norm.eps == 1e-5
    assert handed["return_kv"]


def test_extend_hands_the_prologue_the_backend_slots(monkeypatch):
    """Padded extend rows are prepared; only the backend's slots are written."""
    handed = _prologue(
        monkeypatch,
        qk_norm=None,
        mode=ForwardMode.EXTEND,
        rows=4,
        slots=torch.arange(3),
    )
    assert handed["cache"].slots.numel() == 3


def test_decode_rows_must_each_have_a_slot(monkeypatch):
    with pytest.raises(ValueError, match="2 decode write slots for 3 rows"):
        _prologue(
            monkeypatch,
            qk_norm=None,
            mode=ForwardMode.DECODE,
            rows=3,
            slots=torch.arange(2),
        )


@pytest.mark.parametrize("positions", [[0, 5, 4095], [0, 5, 4097]])
def test_longrope_moves_every_token_to_the_long_rows_past_the_original_context(
    positions,
):
    short, long = [1.0] * 32, [4.0] * 32
    rope = Phi3LongRoPEScaledRotaryEmbedding(
        64, 64, 8192, 4096, 10000, True, short, long
    )
    positions = torch.tensor(positions)
    rotary = rope.as_rotary(positions)
    if (positions > 4096).any():
        table = rope._compute_cos_sin_cache(8192, long, rope.long_mscale)
    else:
        table = rope._compute_cos_sin_cache(4096, short, rope.short_mscale)
    assert torch.equal(rotary.cos_sin_cache[rotary.positions], table[positions])


def test_mrope_hands_the_prologue_its_sections():
    rope = MRotaryEmbedding(
        128, 128, 4096, 10000, True, torch.bfloat16, [24, 20, 20], True
    )
    rotary = rope.as_rotary(torch.zeros(3, 5, dtype=torch.int64))
    assert rotary.mrope == MRope((24, 20, 20), True)
    assert rotary.style is RopeStyle.NEOX
    assert rotary.cos_sin_cache.dtype == torch.float32


@pytest.mark.parametrize("sparse", [True, False])
def test_msa_publishes_a_layer_after_every_field_it_writes(sparse):
    """The prologue wrote K/V; a sparse layer's kernel still writes its index keys."""
    from tokenspeed.runtime.layers.attention.backends.paged.msa import (
        MSAHybridAttnBackend,
    )

    order = []

    class _Leaf:
        def forward_extend(self, q, *args, **kwargs):
            order.append("attention")
            return q

    router = SimpleNamespace(
        write_locations=lambda layer, mode: torch.arange(4),
        _leaf_for=lambda layer: _Leaf(),
    )
    backend = object.__new__(MSAHybridAttnBackend)
    backend.step_counter = SimpleNamespace(record_cache=lambda: order.append("record"))
    backend.sparse_layer_ids = {0} if sparse else set()
    backend.sparse_router = backend.full_router = router
    layer = paged_attention.PagedAttention(
        4, 64, 1.0, num_kv_heads=2, layer_id=0, rotary_emb=None, qk_norm=None
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=1,
    )
    layer.forward(torch.zeros(4, 4 * 64), None, None, None, ctx)
    assert order == (["attention", "record"] if sparse else ["record", "attention"])


def _functions(tree: ast.Module):
    """Module functions and class methods, by qualified name."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{item.name}", item
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def test_models_run_the_prologue_steps_only_through_the_prologue():
    """Model and drafter code normalizes, rotates, quantizes and writes
    attention K/V only through the attention prologue, apart from draft
    context injection and keys the prologue does not own."""
    runtime = pathlib.Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime"
    callers = set()
    for path in [
        *runtime.glob("models/**/*.py"),
        *runtime.glob("execution/drafter/*.py"),
    ]:
        for name, fn in _functions(ast.parse(path.read_text())):
            if any(
                isinstance(node, ast.Call) and _called_name(node) in _PROLOGUE_STEPS
                for node in ast.walk(fn)
            ):
                callers.add(f"{path.relative_to(runtime)}:{name}")
    assert callers == _CONTEXT_KV_WRITERS | _OTHER_KEY_OWNERS
