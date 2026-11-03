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

"""Benchmark generators for dense grouped-query attention and its prologue."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "prepare_gqa_prologue",
    "prepare_mha_decode",
    "prepare_mha_extend",
    "prepare_mha_prefill",
]

_ROPE_STYLES = frozenset({"gptj", "neox", "none"})


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"Attention {name} must be a positive integer")
    return value


def _flag(parameters: dict[str, Any], name: str) -> bool:
    value = parameters[name]
    if not isinstance(value, bool):
        raise _invalid(f"Attention {name} must be a boolean")
    return value


def _common(request: BenchmarkRequest) -> tuple[int, int, int, int]:
    """Validate the shared request fields; returns window_left and head geometry."""
    if request.registration is not None and request.solution is not None:
        raise _invalid("registration and solution are mutually exclusive")
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid("Attention benchmark correctness validation is not implemented")
    if parameters["dtype"] != "bfloat16":
        raise _invalid("Attention dtype must be bfloat16")
    window_left = parameters["window_left"]
    if (
        not isinstance(window_left, int)
        or isinstance(window_left, bool)
        or window_left < -1
    ):
        raise _invalid("Attention window_left must be -1 or a non-negative integer")
    num_q_heads = _positive(parameters, "num_q_heads")
    num_kv_heads = _positive(parameters, "num_kv_heads")
    head_dim = _positive(parameters, "head_dim")
    if num_q_heads % num_kv_heads:
        raise _invalid("Attention num_kv_heads must divide num_q_heads")
    return window_left, num_q_heads, num_kv_heads, head_dim


def _selected_spec(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    roles: tuple[str, ...],
    traits: dict[str, object],
) -> KernelSpec:
    """Name the registration the operation's own selection resolves to."""
    load_builtin_kernels()
    try:
        selected = select_kernel(
            request.family,
            request.mode,
            format_signature(
                **{role: dense_tensor_format(torch.bfloat16) for role in roles}
            ),
            platform=platform,
            traits=traits,
            solution=request.solution,
            override=request.registration,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)


def _paged_cache(
    seq_lens: list[int],
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return K/V page pools and a page table scattering each sequence's pages.

    Pages are assigned in a seeded random order, so consecutive pages of one
    sequence are not adjacent in memory, as after a serving cache has churned.
    """
    pages_per_seq = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]
    max_pages = max(pages_per_seq)
    num_pages = sum(pages_per_seq)
    order = torch.randperm(num_pages, device="cuda", generator=generator).to(
        torch.int32
    )
    page_table = torch.zeros(
        (len(seq_lens), max_pages), dtype=torch.int32, device="cuda"
    )
    start = 0
    for row, pages in enumerate(pages_per_seq):
        page_table[row, :pages] = order[start : start + pages]
        start += pages
    k_cache, v_cache = (
        _randn((num_pages, page_size, num_kv_heads, head_dim), generator)
        for _ in range(2)
    )
    return k_cache, v_cache, page_table


def prepare_mha_decode(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one paged-KV decode attention call.

    ``batch`` requests each attend ``q_len`` new query rows to a cache of
    ``context_tokens`` tokens (including the new rows). ``max_seqlen_k`` is the
    server's context limit, which kernels may use to size their KV splits.
    """

    window_left, num_q_heads, num_kv_heads, head_dim = _common(request)
    parameters = request.parameters
    batch = _positive(parameters, "batch")
    q_len = _positive(parameters, "q_len")
    context = _positive(parameters, "context_tokens")
    page_size = _positive(parameters, "page_size")
    max_seqlen_k = _positive(parameters, "max_seqlen_k")
    if not q_len <= context <= max_seqlen_k:
        raise _invalid("Attention needs q_len <= context_tokens <= max_seqlen_k")

    spec = _selected_spec(
        request,
        platform,
        ("q", "k_cache", "v_cache"),
        {
            "q_len": q_len,
            "head_dim": head_dim,
            "page_size": page_size,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "sliding_window": window_left >= 0,
        },
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    k_cache, v_cache, page_table = _paged_cache(
        [context] * batch, page_size, num_kv_heads, head_dim, generator
    )
    q = _randn((batch * q_len, num_q_heads, head_dim), generator)
    cache_seqlens = torch.full((batch,), context, dtype=torch.int32, device="cuda")

    from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

    def invoke() -> object:
        return mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=q_len,
            window_left=window_left,
            solution=request.solution,
            override=request.registration,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "batch": batch,
            "q_len": q_len,
            "context_tokens": context,
            "max_seqlen_k": max_seqlen_k,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "window_left": window_left,
            "dtype": "bfloat16",
            "kv_dtype": "bfloat16",
        },
        validation=None,
    )


def prepare_mha_prefill(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one causal prefill over uncached K/V rows.

    ``sequences`` prompts of ``tokens_per_sequence`` rows each attend their own
    prefix, as the first chunk of a prompt does when attention reads the
    freshly projected K/V rather than the cache.
    """

    window_left, num_q_heads, num_kv_heads, head_dim = _common(request)
    parameters = request.parameters
    sequences = _positive(parameters, "sequences")
    tokens_per_sequence = _positive(parameters, "tokens_per_sequence")
    total = sequences * tokens_per_sequence

    spec = _selected_spec(
        request,
        platform,
        ("q", "k", "v"),
        {
            "head_dim": head_dim,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "skip_softmax": False,
            "sliding_window": window_left >= 0,
        },
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    q = _randn((total, num_q_heads, head_dim), generator)
    k = _randn((total, num_kv_heads, head_dim), generator)
    v = _randn((total, num_kv_heads, head_dim), generator)
    cu_seqlens_cpu = list(range(0, total + 1, tokens_per_sequence))
    cu_seqlens = torch.tensor(cu_seqlens_cpu, dtype=torch.int32, device="cuda")

    from tokenspeed_kernel.ops.attention.mha import mha_prefill

    def invoke() -> object:
        return mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=tokens_per_sequence,
            window_left=window_left,
            solution=request.solution,
            override=request.registration,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "sequences": sequences,
            "tokens_per_sequence": tokens_per_sequence,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "window_left": window_left,
            "dtype": "bfloat16",
        },
        validation=None,
    )


def prepare_mha_extend(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one causal extend over a paged KV cache.

    ``sequences`` requests each append ``extend_tokens`` query rows after
    ``prefix_tokens`` cached ones, as a later chunk of a chunked prefill does;
    the cache already holds the new rows' K/V.
    """

    window_left, num_q_heads, num_kv_heads, head_dim = _common(request)
    parameters = request.parameters
    sequences = _positive(parameters, "sequences")
    extend_tokens = _positive(parameters, "extend_tokens")
    prefix_tokens = _positive(parameters, "prefix_tokens")
    page_size = _positive(parameters, "page_size")
    max_seqlen_k = _positive(parameters, "max_seqlen_k")
    seq_len = prefix_tokens + extend_tokens
    if seq_len > max_seqlen_k:
        raise _invalid("Attention prefix and extend tokens exceed max_seqlen_k")

    spec = _selected_spec(
        request,
        platform,
        ("q", "k_cache", "v_cache"),
        {
            "head_dim": head_dim,
            "page_size": page_size,
            "is_causal": True,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "sliding_window": window_left >= 0,
        },
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    k_cache, v_cache, page_table = _paged_cache(
        [seq_len] * sequences, page_size, num_kv_heads, head_dim, generator
    )
    q = _randn((sequences * extend_tokens, num_q_heads, head_dim), generator)
    cu_seqlens_q = torch.arange(
        0,
        sequences * extend_tokens + 1,
        extend_tokens,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens_kv = torch.arange(
        0, sequences * seq_len + 1, seq_len, dtype=torch.int32, device="cuda"
    )
    cache_seqlens = torch.full((sequences,), seq_len, dtype=torch.int32, device="cuda")

    from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache

    def invoke() -> object:
        return mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=extend_tokens,
            max_seqlen_k=max_seqlen_k,
            is_causal=True,
            window_left=window_left,
            solution=request.solution,
            override=request.registration,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "sequences": sequences,
            "extend_tokens": extend_tokens,
            "prefix_tokens": prefix_tokens,
            "max_seqlen_k": max_seqlen_k,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "window_left": window_left,
            "dtype": "bfloat16",
            "kv_dtype": "bfloat16",
        },
        validation=None,
    )


def prepare_gqa_prologue(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one GQA prologue: per-head QK norm, RoPE and the KV cache write.

    Q, K and V are column views of one packed projection ``[tokens,
    qkv_row_stride]``, as split from a fused QKV projection. ``has_norm``
    applies the per-head RMSNorm and ``rope`` the rotary pairing (``"none"``
    for NoPE). Without either, the call only writes K/V rows that a producer
    already prepared. Rows go to distinct random slots of a ``cache_slots``
    pool; positions are random below ``max_position``.
    """

    if request.registration is not None and request.solution is not None:
        raise _invalid("registration and solution are mutually exclusive")
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid("Attention benchmark correctness validation is not implemented")
    if parameters["dtype"] != "bfloat16":
        raise _invalid("Attention dtype must be bfloat16")
    tokens = _positive(parameters, "tokens")
    num_q_heads = _positive(parameters, "num_q_heads")
    num_kv_heads = _positive(parameters, "num_kv_heads")
    head_dim = _positive(parameters, "head_dim")
    qkv_row_stride = _positive(parameters, "qkv_row_stride")
    cache_slots = _positive(parameters, "cache_slots")
    max_position = _positive(parameters, "max_position")
    has_norm = _flag(parameters, "has_norm")
    return_kv = _flag(parameters, "return_kv")
    rope = parameters["rope"]
    if rope not in _ROPE_STYLES:
        raise _invalid(
            f"Attention rope must be one of: {', '.join(sorted(_ROPE_STYLES))}"
        )
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    if qkv_row_stride < q_width + 2 * kv_width:
        raise _invalid("Attention qkv_row_stride is narrower than Q, K and V")
    if tokens > cache_slots:
        raise _invalid("Attention tokens exceed cache_slots")

    from tokenspeed_kernel.ops.attention.prologue import (
        HeadKVCache,
        HeadNorm,
        RopeStyle,
        Rotary,
        gqa_prologue,
    )

    spec = _selected_spec(
        request,
        platform,
        ("q",),
        {
            "head_dim": head_dim,
            "token_heads": tokens * num_q_heads,
            "has_norm": has_norm,
            "kv_format": "native",
            "kv_convert": False,
            "mrope": False,
            "partial_rotary": False,
            "return_kv": return_kv,
            "rope_style": rope,
        },
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    packed = _randn((tokens, qkv_row_stride), generator)
    q = packed[:, :q_width]
    k = packed[:, q_width : q_width + kv_width]
    v = packed[:, q_width + kv_width : q_width + 2 * kv_width]
    k_cache, v_cache = (
        torch.zeros(
            (cache_slots, num_kv_heads, head_dim), dtype=torch.bfloat16, device="cuda"
        )
        for _ in range(2)
    )
    slots = torch.randperm(cache_slots, device="cuda", generator=generator)[:tokens]
    cache = HeadKVCache(k_cache=k_cache, v_cache=v_cache, scales=None, slots=slots)
    norm = (
        HeadNorm(
            q_weight=torch.full((head_dim,), 0.125, dtype=torch.float32, device="cuda"),
            k_weight=torch.ones(head_dim, dtype=torch.float32, device="cuda"),
            weight_offset=0.0,
            eps=1e-5,
        )
        if has_norm
        else None
    )
    rotary = None
    if rope != "none":
        angles = torch.rand(
            (max_position, head_dim // 2),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ).mul_(2 * torch.pi)
        rotary = Rotary(
            cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
            positions=torch.randint(
                0, max_position, (tokens,), device="cuda", generator=generator
            ),
            style=RopeStyle(rope),
            mrope=None,
        )

    def invoke() -> object:
        return gqa_prologue(
            q,
            k,
            v,
            norm=norm,
            rotary=rotary,
            cache=cache,
            return_kv=return_kv,
            solution=request.solution,
            override=request.registration,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "tokens": tokens,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "qkv_row_stride": qkv_row_stride,
            "cache_slots": cache_slots,
            "max_position": max_position,
            "has_norm": has_norm,
            "rope": rope,
            "return_kv": return_kv,
            "dtype": "bfloat16",
            "kv_dtype": "bfloat16",
        },
        validation=None,
    )
