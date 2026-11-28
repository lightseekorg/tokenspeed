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
from tokenspeed_kernel.signature import (
    FormatSignature,
    dense_tensor_format,
    format_signature,
)

__all__ = [
    "prepare_gqa_prologue",
    "prepare_mha_decode",
    "prepare_mha_extend",
    "prepare_mha_prefill",
]

_ROPE_STYLES = frozenset({"gptj", "neox", "none"})
# KV cache storage: the model dtype, or FP8 E4M3 at unit scale, as
# ``--kv-cache-dtype fp8_e4m3`` stores it for a checkpoint without KV-cache
# scales. Attention reads an FP8 cache with FP8 queries, as the serving
# backend converts them.
_KV_CACHE_DTYPES = {"bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn}
# The GQA prologue returns the query in the model dtype, or in FP8 for a
# decode that reads an FP8 cache.
_QUERY_DTYPES = _KV_CACHE_DTYPES
# The decode output gate's RMSNorm epsilon; it does not affect timing.
_OUTPUT_GATE_EPS = 1e-5


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"Attention {name} must be a positive integer")
    return value


def _non_negative(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _invalid(f"Attention {name} must be a non-negative integer")
    return value


def _dtype(
    parameters: dict[str, Any], name: str, choices: dict[str, torch.dtype]
) -> torch.dtype:
    value = parameters[name]
    if value not in choices:
        raise _invalid(f"Attention {name} must be one of: {', '.join(choices)}")
    return choices[value]


def _dense_signature(roles: tuple[str, ...], dtype: torch.dtype) -> FormatSignature:
    return format_signature(**{role: dense_tensor_format(dtype) for role in roles})


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
    signature: FormatSignature,
    traits: dict[str, object],
    feature_choices: tuple[frozenset[str] | None, ...],
) -> KernelSpec:
    """Name the registration the operation's own selection resolves to.

    ``feature_choices`` lists the feature sets the operation tries, most
    preferred first (``(None,)`` for an operation that asks for none); the
    first one a registration serves selects it.
    """
    load_builtin_kernels()
    error: NoKernelFoundError | None = None
    for features in feature_choices:
        try:
            selected = select_kernel(
                request.family,
                request.mode,
                signature,
                features=features,
                platform=platform,
                traits=traits,
                solution=request.solution,
                override=request.registration,
            )
        except NoKernelFoundError as miss:
            error = miss
            continue
        break
    else:
        assert error is not None
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
    kv_cache_dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return K/V page pools in ``kv_cache_dtype`` and a page table scattering
    each sequence's pages.

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
        _randn((num_pages, page_size, num_kv_heads, head_dim), generator).to(
            kv_cache_dtype
        )
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
    ``kv_cache_dtype`` stores the cache (an FP8 cache is read with FP8
    queries). ``output_gate`` passes a gate for the RMSNorm-sigmoid output
    gate, which a kernel with the ``rmsnorm_output_gate`` feature applies
    where it stores the output; with any other kernel the timed call also runs
    the separate gate op. ``split_schedule`` passes the split schedule that
    serving computes once a step, outside this call, for a target's
    full-attention verify decode; it is computed before timing.
    """

    window_left, num_q_heads, num_kv_heads, head_dim = _common(request)
    parameters = request.parameters
    batch = _positive(parameters, "batch")
    q_len = _positive(parameters, "q_len")
    context = _positive(parameters, "context_tokens")
    page_size = _positive(parameters, "page_size")
    max_seqlen_k = _positive(parameters, "max_seqlen_k")
    kv_cache_dtype = _dtype(parameters, "kv_cache_dtype", _KV_CACHE_DTYPES)
    output_gate = _flag(parameters, "output_gate")
    split_schedule = _flag(parameters, "split_schedule")
    if not q_len <= context <= max_seqlen_k:
        raise _invalid("Attention needs q_len <= context_tokens <= max_seqlen_k")
    if split_schedule and window_left >= 0:
        raise _invalid("Attention split_schedule serves full attention only")

    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_schedule,
        mha_decode_takes_schedule,
        mha_decode_with_kvcache,
    )

    rows = batch * q_len
    if split_schedule and not mha_decode_takes_schedule(
        # Only the tensors' dtypes and shapes are read.
        q=torch.empty(
            (rows, num_q_heads, head_dim), dtype=kv_cache_dtype, device="meta"
        ),
        k_cache=torch.empty(
            (1, page_size, num_kv_heads, head_dim), dtype=kv_cache_dtype, device="meta"
        ),
        v_cache=torch.empty(
            (1, page_size, num_kv_heads, head_dim), dtype=kv_cache_dtype, device="meta"
        ),
        max_seqlen_q=q_len,
        block_draft=False,
        solution=request.solution,
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.NOT_APPLICABLE,
            "No schedule-reading decode kernel serves these calls",
        )
    features = frozenset({"split_schedule"}) if split_schedule else frozenset()
    # mha_decode_with_kvcache prefers a kernel that gates where it stores the
    # output, for 64-wide heads; otherwise it runs the separate gate op after.
    feature_choices = (
        (features | {"rmsnorm_output_gate"}, features or None)
        if output_gate and head_dim == 64
        else (features or None,)
    )
    spec = _selected_spec(
        request,
        platform,
        _dense_signature(("q", "k_cache", "v_cache"), kv_cache_dtype),
        {
            "block_draft": False,
            "q_len": q_len,
            "head_dim": head_dim,
            "page_size": page_size,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "sliding_window": window_left >= 0,
            "window_sink": False,
        },
        feature_choices,
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    k_cache, v_cache, page_table = _paged_cache(
        [context] * batch, page_size, num_kv_heads, head_dim, kv_cache_dtype, generator
    )
    q = _randn((rows, num_q_heads, head_dim), generator).to(kv_cache_dtype)
    gate = _randn((rows, num_q_heads * head_dim), generator) if output_gate else None
    cache_seqlens = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    schedule = (
        mha_decode_schedule(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=q_len,
            block_draft=False,
            solution=request.solution,
        )
        if split_schedule
        else None
    )

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
            schedule=schedule,
            solution=request.solution,
            override=request.registration,
            block_draft=False,
            output_gate=gate,
            output_gate_eps=_OUTPUT_GATE_EPS if output_gate else None,
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
            "kv_cache_dtype": parameters["kv_cache_dtype"],
            "output_gate": output_gate,
            "split_schedule": split_schedule,
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
        _dense_signature(("q", "k", "v"), torch.bfloat16),
        {
            "head_dim": head_dim,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "skip_softmax": False,
            "sliding_window": window_left >= 0,
        },
        (None,),
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
    ``prefix_tokens`` cached ones (none for a prompt's first chunk), as a
    chunk of a chunked prefill does; the cache already holds the new rows'
    K/V. ``kv_cache_dtype`` stores the cache; an FP8 cache is read with FP8
    queries. ``max_seqlen_k`` is the requests' length, as serving passes the
    longest request's length: kernels read the cached prefix as
    ``max_seqlen_k - max_seqlen_q``.
    """

    window_left, num_q_heads, num_kv_heads, head_dim = _common(request)
    parameters = request.parameters
    sequences = _positive(parameters, "sequences")
    extend_tokens = _positive(parameters, "extend_tokens")
    prefix_tokens = _non_negative(parameters, "prefix_tokens")
    page_size = _positive(parameters, "page_size")
    kv_cache_dtype = _dtype(parameters, "kv_cache_dtype", _KV_CACHE_DTYPES)
    if "max_seqlen_k" in parameters:
        raise _invalid(
            "Attention extend takes max_seqlen_k from prefix_tokens and "
            "extend_tokens, as serving passes the longest request's length"
        )
    seq_len = prefix_tokens + extend_tokens

    spec = _selected_spec(
        request,
        platform,
        _dense_signature(("q", "k_cache", "v_cache"), kv_cache_dtype),
        {
            "head_dim": head_dim,
            "page_size": page_size,
            "is_causal": True,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "sliding_window": window_left >= 0,
        },
        (None,),
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    k_cache, v_cache, page_table = _paged_cache(
        [seq_len] * sequences,
        page_size,
        num_kv_heads,
        head_dim,
        kv_cache_dtype,
        generator,
    )
    q = _randn((sequences * extend_tokens, num_q_heads, head_dim), generator).to(
        kv_cache_dtype
    )
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
            max_seqlen_k=seq_len,
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
            "max_seqlen_k": seq_len,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "window_left": window_left,
            "dtype": "bfloat16",
            "kv_cache_dtype": parameters["kv_cache_dtype"],
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
    pool in ``kv_cache_dtype`` (the write converts BF16 rows to an FP8
    cache); positions are random below ``max_position``. ``query_dtype`` is
    the returned query's dtype: serving asks for FP8 in a decode that reads
    an FP8 cache, and the model dtype otherwise.
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
    kv_cache_dtype = _dtype(parameters, "kv_cache_dtype", _KV_CACHE_DTYPES)
    query_dtype = _dtype(parameters, "query_dtype", _QUERY_DTYPES)
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
        _dense_signature(("q",), torch.bfloat16),
        {
            "head_dim": head_dim,
            "token_heads": tokens * num_q_heads,
            "has_norm": has_norm,
            "relative_norm": False,
            "kv_conv": "none",
            # An unscaled E4M3 cache is the FP8 format (HeadKVCache.format).
            "kv_format": "fp8" if kv_cache_dtype is torch.float8_e4m3fn else "native",
            "kv_convert": False,
            "mrope": False,
            "partial_rotary": False,
            "return_kv": return_kv,
            "rope_style": rope,
        },
        (None,),
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    packed = _randn((tokens, qkv_row_stride), generator)
    q = packed[:, :q_width]
    k = packed[:, q_width : q_width + kv_width]
    v = packed[:, q_width + kv_width : q_width + 2 * kv_width]
    k_cache, v_cache = (
        torch.zeros(
            (cache_slots, num_kv_heads, head_dim), dtype=kv_cache_dtype, device="cuda"
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
            round_before_rope=False,
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
            query_dtype=query_dtype,
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
            "kv_cache_dtype": parameters["kv_cache_dtype"],
            "query_dtype": parameters["query_dtype"],
        },
        validation=None,
    )
