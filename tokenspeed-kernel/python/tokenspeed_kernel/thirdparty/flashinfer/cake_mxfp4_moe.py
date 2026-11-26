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

"""FlashInfer's Cake MXFP4 x MXFP8 SiTU routed MoE, dense grouped-GEMM chain.

``flashinfer.fused_moe.cake`` ships the kernels and a host plan
(``CakeMxfp4MoEWrapper.plan``) that binds one token count, copies the weights
into the dense layout and runs a warm-up. Serving needs neither: the token
count changes every forward and the weights are large. This module stores the
weights once in the dense layout and enqueues the same launches per call, for
any token count.

The launch sequence and its arguments follow FlashInfer's ``CakeDensePlan``
(Apache-2.0); the kernels, their selection and their workspace layout are
FlashInfer's, read through its plan module.

The dense chain serves 1025..7168 tokens and 8192 upward on the K3 routed
geometry; ``dense_tokens`` gives the count a caller pads to.
"""

from __future__ import annotations

import functools

import torch

_STREAMED = ("w1", "w1_sf", "w2", "w2_sf")
# Token counts tried, in order, when the dense chain does not serve one.
_PAD_MULTIPLES = (1, 1024, 8192)
_DENSE_MIN_TOKENS = 1025


def _plan():
    from flashinfer.fused_moe.cake import cake_mxfp4_situ_moe_plan

    return cake_mxfp4_situ_moe_plan


def available() -> bool:
    """True when FlashInfer ships the Cake MXFP4 kernels and they can load."""
    try:
        return bool(_plan().is_available())
    except ImportError:
        return False


def dense_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    experts_per_step: int = 64,
) -> dict[str, torch.Tensor]:
    """Checkpoint MXFP4 experts in the dense chain's expert-major layout.

    Args:
        w13: ``[experts, 2 * intermediate, hidden / 2]`` packed codes, rows
            ``[gate | up]``.
        w13_scale: ``[experts, 2 * intermediate, hidden / 32]`` UE8M0 scales.
        w2: ``[experts, hidden, intermediate / 2]`` packed codes.
        w2_scale: ``[experts, hidden, intermediate / 32]`` UE8M0 scales.
        experts_per_step: Experts converted at a time, bounding the temporary
            memory to a few copies of that many experts.

    Returns:
        ``w1`` ``[experts, 2 * intermediate, hidden / 2]`` (up and gate rows
        interleaved in 64-row groups), ``w1_sf`` ``[experts, 2 * intermediate
        / 128, hidden / 128, 512]``, ``w2`` ``[experts, hidden, intermediate /
        2]`` and ``w2_sf`` ``[experts, hidden / 128, intermediate / 128, 4,
        128]``, all uint8.
    """
    plan = _plan()
    experts, w13_rows, half_hidden = w13.shape
    hidden, shard = 2 * half_hidden, w13_rows // 2

    def up_gate(t: torch.Tensor) -> torch.Tensor:
        t = t.view(torch.uint8).reshape(t.shape[0], w13_rows, -1)
        return torch.cat([t[:, shard:], t[:, :shard]], dim=1).contiguous()

    def rows(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.uint8).reshape(t.shape[0], hidden, -1).contiguous()

    parts = []
    for start in range(0, experts, experts_per_step):
        step = slice(start, start + experts_per_step)
        prepared = plan.prepare_cake_mxfp4_weights(
            up_gate(w13[step]),
            up_gate(w13_scale[step]),
            rows(w2[step]),
            rows(w2_scale[step]),
        )
        parts.append(
            plan.dense_weight_operands(
                prepared,
                local_experts=prepared["num_local_experts"],
                hidden=hidden,
                shard=shard,
            )
        )
        del prepared
    return {
        name: torch.cat([part[name] for part in parts]).contiguous()
        for name in _STREAMED
    }


@functools.cache
def _wrapper(
    *,
    device_index: int,
    num_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    num_local_experts: int,
    local_expert_offset: int,
):
    from flashinfer.fused_moe.cake import CakeMxfp4MoEWrapper

    # The wrapper is host metadata; it reads the SM count of the current device.
    with torch.cuda.device(device_index):
        return CakeMxfp4MoEWrapper(
            num_experts,
            top_k,
            hidden,
            intermediate,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
        )


@functools.lru_cache(maxsize=4096)
def _dense_decision(wrapper, tokens: int):
    """(decision, workspace fields, workspace bytes) when the dense chain serves ``tokens``."""
    decision = wrapper.decide(tokens)
    if decision.path != "dense" or not decision.supported:
        return None
    fields, size = wrapper.policy.workspace_fields(tokens)
    return decision, tuple(fields), size


def dense_tokens(
    tokens: int,
    *,
    device_index: int,
    num_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    num_local_experts: int,
    local_expert_offset: int,
) -> int:
    """The smallest token count at or above ``tokens`` the dense chain serves.

    Callers pad their rows to it; the padded rows are ordinary rows whose
    output is discarded.
    """
    wrapper = _wrapper(
        device_index=device_index,
        num_experts=num_experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
    )
    floor = max(tokens, _DENSE_MIN_TOKENS)
    for multiple in _PAD_MULTIPLES:
        candidate = -(-floor // multiple) * multiple
        if _dense_decision(wrapper, candidate) is not None:
            return candidate
    raise NotImplementedError(
        f"FlashInfer's Cake dense MoE chain serves no token count near {tokens}"
    )


def launch_forms(tokens: int, **geometry) -> tuple[str, ...] | None:
    """Names of the kernels the dense chain launches for ``tokens``, or None
    when it does not serve that count. ``geometry`` as in ``dense_tokens``."""
    planned = _dense_decision(_wrapper(**geometry), tokens)
    if planned is None:
        return None
    return tuple(launch.form for launch in planned[0].launch_plan)


@functools.cache
def _warmup_token_counts(wrapper, last: int) -> tuple[int, ...]:
    counts: dict[tuple[str, ...], int] = {}
    for tokens in range(_DENSE_MIN_TOKENS, last + 1):
        decision = wrapper.decide(tokens)
        if decision.path == "dense" and decision.supported:
            forms = tuple(launch.form for launch in decision.launch_plan)
            counts.setdefault(forms, tokens)
    return tuple(sorted(counts.values()))


def warmup_token_counts(max_tokens: int, **geometry) -> tuple[int, ...]:
    """One served token count per distinct set of kernels up to ``max_tokens``.

    FlashInfer picks the kernels by token count; running the chain once at each
    returned count loads every kernel a batch of at most ``max_tokens`` needs.
    The scan is a host-only table lookup per count (under a second for 24,576)
    and runs once per process. ``geometry`` as in ``dense_tokens``.
    """
    return _warmup_token_counts(
        _wrapper(**geometry), dense_tokens(max_tokens, **geometry)
    )


@functools.cache
def _module(device_index: int, kind: str, args: tuple, kwargs: tuple):
    """One launch handle per kernel and device; FlashInfer binds a handle to
    the device of its first launch."""
    plan = _plan()
    routing, sort = plan._routing_module(), plan._moe_sort_module()
    if kind == "preprocess":
        return routing.build_preprocess_module(*args)
    if kind == "sort_init":
        return sort.build_init_module(*args)
    if kind == "sort_coop":
        return sort.build_coop_module(*args)
    if kind == "gemm1":
        return plan._gemm1_dense_module().build_module(*args, **dict(kwargs))
    return plan._gemm2_dense_module().build_module(*args, **dict(kwargs))


def situ_routed_moe(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    beta: torch.Tensor,
    linear_beta: torch.Tensor,
    num_experts: int,
    local_expert_offset: int,
    output: torch.Tensor,
) -> None:
    """Routed MXFP4 SiTU experts over precomputed top-k, finalized into ``output``.

    Args:
        x: ``[tokens, hidden]`` float8_e4m3fn activations, ``tokens`` a count
            ``dense_tokens`` returns.
        x_sf: ``[tokens, hidden / 32]`` uint8 UE8M0 scales of ``x``.
        topk_ids: ``[tokens, top_k]`` int32 global expert ids.
        topk_weights: ``[tokens, top_k]`` FP32 or BF16 route weights.
        weights: The ``dense_weights`` tensors of this rank's experts.
        beta: ``[local experts]`` FP32 SiTU beta.
        linear_beta: ``[local experts]`` FP32 SiTU linear beta.
        num_experts: Experts of the model.
        local_expert_offset: Global id of this rank's first expert.
        output: ``[tokens, hidden]`` BF16 destination, overwritten with the
            weighted sum over this rank's experts.
    """
    plan = _plan()
    device = x.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    local_experts, w13_rows, _ = weights["w1"].shape
    shard = w13_rows // 2
    wrapper = _wrapper(
        device_index=index,
        num_experts=num_experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=shard,
        num_local_experts=local_experts,
        local_expert_offset=local_expert_offset,
    )
    planned = _dense_decision(wrapper, tokens)
    if planned is None:
        raise ValueError(
            f"FlashInfer's Cake dense MoE chain does not serve {tokens} tokens; "
            "pad to dense_tokens()"
        )
    decision, fields, workspace_bytes = planned
    policy = wrapper.policy
    backend = wrapper.backend
    sms = plan._runner_sm_count(wrapper, device)

    # Scratch of one call: FlashInfer's kernels need it 256-byte aligned, which
    # the CUDA caching allocator's 512-byte blocks satisfy.
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    b = {
        f.name: workspace.narrow(0, f.offset, f.nbytes)
        .view(getattr(torch, plan._TORCH_DTYPE[f.dtype]))
        .view(f.shape)
        for f in fields
    }
    b["w1_alpha"].fill_(1.0)
    b["w2_alpha"].fill_(1.0)
    cfg = plan.dense_launch_config(policy, decision)
    mode, weights_src, w_strides, route_weights, route_ids = plan._route_operands(
        b, topk_ids, topk_weights
    )
    routing, sort = plan._routing_module(), plan._moe_sort_module()
    gemm1_dense = plan._gemm1_dense_module()

    pre_cfg = routing.PreprocessConfig(
        mode=mode, threads=plan.ROUTE_PREPROCESS_THREADS, clear=cfg["clear"]
    )
    launches = [
        (
            _module(index, "preprocess", (pre_cfg, backend), ()),
            (routing.preprocess_grid(pre_cfg, tokens, top_k, hidden), 1, 1),
            plan.preprocess_launch_bindings(
                b,
                topk_ids=topk_ids,
                weights_src=weights_src,
                w_strides=w_strides,
                output=output,
                num_tokens=tokens,
                top_k=top_k,
                hidden=hidden,
            ),
        )
    ]
    sort_cfg = cfg["sort"]
    unused = plan.unused_launch_operands(device, group_capacity=decision.tiles)
    alt_tile = cfg["alt_tile"]
    dual = alt_tile is not None
    sort_init_args, sort_coop_args = plan.moe_sort_launch_bindings(
        b,
        topk_ids=route_ids,
        num_tokens=tokens,
        top_k=top_k,
        num_experts=num_experts,
        local_experts=local_experts,
        local_offset=local_expert_offset,
        tile=decision.dense.tile,
        alt_tile=alt_tile,
        permille=policy.dual_tile_threshold_permille(),
        mixed=False,
        narrow_tile=0,
        unused=unused,
    )
    launches.append(
        (
            _module(index, "sort_init", (sort_cfg,), ()),
            (sort.init_grid(sort_cfg, num_experts), 1, 1),
            sort_init_args,
        )
    )
    launches.append(
        (
            _module(index, "sort_coop", (sort_cfg,), ()),
            (sort.coop_grid(sms), 1, 1),
            sort_coop_args,
        )
    )

    dbg = torch.zeros(64 * sms, dtype=torch.int32, device=device)
    forms = [launch.form for launch in decision.launch_plan]
    g1_weights = {"w1": weights["w1"], "w1_sf": weights["w1_sf"]}

    def gemm1(config, form: str, *, alternate: bool):
        tile_m, n_tile, zero_fill, secondary, row_group, early, l2_hint, split = config
        module = _module(
            index,
            "gemm1",
            (backend, tile_m, n_tile, zero_fill, secondary, row_group),
            (
                ("use_pdl", cfg["pdl_alt"] if alternate else cfg["pdl"]),
                ("pdl_trigger_early", early),
                ("weight_l2_hint", l2_hint),
                *((("cluster_split", split),) if not alternate else ()),
                *sorted(plan.dense_gemm1_raster_kwargs(form).items()),
            ),
        )
        if alternate:
            items = (decision.rows // alt_tile) * (2 * shard // n_tile)
            grid = (2 * min(items, module.max_active_clusters()), 1, 1)
            split = False
        else:
            items = (decision.rows // tile_m) * (2 * shard // n_tile)
            grid = (
                gemm1_dense.cluster_split_grid(items, sms)
                if split
                else (min(items, sms), 1, 1)
            )
        bindings = plan.gemm1_dense_launch_bindings(
            b,
            weights=g1_weights,
            x=x,
            x_sf=x_sf,
            hidden=hidden,
            shard=shard,
            top_k=top_k,
            num_tokens=tokens,
            rows=decision.rows,
            tile_m=tile_m,
            n_tile=n_tile,
            row_group_list=row_group,
            zero_fill=zero_fill,
            secondary=secondary,
            output=output,
            beta=beta,
            linear_beta=linear_beta,
            beta_stride=1,
            linear_beta_stride=1,
            unused=unused,
            dual=dual,
            **(
                {
                    "cluster_split_max_tiles": (
                        module.max_active_clusters() if split else None
                    )
                }
                if not alternate
                else {}
            ),
        )
        return module, grid, bindings

    def gemm2(n_tile: int, cta_group: int, cluster_n: int, raster, *, alternate: bool):
        raster_m, swizzle = raster
        module = _module(
            index,
            "gemm2",
            (backend, n_tile),
            (
                ("use_pdl", cfg["pdl_alt"] if alternate else cfg["pdl"]),
                ("cta_group", cta_group),
                ("cluster_n", cluster_n),
                ("raster_along_m", raster_m),
                ("swizzle", swizzle),
            ),
        )
        n_tiles = -(-hidden // n_tile)
        if alternate:
            tiles = decision.rows // alt_tile
            pair = cta_group * cluster_n
            grid = (pair * min(tiles * n_tiles, sms // pair), 1, 1)
        else:
            tiles = decision.tiles
            grid = (
                cluster_n * min(tiles * (-(-n_tiles // cluster_n)), sms // cluster_n),
                1,
                1,
            )
        bindings = plan.gemm2_dense_launch_bindings(
            b,
            w2=weights["w2"],
            w2_sf=weights["w2_sf"],
            route_weights=route_weights,
            hidden=hidden,
            shard=shard,
            top_k=top_k,
            tiles=tiles,
            rows=decision.rows,
            n_tile=n_tile,
            output=output,
            dbg=dbg,
            unused=unused,
            dual=dual if not alternate else True,
            **({"secondary": True} if alternate else {}),
        )
        return module, grid, bindings

    launches.append(gemm1(cfg["gemm1"], forms[3], alternate=False))
    if dual:
        launches.append(gemm1(cfg["gemm1_alt"], forms[4], alternate=True))
    n2, cluster_n = cfg["gemm2"]
    launches.append(gemm2(n2, 1, cluster_n, cfg["gemm2_raster"], alternate=False))
    if dual:
        alt_n2, alt_cta_group, alt_cluster_n = cfg["gemm2_alt"]
        launches.append(
            gemm2(
                alt_n2,
                alt_cta_group,
                alt_cluster_n,
                cfg["gemm2_alt_raster"],
                alternate=True,
            )
        )
    with torch.cuda.device(device):
        for module, grid, bindings in launches:
            module.launch(grid=grid, **bindings)
