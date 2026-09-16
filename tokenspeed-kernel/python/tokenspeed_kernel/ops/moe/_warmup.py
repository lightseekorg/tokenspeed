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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.ops.tuning import set_autotune_max_num_tokens
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, WarmupBehavior

__all__ = ["MoeApplyWarmupConfig"]


def _fields(raw: Mapping[str, object], names: set[str], context: str) -> None:
    missing = sorted(names - set(raw))
    unknown = sorted(set(raw) - names)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _mapping(
    raw: Mapping[str, object], name: str, context: str
) -> Mapping[str, object]:
    value = raw[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"{context}.{name} must be an object")
    return value


def _integer(raw: Mapping[str, object], name: str, context: str) -> int:
    value = raw[name]
    if type(value) is not int:
        raise TypeError(f"{context}.{name} must be an integer")
    return value


def _boolean(raw: Mapping[str, object], name: str, context: str) -> bool:
    value = raw[name]
    if type(value) is not bool:
        raise TypeError(f"{context}.{name} must be a boolean")
    return value


def _string(raw: Mapping[str, object], name: str, context: str) -> str:
    value = raw[name]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{name} must be a non-empty string")
    return value


def _nullable_string(raw: Mapping[str, object], name: str, context: str) -> str | None:
    value = raw[name]
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError(f"{context}.{name} must be a non-empty string or null")
    return value


def _number(raw: Mapping[str, object], name: str, context: str) -> float:
    value = raw[name]
    if type(value) not in {int, float}:
        raise TypeError(f"{context}.{name} must be a number")
    return float(value)


def _nullable_number(
    raw: Mapping[str, object], name: str, context: str
) -> float | None:
    value = raw[name]
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise TypeError(f"{context}.{name} must be a number or null")
    return float(value)


def _dtype(name: str) -> torch.dtype:
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in dtypes:
        raise ValueError(f"unsupported MoE warmup dtype {name!r}")
    return dtypes[name]


def _routing_method(name: str) -> int:
    methods = {
        "default": 0,
        "renormalize": 1,
        "deepseek_v3": 2,
        "llama4": 3,
        "renormalize_naive": 4,
        "topk": 5,
        "sigmoid_renorm": 6,
        "minimax2": 7,
    }
    if name not in methods:
        raise ValueError(f"unsupported MoE routing method {name!r}")
    return methods[name]


@dataclass(frozen=True)
class _MoeShape:
    hidden_size: int
    global_num_experts: int
    local_num_experts: int
    top_k: int
    global_intermediate_size: int
    intermediate_size_per_partition: int

    @classmethod
    def parse(cls, raw: Mapping[str, object], context: str) -> _MoeShape:
        names = {
            "hidden_size",
            "global_num_experts",
            "local_num_experts",
            "top_k",
            "global_intermediate_size",
            "intermediate_size_per_partition",
        }
        _fields(raw, names, context)
        values = {name: _integer(raw, name, context) for name in names}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"{context} values must be positive")
        if values["top_k"] > values["global_num_experts"]:
            raise ValueError(f"{context}.top_k cannot exceed global_num_experts")
        return cls(**values)


@dataclass(frozen=True)
class _MoePlacement:
    moe_tp_size: int
    moe_tp_rank: int
    moe_ep_size: int
    moe_ep_rank: int
    local_expert_offset: int

    @classmethod
    def parse(cls, raw: Mapping[str, object], context: str) -> _MoePlacement:
        names = {
            "moe_tp_size",
            "moe_tp_rank",
            "moe_ep_size",
            "moe_ep_rank",
            "local_expert_offset",
        }
        _fields(raw, names, context)
        values = {name: _integer(raw, name, context) for name in names}
        if values["moe_tp_size"] <= 0 or values["moe_ep_size"] <= 0:
            raise ValueError(f"{context} parallel sizes must be positive")
        if not 0 <= values["moe_tp_rank"] < values["moe_tp_size"]:
            raise ValueError(f"{context}.moe_tp_rank is out of range")
        if not 0 <= values["moe_ep_rank"] < values["moe_ep_size"]:
            raise ValueError(f"{context}.moe_ep_rank is out of range")
        if values["moe_tp_size"] > 1 and values["moe_ep_size"] > 1:
            raise ValueError("MoE TP and EP cannot both exceed one")
        return cls(**values)


@dataclass(frozen=True)
class _TokenDomain:
    policy: str
    maximum: int
    round_up: bool

    @classmethod
    def parse(cls, raw: Mapping[str, object], context: str) -> _TokenDomain:
        _fields(raw, {"policy", "maximum", "round_up"}, context)
        policy = _string(raw, "policy", context)
        maximum = _integer(raw, "maximum", context)
        round_up = _boolean(raw, "round_up", context)
        if policy != "flashinfer_hybrid":
            raise ValueError(f"{context}.policy must be 'flashinfer_hybrid'")
        if maximum <= 0:
            raise ValueError(f"{context}.maximum must be positive")
        if round_up:
            raise ValueError(f"{context}.round_up must be false")
        return cls(policy=policy, maximum=maximum, round_up=round_up)


@dataclass(frozen=True)
class _MoeWeights:
    kind: str
    scale_format: str | None
    block_size: int | tuple[int, int] | None
    layout: str
    shuffle: bool

    @classmethod
    def parse(cls, raw: Mapping[str, object], context: str) -> _MoeWeights:
        _fields(
            raw, {"kind", "scale_format", "block_size", "layout", "shuffle"}, context
        )
        kind = _string(raw, "kind", context)
        if kind not in {"nvfp4", "mxfp4", "fp8", "unquant"}:
            raise ValueError(f"{context}.kind is unsupported: {kind!r}")
        scale_format = _nullable_string(raw, "scale_format", context)
        raw_block_size = raw["block_size"]
        if raw_block_size is None:
            block_size = None
        elif type(raw_block_size) is int:
            block_size = raw_block_size
        elif (
            isinstance(raw_block_size, list)
            and len(raw_block_size) == 2
            and all(type(value) is int for value in raw_block_size)
        ):
            block_size = (raw_block_size[0], raw_block_size[1])
        else:
            raise TypeError(f"{context}.block_size must be an integer, pair, or null")
        layout = _string(raw, "layout", context)
        shuffle = _boolean(raw, "shuffle", context)
        if layout != "backend_required" or not shuffle:
            raise ValueError(
                f"{context} requires layout='backend_required' and shuffle=true"
            )
        expected = {
            "nvfp4": ("ue4m3", 16),
            "mxfp4": ("ue8m0", 32),
            "fp8": ("float32", (128, 128)),
            "unquant": (None, None),
        }[kind]
        if (scale_format, block_size) != expected:
            raise ValueError(
                f"{context} requires scale_format={expected[0]!r}, "
                f"block_size={expected[1]!r} for {kind}"
            )
        return cls(
            kind=kind,
            scale_format=scale_format,
            block_size=block_size,
            layout=layout,
            shuffle=shuffle,
        )


@dataclass(frozen=True)
class _MoeActivation:
    kind: str
    alpha: float | None
    beta: float | None
    clamp: float | None

    @classmethod
    def parse(cls, raw: Mapping[str, object], context: str) -> _MoeActivation:
        _fields(raw, {"kind", "alpha", "beta", "clamp"}, context)
        kind = _string(raw, "kind", context)
        if kind not in {"silu", "swiglu", "situ"}:
            raise ValueError(f"{context}.kind is unsupported: {kind!r}")
        alpha = _nullable_number(raw, "alpha", context)
        beta = _nullable_number(raw, "beta", context)
        clamp = _nullable_number(raw, "clamp", context)
        if kind == "situ" and (
            alpha is None or alpha <= 0 or beta is None or beta <= 0
        ):
            raise ValueError(f"{context} SiTU alpha and beta must be positive")
        return cls(kind=kind, alpha=alpha, beta=beta, clamp=clamp)


@dataclass(frozen=True)
class _MoeRouting:
    kind: str
    with_bias: bool
    method: str
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    renormalize: bool
    generator: str

    @classmethod
    def parse(cls, raw: Mapping[str, object], context: str) -> _MoeRouting:
        names = {
            "kind",
            "with_bias",
            "method",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
            "renormalize",
            "generator",
        }
        _fields(raw, names, context)
        kind = _string(raw, "kind", context)
        if kind not in {"kernel_routing", "precomputed_topk"}:
            raise ValueError(f"{context}.kind is unsupported: {kind!r}")
        n_group = _integer(raw, "n_group", context)
        topk_group = _integer(raw, "topk_group", context)
        if n_group <= 0 or topk_group <= 0 or topk_group > n_group:
            raise ValueError(f"{context} group counts are invalid")
        generator = _string(raw, "generator", context)
        if generator != "balanced_distinct_topk.v1":
            raise ValueError(f"{context}.generator is unsupported: {generator!r}")
        method = _string(raw, "method", context)
        _routing_method(method)
        routed_scaling_factor = _number(raw, "routed_scaling_factor", context)
        if routed_scaling_factor <= 0:
            raise ValueError(f"{context}.routed_scaling_factor must be positive")
        return cls(
            kind=kind,
            with_bias=_boolean(raw, "with_bias", context),
            method=method,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=_boolean(raw, "renormalize", context),
            generator=generator,
        )


@dataclass(frozen=True)
class _MoeTuning:
    strategy: str
    warmup_iterations: int
    coarse_iterations: int
    fine_iterations: int
    use_cold_l2: bool
    use_cuda_graph: bool
    weight_copies: int
    random_seed: int
    require_complete_coverage: bool

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> _MoeTuning:
        context = "moe.apply tuning"
        names = {
            "strategy",
            "warmup_iterations",
            "coarse_iterations",
            "fine_iterations",
            "use_cold_l2",
            "use_cuda_graph",
            "weight_copies",
            "random_seed",
            "require_complete_coverage",
        }
        _fields(raw, names, context)
        strategy = _string(raw, "strategy", context)
        if strategy not in {"flashinfer_native", "two_stage_with_heuristic_floor"}:
            raise ValueError(f"{context}.strategy is unsupported: {strategy!r}")
        integers = {
            name: _integer(raw, name, context)
            for name in (
                "warmup_iterations",
                "coarse_iterations",
                "fine_iterations",
                "weight_copies",
                "random_seed",
            )
        }
        if (
            any(value < 0 for value in integers.values())
            or integers["weight_copies"] == 0
        ):
            raise ValueError(
                f"{context} iteration counts and weight copies are invalid"
            )
        return cls(
            strategy=strategy,
            warmup_iterations=integers["warmup_iterations"],
            coarse_iterations=integers["coarse_iterations"],
            fine_iterations=integers["fine_iterations"],
            use_cold_l2=_boolean(raw, "use_cold_l2", context),
            use_cuda_graph=_boolean(raw, "use_cuda_graph", context),
            weight_copies=integers["weight_copies"],
            random_seed=integers["random_seed"],
            require_complete_coverage=_boolean(
                raw, "require_complete_coverage", context
            ),
        )


@dataclass(frozen=True)
class _MoeCase:
    expected_registration: str | None
    shape: _MoeShape
    placement: _MoePlacement
    token_domain: _TokenDomain
    weights: _MoeWeights
    input_dtype: str
    output_dtype: str
    activation: _MoeActivation
    routing: _MoeRouting
    finalization: str
    enable_pdl: bool
    per_token_scale: bool
    gemm1_lora_delta: bool
    fused_shared_experts: int

    @classmethod
    def parse(cls, raw: Mapping[str, object], index: int) -> _MoeCase:
        context = f"moe.apply case {index}"
        names = {
            "expected_registration",
            "shape",
            "placement",
            "token_domain",
            "weights",
            "input_dtype",
            "output_dtype",
            "activation",
            "routing",
            "finalization",
            "enable_pdl",
            "per_token_scale",
            "gemm1_lora_delta",
            "fused_shared_experts",
        }
        _fields(raw, names, context)
        expected = raw["expected_registration"]
        if expected is not None and (not isinstance(expected, str) or not expected):
            raise TypeError(f"{context}.expected_registration must be a string or null")
        finalization_raw = _mapping(raw, "finalization", context)
        _fields(finalization_raw, {"kind"}, f"{context}.finalization")
        finalization = _string(finalization_raw, "kind", f"{context}.finalization")
        if finalization not in {"finalized", "deferred"}:
            raise ValueError(f"{context}.finalization.kind is unsupported")
        fused_shared_experts = _integer(raw, "fused_shared_experts", context)
        if fused_shared_experts != 0:
            raise ValueError(f"{context} fused shared experts are not supported yet")
        if _boolean(raw, "per_token_scale", context):
            raise ValueError(f"{context} per-token scales are not supported yet")
        if _boolean(raw, "gemm1_lora_delta", context):
            raise ValueError(f"{context} GEMM1 LoRA delta is not supported yet")
        case = cls(
            expected_registration=expected,
            shape=_MoeShape.parse(_mapping(raw, "shape", context), f"{context}.shape"),
            placement=_MoePlacement.parse(
                _mapping(raw, "placement", context), f"{context}.placement"
            ),
            token_domain=_TokenDomain.parse(
                _mapping(raw, "token_domain", context), f"{context}.token_domain"
            ),
            weights=_MoeWeights.parse(
                _mapping(raw, "weights", context), f"{context}.weights"
            ),
            input_dtype=_string(raw, "input_dtype", context),
            output_dtype=_string(raw, "output_dtype", context),
            activation=_MoeActivation.parse(
                _mapping(raw, "activation", context), f"{context}.activation"
            ),
            routing=_MoeRouting.parse(
                _mapping(raw, "routing", context), f"{context}.routing"
            ),
            finalization=finalization,
            enable_pdl=_boolean(raw, "enable_pdl", context),
            per_token_scale=False,
            gemm1_lora_delta=False,
            fused_shared_experts=fused_shared_experts,
        )
        case.validate(context)
        return case

    def validate(self, context: str) -> None:
        _dtype(self.input_dtype)
        if _dtype(self.output_dtype) is not torch.bfloat16:
            raise ValueError(f"{context}.output_dtype must be bfloat16")
        if (
            self.shape.local_num_experts * self.placement.moe_ep_size
            != self.shape.global_num_experts
        ):
            raise ValueError(f"{context} local expert geometry is inconsistent with EP")
        expected_offset = self.placement.moe_ep_rank * self.shape.local_num_experts
        if self.placement.local_expert_offset != expected_offset:
            raise ValueError(
                f"{context}.local_expert_offset is inconsistent with EP rank"
            )
        expected_global_i = (
            self.shape.intermediate_size_per_partition * self.placement.moe_tp_size
        )
        if self.shape.global_intermediate_size != expected_global_i:
            raise ValueError(f"{context} intermediate geometry is inconsistent with TP")
        block = (
            128
            if self.weights.kind in {"fp8", "unquant"}
            else int(self.weights.block_size)
        )
        if (
            self.shape.hidden_size % block
            or self.shape.intermediate_size_per_partition % block
        ):
            raise ValueError(
                f"{context} dimensions do not satisfy weight block alignment"
            )
        if (
            self.weights.kind == "mxfp4"
            and self.activation.kind == "situ"
            and self.input_dtype != "bfloat16"
        ):
            raise ValueError(f"{context} MXFP4 SiTU requires bfloat16 input")
        if (
            self.weights.kind == "nvfp4"
            and self.activation.kind == "situ"
            and self.routing.kind != "precomputed_topk"
        ):
            raise ValueError(f"{context} NVFP4 SiTU requires precomputed routing")

    @property
    def internal_activation_dtype(self) -> str:
        if self.weights.kind == "mxfp4" and self.activation.kind == "situ":
            return "fp8"
        return "input"


class _SyntheticMoeWeights(torch.nn.Module):
    def __init__(
        self, case: _MoeCase, generator: torch.Generator, device: torch.device
    ):
        super().__init__()
        e = case.shape.local_num_experts
        h = case.shape.hidden_size
        i = case.shape.intermediate_size_per_partition
        kind = case.weights.kind
        if kind in {"nvfp4", "mxfp4"}:
            block = int(case.weights.block_size)
            self.w13_weight = self._parameter(
                torch.randint(
                    0,
                    256,
                    (e, 2 * i, h // 2),
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                )
            )
            self.w2_weight = self._parameter(
                torch.randint(
                    0,
                    256,
                    (e, h, i // 2),
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                )
            )
            scale_dtype = torch.float8_e4m3fn if kind == "nvfp4" else torch.uint8
            self.w13_weight_scale = self._parameter(
                torch.randint(
                    1,
                    127,
                    (e, 2 * i, h // block),
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                ).view(scale_dtype)
            )
            self.w2_weight_scale = self._parameter(
                torch.randint(
                    1,
                    127,
                    (e, h, i // block),
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                ).view(scale_dtype)
            )
            if kind == "nvfp4":
                self.w13_weight_scale_2 = self._parameter(
                    torch.ones(e, dtype=torch.float32, device=device)
                )
                self.w2_weight_scale_2 = self._parameter(
                    torch.ones(e, dtype=torch.float32, device=device)
                )
                self.w13_input_scale = self._parameter(
                    torch.ones(e, dtype=torch.float32, device=device)
                )
                self.w2_input_scale = self._parameter(
                    torch.ones(e, dtype=torch.float32, device=device)
                )
        elif kind == "fp8":
            self.w13_weight = self._parameter(
                torch.randint(
                    0,
                    120,
                    (e, 2 * i, h),
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                ).view(torch.float8_e4m3fn)
            )
            self.w2_weight = self._parameter(
                torch.randint(
                    0,
                    120,
                    (e, h, i),
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                ).view(torch.float8_e4m3fn)
            )
            self.w13_weight_scale_inv = self._parameter(
                torch.ones(
                    (e, 2 * i // 128, h // 128), dtype=torch.float32, device=device
                )
            )
            self.w2_weight_scale_inv = self._parameter(
                torch.ones((e, h // 128, i // 128), dtype=torch.float32, device=device)
            )
        else:
            self.w13_weight = self._parameter(
                torch.zeros((e, 2 * i, h), dtype=torch.bfloat16, device=device)
            )
            self.w2_weight = self._parameter(
                torch.zeros((e, h, i), dtype=torch.bfloat16, device=device)
            )

        self.num_experts = case.shape.global_num_experts
        self.num_local_experts = e
        self.top_k = case.shape.top_k
        self.hidden_size = h
        self.intermediate_size = case.shape.global_intermediate_size
        self.tp_size = case.placement.moe_tp_size
        self.ep_rank = case.placement.moe_ep_rank
        self.w13_input_layout = "concatenated"
        self.activation_situ_beta = case.activation.alpha
        self.activation_situ_linear_beta = case.activation.beta
        if case.activation.kind != "situ" and any(
            value is not None
            for value in (
                case.activation.alpha,
                case.activation.beta,
                case.activation.clamp,
            )
        ):
            self.swiglu_arg = SimpleNamespace(
                alpha=case.activation.alpha,
                limit=case.activation.clamp,
            )
            self.swiglu_beta = case.activation.beta
        method = _routing_method(case.routing.method)
        correction_bias = (
            torch.zeros(
                case.shape.global_num_experts, dtype=torch.float32, device=device
            )
            if case.routing.with_bias
            else None
        )
        self.routing_config = {
            "routing_method_type": method,
            "n_group": case.routing.n_group,
            "topk_group": case.routing.topk_group,
            "routed_scaling_factor": case.routing.routed_scaling_factor,
            "normalize_topk_weights": case.routing.renormalize,
            "correction_bias": correction_bias,
        }
        self._routing_method_type = method
        self._n_group = case.routing.n_group
        self._topk_group = case.routing.topk_group
        self._routed_scaling_factor = case.routing.routed_scaling_factor
        self._normalize_topk_weights = case.routing.renormalize
        self._correction_bias = correction_bias
        self._routing_logits_dtype = (
            torch.float32 if method in {2, 7} else torch.bfloat16
        )
        self._spec = SimpleNamespace(
            num_experts=case.shape.global_num_experts,
            num_local_experts=e,
            top_k=case.shape.top_k,
            ep_rank=case.placement.moe_ep_rank,
        )

    @staticmethod
    def _parameter(value: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(value, requires_grad=False)


@dataclass(frozen=True)
class _MoeInvocation:
    plan: dict
    weights: torch.nn.Module
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_weights: torch.Tensor | None
    topk_ids: torch.Tensor | None
    do_finalize: bool
    enable_pdl: bool

    def run(self) -> None:
        moe_apply(
            plan=self.plan,
            x=self.hidden_states,
            w=self.weights,
            router_logits=self.router_logits,
            topk_weights=self.topk_weights,
            topk_ids=self.topk_ids,
            num_tokens_global=None,
            max_num_tokens_per_gpu=None,
            do_finalize=self.do_finalize,
            low_latency=None,
            overlap_fn=None,
            shared_input=None,
            shared_weight=None,
            shared_out=None,
            enable_pdl=self.enable_pdl,
        )


@dataclass(frozen=True)
class _PreparedMoeWarmup:
    _registration: KernelSpec
    invocations: tuple[_MoeInvocation, ...]

    @property
    def registration(self) -> KernelSpec:
        return self._registration

    def run(self) -> None:
        for invocation in self.invocations:
            invocation.run()


@dataclass(frozen=True)
class MoeApplyWarmupConfig:
    version: int
    tuning: _MoeTuning
    cases: tuple[_MoeCase, ...]

    @property
    def maximum_num_tokens(self) -> int:
        return max(case.token_domain.maximum for case in self.cases)

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> MoeApplyWarmupConfig:
        _fields(raw, {"version", "tuning", "cases"}, "moe.apply definition")
        version = _integer(raw, "version", "moe.apply definition")
        if version != 1:
            raise ValueError("moe.apply definition version must be 1")
        raw_cases = raw["cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise TypeError("moe.apply definition.cases must be a non-empty list")
        cases = []
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, Mapping):
                raise TypeError(f"moe.apply case {index} must be an object")
            cases.append(_MoeCase.parse(raw_case, index))
        return cls(
            version=version,
            tuning=_MoeTuning.parse(_mapping(raw, "tuning", "moe.apply definition")),
            cases=tuple(cases),
        )

    def prepare(self, solution: str, platform: PlatformInfo) -> _PreparedMoeWarmup:
        if platform.vendor != "nvidia" or platform.arch_version < ArchVersion(10, 0):
            raise ValueError("moe.apply FlashInfer warmup requires NVIDIA Blackwell")
        if not torch.cuda.is_available():
            raise RuntimeError("moe.apply warmup requires CUDA")
        if self.tuning.strategy != "flashinfer_native":
            raise NotImplementedError(
                "two_stage_with_heuristic_floor warmup is not implemented yet"
            )
        set_autotune_max_num_tokens(self.maximum_num_tokens)

        planned = []
        registration = None
        registry = KernelRegistry.get()
        for index, case in enumerate(self.cases):
            plan = moe_plan(
                weight_dtype=case.weights.kind,
                input_dtype=_dtype(case.input_dtype),
                activation=case.activation.kind,
                requires_deferred_finalize=case.finalization == "deferred",
                routing_mode=case.routing.kind,
                a2a_backend=None,
                ep_size=case.placement.moe_ep_size,
                ispp=case.shape.intermediate_size_per_partition,
                fp8_scale_block_shape=(
                    case.weights.block_size if case.weights.kind == "fp8" else None
                ),
                internal_activation_dtype=case.internal_activation_dtype,
                with_bias=False,
                deepep_group=None,
                deepep_mode=None,
                deepep_low_latency_max_num_tokens_per_gpu=None,
                solution=solution,
            )
            spec = registry.get_by_name(plan["apply_kernel_name"])
            if spec is None:
                raise RuntimeError(
                    f"selected MoE kernel {plan['apply_kernel_name']!r} is not registered"
                )
            if spec.warmup_behavior is not WarmupBehavior.FLASHINFER_AUTOTUNE:
                raise ValueError(
                    f"selected MoE kernel {spec.name!r} is not FlashInfer-autotunable"
                )
            if (
                case.expected_registration is not None
                and spec.name != case.expected_registration
            ):
                raise ValueError(
                    f"moe.apply case {index} selected {spec.name!r}, expected {case.expected_registration!r}"
                )
            if registration is not None and registration.name != spec.name:
                raise ValueError(
                    "one moe.apply definition cannot resolve to multiple registrations"
                )
            registration = spec
            planned.append((case, plan))

        device = torch.device("cuda", torch.cuda.current_device())
        invocations = []
        for index, (case, plan) in enumerate(planned):
            generator = torch.Generator(device=device).manual_seed(
                self.tuning.random_seed + index
            )
            weights = _SyntheticMoeWeights(case, generator, device)
            moe_process_weights(plan, weights)
            m = case.token_domain.maximum
            hidden_states = torch.randn(
                (m, case.shape.hidden_size),
                dtype=_dtype(case.input_dtype),
                device=device,
                generator=generator,
            )
            router_logits = torch.randn(
                (m, case.shape.global_num_experts),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            if case.routing.kind == "precomputed_topk":
                rows = torch.arange(m, dtype=torch.int64, device=device)[:, None]
                columns = torch.arange(
                    case.shape.top_k, dtype=torch.int64, device=device
                )[None, :]
                topk_ids = (
                    (rows * case.shape.top_k + columns) % case.shape.global_num_experts
                ).to(torch.int32)
                topk_weights = torch.full(
                    (m, case.shape.top_k),
                    case.routing.routed_scaling_factor / case.shape.top_k,
                    dtype=torch.bfloat16,
                    device=device,
                )
            else:
                topk_ids = None
                topk_weights = None
            invocations.append(
                _MoeInvocation(
                    plan=plan,
                    weights=weights,
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    do_finalize=case.finalization == "finalized",
                    enable_pdl=case.enable_pdl,
                )
            )

        if registration is None:
            raise RuntimeError("moe.apply definition has no cases")
        return _PreparedMoeWarmup(
            _registration=registration,
            invocations=tuple(invocations),
        )
