"""Tensor Gluon MegaMoE API."""

import enum
from dataclasses import dataclass, field

import torch
from petit_kernel import ops
from petit_kernel.moe_mxfp4 import (
    MoeKernelLayout,
    repack_moe_kernel_layout,
)


class DataType(enum.Enum):
    int4 = 0
    float8_e4m3fn = 1
    float4_e2m1 = 2
    float16 = 3
    bfloat16 = 4
    float8_e5m2fn = 5
    mxfloat4_e2m1 = 6


class _FusedMoeDataType(enum.IntEnum):
    none = 0
    mxfp4 = 1
    nvfp4 = 2
    channel_scale_fp8 = 3
    blockscale_fp8 = 4
    bf16 = 5


class _FusedMoeWeightOrdering(enum.IntEnum):
    native_mxfp4 = 0
    petit_mxfp4 = 1
    petit_fp8 = 2


class _FusedMoeStages(enum.IntEnum):
    one_stage = 0
    two_stage = 1


class _FusedMoeMfmaShape(enum.IntEnum):
    mfma_fp8_16x16x32 = 0
    mfma_bf16_mxfp4 = 1
    mfma_scale_fp4_mxfp4 = 2


class _FusedMoeActivationFunction(enum.IntEnum):
    silu_dot = 0
    openai_swiglu = 1
    kimi_situ = 2


class _FusedMoeStage1Buffering(enum.IntEnum):
    single_buffer = 0
    double_buffer = 1


class _FusedMoeWeightLoadPolicy(enum.IntEnum):
    cached = 0
    non_temporal = 1


class MegaMoeActivation(enum.Enum):
    bf16 = "bf16"
    mxfp4 = "mxfp4"


class MegaMoeActivationFunction(enum.Enum):
    silu = "silu"
    swiglu = "swiglu"
    kimi_situ = "kimi_situ"


class MegaMoeStages(enum.IntEnum):
    one_stage = 0
    two_stage = 1


class _MegaMoeTileShape(enum.IntEnum):
    n256 = 0
    n128 = 1


class _MegaMoeProducerGeometry(enum.IntEnum):
    cta56 = 0
    cta64 = 1
    cta128 = 2
    cta192 = 3


def _make_fused_moe_base_solution_id(
    activation_type: _FusedMoeDataType | int,
    weight_type: _FusedMoeDataType | int,
    bias_type: _FusedMoeDataType | int,
    weight_ordering: _FusedMoeWeightOrdering | int,
    mfma: _FusedMoeMfmaShape | int,
    stages: _FusedMoeStages | int,
    activation: _FusedMoeActivationFunction | int,
    stage1_buffering: _FusedMoeStage1Buffering | int,
    weight_load_policy: _FusedMoeWeightLoadPolicy | int = (
        _FusedMoeWeightLoadPolicy.cached
    ),
) -> int:
    return (
        (int(_FusedMoeDataType(activation_type)) & 0xF)
        | ((int(_FusedMoeDataType(weight_type)) & 0xF) << 4)
        | ((int(_FusedMoeDataType(bias_type)) & 0xF) << 8)
        | ((int(_FusedMoeWeightOrdering(weight_ordering)) & 0x3) << 12)
        | ((int(_FusedMoeMfmaShape(mfma)) & 0x3) << 14)
        | ((int(_FusedMoeStages(stages)) & 0xF) << 16)
        | ((int(_FusedMoeActivationFunction(activation)) & 0x7) << 20)
        | ((int(_FusedMoeStage1Buffering(stage1_buffering)) & 0x1) << 23)
        | ((int(_FusedMoeWeightLoadPolicy(weight_load_policy)) & 0x1) << 40)
    )


def _make_mega_moe_solution_id(
    activation_type: _FusedMoeDataType | int,
    num_ranks: int,
    num_experts: int,
    topk: int,
    hidden_size: int,
    *,
    inter_dim: int,
    producer_geometry: _MegaMoeProducerGeometry | int,
    stages: _FusedMoeStages | int = _FusedMoeStages.one_stage,
    w2_tile_shape: _MegaMoeTileShape | int = _MegaMoeTileShape.n256,
    activation_function: _FusedMoeActivationFunction | int = (
        _FusedMoeActivationFunction.openai_swiglu
    ),
    has_bias: bool = True,
) -> int:
    activation_type = _FusedMoeDataType(activation_type)
    stages = _FusedMoeStages(stages)
    w2_tile_shape = _MegaMoeTileShape(w2_tile_shape)
    producer_geometry = _MegaMoeProducerGeometry(producer_geometry)
    if activation_type not in (_FusedMoeDataType.bf16, _FusedMoeDataType.mxfp4):
        raise ValueError("MegaMoE activation type must be bf16 or mxfp4")
    if num_ranks not in (2, 4, 8):
        raise ValueError("MegaMoE num_ranks must be one of 2, 4, 8")
    if (
        num_experts < 32
        or num_experts > 1024
        or num_experts % 32
        or num_experts % num_ranks
    ):
        raise ValueError(
            "MegaMoE num_experts must be a 32 multiple in [32, 1024] "
            "and be divisible by ranks"
        )
    if topk <= 0 or topk > 31 or topk > num_experts:
        raise ValueError("MegaMoE topk must be in [1, min(31, num_experts)]")
    if hidden_size <= 0 or hidden_size % 64 or hidden_size // 64 > 255:
        raise ValueError("MegaMoE hidden_size must be a positive 64 multiple")
    if inter_dim <= 0 or inter_dim % 512 or inter_dim // 512 > 32:
        raise ValueError(
            "MegaMoE inter_dim must be a positive 512 multiple at most 16384"
        )
    activation_function = _FusedMoeActivationFunction(activation_function)
    mfma = (
        _FusedMoeMfmaShape.mfma_bf16_mxfp4
        if activation_type == _FusedMoeDataType.bf16
        else _FusedMoeMfmaShape.mfma_scale_fp4_mxfp4
    )
    base = _make_fused_moe_base_solution_id(
        activation_type,
        _FusedMoeDataType.mxfp4,
        _FusedMoeDataType.bf16 if has_bias else _FusedMoeDataType.none,
        _FusedMoeWeightOrdering.native_mxfp4,
        mfma,
        _FusedMoeStages(stages),
        activation_function,
        _FusedMoeStage1Buffering.double_buffer,
    )
    return (
        base
        | ((num_ranks.bit_length() - 1) << 24)
        | (((num_experts // 32 - 1) & 15) << 26)
        | (((num_experts // 32 - 1) >> 4) << 50)
        | ((topk & 15) << 30)
        | ((topk >> 4) << 51)
        | ((hidden_size // 64) << 34)
        | (int(w2_tile_shape) << 42)
        | ((inter_dim // 512 - 1) << 43)
        | (int(producer_geometry) << 48)
    )


def _with_mega_moe_producer_geometry(
    solution_id: int, producer_geometry: _MegaMoeProducerGeometry | int
) -> int:
    producer_geometry = _MegaMoeProducerGeometry(producer_geometry)
    mask = 0x3 << 48
    return (int(solution_id) & ~mask) | (int(producer_geometry) << 48)


def _select_mega_moe_producer_geometry(
    num_experts: int,
    activation_function: MegaMoeActivationFunction | str,
    num_tokens: int,
) -> _MegaMoeProducerGeometry:
    activation_function = MegaMoeActivationFunction(activation_function)
    if activation_function is MegaMoeActivationFunction.kimi_situ:
        return _MegaMoeProducerGeometry.cta56
    if num_experts <= 56:
        return _MegaMoeProducerGeometry.cta56
    if activation_function is MegaMoeActivationFunction.silu:
        if num_experts == 256 and 12 <= num_tokens < 1024:
            return _MegaMoeProducerGeometry.cta192
        if num_experts == 384 and num_tokens < 512:
            return _MegaMoeProducerGeometry.cta192
    if num_tokens < 12:
        return _MegaMoeProducerGeometry.cta128
    if num_tokens < 24:
        return _MegaMoeProducerGeometry.cta64
    return _MegaMoeProducerGeometry.cta56


def create_vmm_symmetric_heap(world_size: int):
    """Collectively create an intra-node VMM symmetric heap."""
    return ops.VmmSymmetricHeap(int(world_size))


@dataclass(frozen=True)
class MegaMoeInputViews:
    tokens: torch.Tensor
    scales: torch.Tensor | None
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor


@dataclass(frozen=True)
class MegaMoeConfig:
    world_size: int
    num_experts: int
    topk: int
    model_dim: int
    activation: MegaMoeActivation
    activation_function: MegaMoeActivationFunction = MegaMoeActivationFunction.swiglu
    stages: MegaMoeStages = MegaMoeStages.two_stage
    inter_dim: int = 3072
    has_bias: bool = True
    max_tokens_per_rank: int = field(init=False, default=1024)
    _solution_id: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        activation = MegaMoeActivation(self.activation)
        activation_function = MegaMoeActivationFunction(self.activation_function)
        stages = MegaMoeStages(self.stages)
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "activation_function", activation_function)
        object.__setattr__(self, "stages", stages)
        if self.world_size not in (2, 4, 8):
            raise ValueError("MegaMoE world_size must be one of 2, 4, 8")
        gpt_oss = (
            self.num_experts == 32
            and self.topk == 4
            and self.model_dim == 2880
            and self.inter_dim == 3072
        )
        gpt_oss_120b = (
            self.world_size == 8
            and self.num_experts == 128
            and self.topk == 4
            and self.model_dim == 2880
            and self.inter_dim == 3072
        )
        deepseek_v32 = (
            self.world_size == 8
            and self.num_experts == 256
            and self.topk == 8
            and self.model_dim == 7168
            and self.inter_dim == 2048
        )
        deepseek_v4 = (
            self.world_size == 8
            and self.num_experts == 384
            and self.topk == 6
            and self.model_dim == 7168
            and self.inter_dim == 3072
        )
        gpt_oss_config = (
            (gpt_oss or gpt_oss_120b)
            and activation_function is MegaMoeActivationFunction.swiglu
            and self.has_bias
        )
        deepseek_config = (
            (deepseek_v32 or deepseek_v4)
            and activation_function is MegaMoeActivationFunction.silu
            and not self.has_bias
        )
        kimi_config = (
            self.world_size == 8
            and self.num_experts == 896
            and self.topk == 16
            and self.model_dim == 3584
            and self.inter_dim == 3072
            and not self.has_bias
            and activation_function is MegaMoeActivationFunction.kimi_situ
        )
        supported = (
            stages is MegaMoeStages.two_stage
            and activation is MegaMoeActivation.mxfp4
            and (gpt_oss_config or deepseek_config or kimi_config)
        )
        if not supported:
            raise ValueError("unsupported registered MegaMoE configuration")
        object.__setattr__(
            self,
            "_solution_id",
            _make_mega_moe_solution_id(
                (
                    _FusedMoeDataType.bf16
                    if activation is MegaMoeActivation.bf16
                    else _FusedMoeDataType.mxfp4
                ),
                self.world_size,
                self.num_experts,
                self.topk,
                self.model_dim,
                inter_dim=self.inter_dim,
                producer_geometry=_MegaMoeProducerGeometry.cta56,
                stages=_FusedMoeStages(stages),
                w2_tile_shape=_MegaMoeTileShape.n256,
                activation_function=(
                    _FusedMoeActivationFunction.silu_dot
                    if activation_function is MegaMoeActivationFunction.silu
                    else (
                        _FusedMoeActivationFunction.kimi_situ
                        if activation_function is MegaMoeActivationFunction.kimi_situ
                        else _FusedMoeActivationFunction.openai_swiglu
                    )
                ),
                has_bias=self.has_bias,
            ),
        )

    @property
    def compute_model_dim(self) -> int:
        return ((self.model_dim + 511) // 512) * 512

    def _solution_id_for_tokens(self, num_tokens: int) -> int:
        return _with_mega_moe_producer_geometry(
            self._solution_id,
            _select_mega_moe_producer_geometry(
                self.num_experts, self.activation_function, num_tokens
            ),
        )

    def input_views(self, heap: object, max_tokens: int) -> MegaMoeInputViews:
        max_tokens = int(max_tokens)
        if max_tokens <= 0 or max_tokens > self.max_tokens_per_rank:
            raise ValueError("invalid MegaMoE token capacity")
        tokens, scales, expert_ids, expert_weights = ops.mega_moe_workspace_input_views(
            heap, max_tokens, self._solution_id
        )
        return MegaMoeInputViews(
            tokens,
            None if scales is None else scales,
            expert_ids,
            expert_weights,
        )

    def quantize(
        self,
        input: torch.Tensor,
        *,
        out: MegaMoeInputViews | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation is not MegaMoeActivation.mxfp4:
            raise ValueError("quantize requires MXFP4 activation")
        if input.ndim != 2 or input.shape[1] != self.model_dim:
            raise ValueError("input must have shape [num_tokens, model_dim]")
        if out is not None and out.scales is None:
            raise ValueError("MXFP4 output must include scales")
        return ops.mega_moe_quantize_mxfp4(
            input,
            None if out is None else out.tokens,
            None if out is None else out.scales,
        )

    def run(
        self,
        heap: object,
        w13: torch.Tensor,
        w2: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
        num_tokens: int,
        *,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        inputs: MegaMoeInputViews | None = None,
    ) -> torch.Tensor:
        num_tokens = int(num_tokens)
        if num_tokens < 0 or num_tokens > self.max_tokens_per_rank:
            raise ValueError("invalid MegaMoE token count")
        valid_output_shapes = (
            (num_tokens, self.model_dim),
            (num_tokens, self.compute_model_dim),
        )
        if out is not None and out.shape not in valid_output_shapes:
            raise ValueError(
                "out must have shape [num_tokens, model_dim] or "
                "[num_tokens, compute_model_dim]"
            )
        input_tokens = None
        input_topk_ids = None
        input_topk_weights = None
        if inputs is not None:
            if self.activation is not MegaMoeActivation.mxfp4:
                raise ValueError("external inputs require MXFP4 activation")
            if inputs.scales is None:
                raise ValueError("MXFP4 inputs must include scales")
            if inputs.expert_ids.dtype != torch.int32:
                raise ValueError("input_topk_ids has invalid dtype")
            if (
                (
                    num_tokens > 0
                    and inputs.scales.data_ptr()
                    != inputs.tokens.data_ptr() + self.model_dim // 2
                )
                or inputs.scales.stride(0) != inputs.tokens.stride(0)
                or inputs.scales.stride(1) != 1
            ):
                raise ValueError("scales must be a view into the input rows")
            input_tokens = inputs.tokens
            input_topk_ids = inputs.expert_ids
            input_topk_weights = inputs.expert_weights
        return ops.mega_moe(
            heap,
            w13,
            w2,
            fc1_scale,
            fc2_scale,
            num_tokens,
            self._solution_id_for_tokens(num_tokens),
            w13_bias,
            w2_bias,
            out,
            input_tokens,
            input_topk_ids,
            input_topk_weights,
        )
