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

"""The 64-aligned FlashInfer BF16 MoE launcher: transform, admission, outputs."""

import functools
import importlib
import inspect
import logging

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.ops.moe import moe_process_weights as kernel_moe_process_weights
from tokenspeed_kernel.thirdparty.flashinfer import trtllm_bf16_moe as adapter

_LAUNCHER = """
class FusedMoeLauncher {
 protected:
  int64_t intermediate_size_factor{2};

  void init_common(ActivationType activation_type) {
    this->intermediate_size_factor = isGatedActivation(activation_type) ? 2 : 1;
  }
};

class Bf16MoeLauncher : public FusedMoeLauncher {
 public:
  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();
    if (gemm1_alpha.has_value()) {
      TVM_FFI_ICHECK(activation_type == ActivationType::Swiglu) << "swiglu only";
    }

    TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
        << "the second dimension of weights must be a multiple of 128.";
  }

  void prepare_moe(int64_t& moe_tactic) override {}
};

class Fp8BlockScaleLauncher : public FusedMoeLauncher {
 public:
  void check_moe() const override {
    TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
        << "the second dimension of weights must be a multiple of 128.";
  }
};
"""
_STOCK_CHECK = "TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)"
_STOCK_STATEMENT = (
    f"{_STOCK_CHECK}\n"
    '        << "the second dimension of weights must be a multiple of 128.";'
)
_RELAXED = "intermediate_size_alignment = intermediate_size_factor == 2 ? 64 : 128;"


def _installed_launcher() -> str:
    jit_env = pytest.importorskip("flashinfer.jit.env")
    path = jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_kernel_launcher.cu"
    if not path.exists():
        pytest.skip("installed FlashInfer ships no TRT-LLM MoE launcher source")
    return path.read_text()


def test_only_the_bf16_check_is_relaxed():
    relaxed = adapter._relax_bf16_intermediate_check(_LAUNCHER)
    bf16, fp8 = relaxed.split("class Fp8BlockScaleLauncher")
    assert _RELAXED in bf16 and _STOCK_CHECK not in bf16
    assert "multiple of 128." in fp8 and _RELAXED not in fp8
    # Everything else, including the BF16 check's neighbours, is unchanged.
    stock_bf16, stock_fp8 = _LAUNCHER.split("class Fp8BlockScaleLauncher")
    assert fp8 == stock_fp8
    before, _, after = stock_bf16.partition(_STOCK_CHECK)
    assert bf16.startswith(before)
    assert bf16.endswith(after.partition(";")[2])


@pytest.mark.parametrize(
    "source",
    [
        "",
        _LAUNCHER.replace(
            _STOCK_STATEMENT, _STOCK_STATEMENT.replace("% 128", "% 256"), 1
        ),
        _LAUNCHER.replace(
            _STOCK_STATEMENT, f"{_STOCK_STATEMENT}\n    {_STOCK_STATEMENT}", 1
        ),
        _LAUNCHER.replace(
            "void check_moe() const override {", "void check() const {", 1
        ),
        _LAUNCHER.replace(" ? 2 : 1;", " ? 2 : 3;"),
        _LAUNCHER.replace("class Bf16MoeLauncher", "class Bf16Launcher"),
        _LAUNCHER + _LAUNCHER[_LAUNCHER.index("class Bf16MoeLauncher") :],
    ],
    ids=[
        "empty",
        "other-multiple",
        "check-twice",
        "check-outside-check_moe",
        "unknown-gated-factor",
        "renamed-class",
        "class-twice",
    ],
)
def test_unrecognized_launcher_fails_closed(source):
    with pytest.raises(RuntimeError, match="expected exactly one"):
        adapter._relax_bf16_intermediate_check(source)


def test_installed_launcher_is_relaxed_exactly_once():
    stock = _installed_launcher()
    relaxed = adapter._relax_bf16_intermediate_check(stock)
    assert relaxed.count(_RELAXED) == 1
    # Only the BF16 check moves: every other launcher keeps its % 128 check.
    assert relaxed.count(_STOCK_CHECK) == stock.count(_STOCK_CHECK) - 1
    removed = stock.index(_STOCK_CHECK, stock.index("class Bf16MoeLauncher"))
    assert relaxed[:removed] == stock[:removed]
    tail = stock[removed:].partition(";")[2]
    assert relaxed.endswith(tail)


def test_mutated_installed_launcher_is_refused():
    stock = _installed_launcher()
    start = stock.index("class Bf16MoeLauncher")
    check = stock.index(_STOCK_CHECK, start)
    mutated = stock[:check] + stock[check:].replace("% 128", "% 256", 1)
    with pytest.raises(RuntimeError, match="expected exactly one"):
        adapter._relax_bf16_intermediate_check(mutated)


@pytest.fixture
def flashinfer_jit(monkeypatch, tmp_path):
    """FlashInfer's JIT with an empty workspace and no nvcc; yields its CUDA home."""
    _installed_launcher()
    cpp_ext = pytest.importorskip("flashinfer.jit.cpp_ext")
    jit_env = pytest.importorskip("flashinfer.jit.env")
    cuda_home = tmp_path / "cuda"
    (cuda_home / "bin").mkdir(parents=True)
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: str(cuda_home))
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "cached_ops")
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.delenv("FLASHINFER_NVCC", raising=False)
    adapter.gated_ispp_alignment.cache_clear()
    yield cuda_home
    adapter.gated_ispp_alignment.cache_clear()


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)


def _build_private_module(monkeypatch):
    """Leave a private module built earlier in FlashInfer's JIT workspace."""
    jit_env = pytest.importorskip("flashinfer.jit.env")
    fused_moe = pytest.importorskip("flashinfer.jit.fused_moe")
    from flashinfer.jit.core import JitSpecNvcc

    # FlashInfer's SM100 module, reduced to its name and launcher source.
    launcher = jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_kernel_launcher.cu"
    stock = JitSpecNvcc("fused_moe_trtllm_sm100", [launcher], None, None, None, None)
    monkeypatch.setattr(
        fused_moe, "gen_trtllm_gen_fused_moe_sm100_module", lambda: stock
    )
    library = adapter._relaxed_spec().jit_library_path
    library.parent.mkdir(parents=True)
    library.touch()


def test_gated_alignment_falls_back_to_the_stock_launcher(flashinfer_jit, monkeypatch):
    def refuse(source):
        raise RuntimeError("unrecognized launcher")

    _executable(flashinfer_jit / "bin" / "nvcc")
    assert adapter.gated_ispp_alignment() == adapter.GATED_ISPP_ALIGNMENT
    adapter.gated_ispp_alignment.cache_clear()
    monkeypatch.setattr(adapter, "_relax_bf16_intermediate_check", refuse)
    assert adapter.gated_ispp_alignment() == adapter.STOCK_ISPP_ALIGNMENT


@pytest.mark.parametrize("drift", ["wrapped-entry-point", "no-csrc-dir"])
def test_gated_alignment_falls_back_on_unrecognized_flashinfer(
    flashinfer_jit, monkeypatch, drift
):
    core = pytest.importorskip("flashinfer.fused_moe.core")
    jit_env = pytest.importorskip("flashinfer.jit.env")
    _executable(flashinfer_jit / "bin" / "nvcc")
    if drift == "wrapped-entry-point":
        routed = functools.partial(core.trtllm_bf16_routed_moe)
        monkeypatch.setattr(core, "trtllm_bf16_routed_moe", routed)
    else:
        monkeypatch.delattr(jit_env, "FLASHINFER_CSRC_DIR")
    adapter._entrypoints.cache_clear()
    try:
        assert adapter.gated_ispp_alignment() == adapter.STOCK_ISPP_ALIGNMENT
    finally:
        monkeypatch.undo()
        adapter._entrypoints.cache_clear()


@pytest.mark.parametrize(
    "jit, alignment",
    [
        ("no-nvcc", adapter.STOCK_ISPP_ALIGNMENT),
        ("built", adapter.STOCK_ISPP_ALIGNMENT),
        ("nvcc", adapter.GATED_ISPP_ALIGNMENT),
        ("FLASHINFER_NVCC", adapter.GATED_ISPP_ALIGNMENT),
        ("FLASHINFER_DISABLE_JIT", adapter.STOCK_ISPP_ALIGNMENT),
    ],
)
def test_gated_alignment_needs_flashinfer_nvcc(
    flashinfer_jit, monkeypatch, tmp_path, caplog, jit, alignment
):
    nvcc = flashinfer_jit / "bin" / "nvcc"
    if jit == "built":
        # ninja rebuilds it whenever the build changes (CUDA home, flags,
        # headers), which needs nvcc.
        _build_private_module(monkeypatch)
    elif jit == "nvcc":
        _executable(nvcc)
    elif jit == "FLASHINFER_NVCC":
        _executable(tmp_path / "toolchain" / "nvcc")
        monkeypatch.setenv("FLASHINFER_NVCC", str(tmp_path / "toolchain" / "nvcc"))
    elif jit == "FLASHINFER_DISABLE_JIT":
        # Without JIT, FlashInfer neither compiles nor reuses JIT-built modules.
        _executable(nvcc)
        _build_private_module(monkeypatch)
        monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")

    with caplog.at_level(logging.WARNING, logger=adapter.logger.name):
        assert adapter.gated_ispp_alignment() == alignment
        # Decided once per process: a compiler that appears later is ignored.
        _executable(nvcc)
        assert adapter.gated_ispp_alignment() == alignment
    warnings = [record.getMessage() for record in caplog.records]
    if alignment == adapter.GATED_ISPP_ALIGNMENT:
        assert warnings == []
    else:
        reason = jit if jit == "FLASHINFER_DISABLE_JIT" else "nvcc is missing"
        assert len(warnings) == 1
        assert "keeps FlashInfer's multiple of 128" in warnings[0]
        assert reason in warnings[0]


@pytest.mark.parametrize("routing_mode", [None, "precomputed_topk"])
def test_unbuildable_private_module_keeps_trtllm_at_128(
    b200_platform, flashinfer_jit, monkeypatch, routing_mode
):
    """Without nvcc, only multiples of 128 select TRT-LLM."""
    from tokenspeed_kernel.platform import Platform
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import NoKernelFoundError

    unquant = pytest.importorskip("tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant")
    cutlass = pytest.importorskip(
        "tokenspeed_kernel.ops.moe.flashinfer.cutlass_unquant"
    )
    if KernelRegistry.get().get_by_name("flashinfer_trtllm_unquant_moe_apply") is None:
        pytest.skip("flashinfer_trtllm unquant MoE kernels are not registered")

    def planned(ispp, **kwargs):
        return kernel_moe_plan(
            "unquant",
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode=routing_mode,
            ep_size=1,
            ispp=ispp,
            hidden=2048,
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            internal_activation_dtype="input",
            fast_math=True,
            combine_order="rank",
            **kwargs,
        )["solution"]

    # Register both solutions again, as at import, into a scratch registry; the
    # reload rebinds the module's alignment, which teardown restores.
    alignment = unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT
    monkeypatch.setattr(unquant, "TRTLLM_UNQUANT_ISPP_ALIGNMENT", alignment)
    real_platform, real_registry = Platform.get(), KernelRegistry.get()
    try:
        Platform.override(b200_platform)
        KernelRegistry.reset()
        importlib.reload(unquant)
        importlib.reload(cutlass)
        assert unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT == adapter.STOCK_ISPP_ALIGNMENT
        for ispp in (64, 192):
            assert planned(ispp) == "flashinfer_cutlass"
            with pytest.raises(NoKernelFoundError):
                planned(ispp, solution="flashinfer_trtllm")
        assert planned(256) == "flashinfer_trtllm"
    finally:
        Platform.override(real_platform)
        KernelRegistry._instance = real_registry


def test_other_gpus_keep_128_without_checking_flashinfer_jit(
    h100_platform, flashinfer_jit, monkeypatch, caplog
):
    """Outside SM100-SM103 the private launcher is neither checked nor warned about."""
    from tokenspeed_kernel.platform import Platform
    from tokenspeed_kernel.registry import KernelRegistry

    unquant = pytest.importorskip("tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant")
    alignment = unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT
    monkeypatch.setattr(unquant, "TRTLLM_UNQUANT_ISPP_ALIGNMENT", alignment)
    real_platform, real_registry = Platform.get(), KernelRegistry.get()
    try:
        Platform.override(h100_platform)
        KernelRegistry.reset()
        with caplog.at_level(logging.WARNING, logger=adapter.logger.name):
            importlib.reload(unquant)
    finally:
        Platform.override(real_platform)
        KernelRegistry._instance = real_registry
    assert unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT == adapter.STOCK_ISPP_ALIGNMENT
    assert adapter.logger.name not in [record.name for record in caplog.records]


def test_entrypoints_keep_upstream_untouched():
    core = pytest.importorskip("flashinfer.fused_moe.core")
    before = dict(vars(core))
    private = adapter._entrypoints()
    assert vars(core) == before
    for name in ("trtllm_bf16_moe", "trtllm_bf16_routed_moe"):
        assert private[name].__globals__ is private
        assert private[name] is not getattr(core, name)
        assert inspect.signature(private[name]) == inspect.signature(
            getattr(core, name)
        )
    factory = private["_get_trtllm_moe_sm100_module_impl"]
    assert isinstance(factory, functools._lru_cache_wrapper)
    assert inspect.unwrap(factory).__globals__ is private
    assert private["gen_trtllm_gen_fused_moe_sm100_module"] is adapter._relaxed_spec
    register = private["register_custom_op"]
    assert register.keywords == {"prefix": "tokenspeed_flashinfer_bf16_ispp64"}


def test_entrypoints_require_private_dispatch(monkeypatch):
    core = pytest.importorskip("flashinfer.fused_moe.core")

    def bypasses_factory(*args, **kwargs):
        return None

    adapter._entrypoints.cache_clear()
    monkeypatch.setattr(core, "trtllm_bf16_moe", bypasses_factory)
    try:
        with pytest.raises(RuntimeError, match="trtllm_bf16_moe no longer uses"):
            adapter._entrypoints()
    finally:
        monkeypatch.undo()
        adapter._entrypoints.cache_clear()


@pytest.mark.parametrize("ispp", [64, 96, 128, 192, 320])
@pytest.mark.parametrize("routing_mode", [None, "precomputed_topk"])
def test_trtllm_unquant_admits_gated_sizes_the_launcher_accepts(
    b200_platform, ispp, routing_mode
):
    from tokenspeed_kernel.platform import ArchVersion, Platform
    from tokenspeed_kernel.registry import KernelRegistry

    unquant = pytest.importorskip("tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant")
    registry = KernelRegistry.get()
    if registry.get_by_name("flashinfer_trtllm_unquant_moe_apply") is None:
        pytest.skip("flashinfer_trtllm unquant MoE kernels are not registered")
    spec = registry.get_by_name("flashinfer_trtllm_unquant_moe_apply")
    assert spec.traits["ispp_alignment"] == frozenset(
        {unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT}
    )
    # Registration adopts the launcher's alignment only on SM100-SM103.
    expected = adapter.STOCK_ISPP_ALIGNMENT
    if ArchVersion(10, 0) <= Platform.get().arch_version <= ArchVersion(10, 3):
        expected = adapter.gated_ispp_alignment()
    assert unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT == expected

    real_platform = Platform.get()
    try:
        Platform.override(b200_platform)
        registry.clear_cache()
        plan = kernel_moe_plan(
            "unquant",
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode=routing_mode,
            ep_size=1,
            ispp=ispp,
            hidden=2048,
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            internal_activation_dtype="input",
            fast_math=True,
            combine_order="rank",
        )
    finally:
        Platform.override(real_platform)
        registry.clear_cache()
    admitted = ispp % unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT == 0
    assert (plan["solution"] == "flashinfer_trtllm") == admitted


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


# Precomputed top-k, and in-kernel routing with RenormalizeNaive (4, the Qwen3
# MoE blocks), Renormalize (1, the default routing_method_type) and DeepSeekV3
# (2, DeepseekV3ForCausalLM).
@pytest.mark.parametrize(
    "routing_mode, routing_method_type",
    [
        ("precomputed_topk", None),
        ("kernel_routing", 4),
        ("kernel_routing", 1),
        ("kernel_routing", 2),
    ],
    ids=["precomputed_topk", "renormalize_naive", "renormalize", "deepseek_v3"],
)
@pytest.mark.parametrize("intermediate_size", [64, 160, 192, 320])
def test_64_aligned_outputs_match_128_padded(
    monkeypatch, intermediate_size, routing_mode, routing_method_type
):
    """Serving at a multiple of 64 matches the previous 128 padding bit for bit."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("TRT-LLM BF16 MoE kernels need SM100 or SM103")
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_unquant as unquant

    if unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT != adapter.GATED_ISPP_ALIGNMENT:
        pytest.skip("the installed FlashInfer launcher cannot be relaxed or built")

    # Record which launcher each run reaches.
    launchers = []

    def recorded(kind, launcher):
        def call(*args, **kwargs):
            launchers.append(kind)
            return launcher(*args, **kwargs)

        return call

    for module, kind in ((unquant, "stock"), (adapter, "private")):
        for name in ("trtllm_bf16_moe", "trtllm_bf16_routed_moe"):
            monkeypatch.setattr(module, name, recorded(kind, getattr(module, name)))

    num_experts, top_k, hidden, num_tokens = 16, 4, 1024, 37
    generator = torch.Generator(device="cuda").manual_seed(intermediate_size)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(
            torch.bfloat16
        )

    inter = intermediate_size
    gate = randn(num_experts, inter, hidden, scale=hidden**-0.5)
    up = randn(num_experts, inter, hidden, scale=hidden**-0.5)
    down = randn(num_experts, hidden, inter, scale=inter**-0.5)
    x = randn(num_tokens, hidden, scale=2.0)
    router_logits = randn(num_tokens, num_experts)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(router_logits.float(), dim=-1), top_k, dim=-1
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    routing_config = {"routing_method_type": routing_method_type}
    if routing_method_type == 2:
        # One expert group and an fp32 correction bias; the kernel wrapper
        # casts the logits to fp32 for this routing method.
        routing_config.update(
            n_group=1,
            topk_group=1,
            routed_scaling_factor=2.5,
            correction_bias=torch.randn(
                num_experts, device="cuda", generator=generator
            ),
        )

    def run(ispp):
        plan = kernel_moe_plan(
            "unquant",
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode=routing_mode,
            ep_size=1,
            ispp=ispp,
            hidden=hidden,
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            internal_activation_dtype="input",
            solution="flashinfer_trtllm",
            fast_math=True,
            combine_order="rank",
        )
        # Zero-pad each half of w13 and the columns of w2 like the loader.
        pad = torch.zeros(num_experts, ispp - inter, hidden, device="cuda")
        w = torch.nn.Module()
        w.w13_weight = torch.nn.Parameter(
            torch.cat([gate, pad.to(gate), up, pad.to(up)], dim=1),
            requires_grad=False,
        )
        w.w2_weight = torch.nn.Parameter(
            torch.cat([down, pad.to(down).transpose(1, 2)], dim=2).contiguous(),
            requires_grad=False,
        )
        w.num_experts = num_experts
        w.num_local_experts = num_experts
        w.top_k = top_k
        w.intermediate_size = ispp
        w.tp_size = 1
        w.ep_rank = 0
        w.routing_config = routing_config
        kernel_moe_process_weights(plan, w)
        return kernel_moe_apply(
            plan,
            x,
            w,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
        )

    served = run(_round_up(inter, adapter.GATED_ISPP_ALIGNMENT))
    assert set(launchers) == {"private"}
    launchers.clear()
    padded = run(_round_up(inter, adapter.STOCK_ISPP_ALIGNMENT))
    assert set(launchers) == {"stock"}
    assert torch.equal(served, padded)


# A DeepSeekV3 routingMainKernel as FlashInfer 0.7.0 ships it, abridged.
_ROUTING = """
template <typename KernelParams>
__global__ void routingMainKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  static constexpr float invalidScoreFloat = float{-INFINITY};
  const OutputT invalidScore = OutputT{invalidScoreFloat};

  auto biasVal = (expertSelected && params.mPtrRoutingBias != nullptr)
                     ? static_cast<OutputT>(
                           loadScalar(params.mPtrRoutingBias, threadExpert, params.mDtypeBias))
                     : (expertSelected ? OutputT{0} : invalidScore);

  if (params.mPtrScores != nullptr) {
    float score =
        expertSelected ? static_cast<float>(params.mPtrScores[scoreIdx]) : invalidScoreFloat;
    auto scoreSigmoid = sigmoid_accurate(score);
    auto scoreBias = float{scoreSigmoid + float{biasVal}};
  }
}

template <typename KernelParams>
__global__ void routingIndicesClusterKernel(KernelParams params) {
  auto scoreSigmoid = sigmoid_accurate(score);
}
"""
_STOCK_BIAS = """static_cast<OutputT>(
                           loadScalar(params.mPtrRoutingBias, threadExpert, params.mDtypeBias))
                     : (expertSelected ? OutputT{0} : invalidScore);"""
_FP32_BIAS = """loadScalar(params.mPtrRoutingBias, threadExpert, params.mDtypeBias)
                     : (expertSelected ? 0.0F : invalidScoreFloat);"""
_INVALID_SCORE = "  const OutputT invalidScore = OutputT{invalidScoreFloat};\n"
_STOCK_SIGMOID = "    auto scoreSigmoid = sigmoid_accurate(score);\n    auto scoreBias"
_EXP_SIGMOID = (
    "    // The tanh form loses small positive probabilities for negative logits.\n"
    "    auto scoreSigmoid = 1.0f / (1.0f + expf(-score));\n    auto scoreBias"
)
# flashinfer-ai/flashinfer#5557's routingMainKernel.
_FP32_ROUTING = (
    _ROUTING.replace(_INVALID_SCORE, "")
    .replace(_STOCK_BIAS, _FP32_BIAS)
    .replace(_STOCK_SIGMOID, _EXP_SIGMOID)
)


def test_routing_edit_is_flashinfer_5557():
    assert adapter._keep_routing_bias_fp32(_ROUTING) == _FP32_ROUTING
    # Already edited (a FlashInfer with #5557, or this adapter's output).
    assert adapter._keep_routing_bias_fp32(_FP32_ROUTING) == _FP32_ROUTING


@pytest.mark.parametrize(
    "source",
    [
        _ROUTING.replace(_INVALID_SCORE, "").replace(_STOCK_BIAS, _FP32_BIAS),
        _ROUTING.replace(_STOCK_SIGMOID, _EXP_SIGMOID),
    ],
    ids=["bias-edited", "sigmoid-edited"],
)
def test_routing_edit_completes_a_partial_edit(source):
    assert adapter._keep_routing_bias_fp32(source) == _FP32_ROUTING


_KERNEL_END = "    auto scoreBias = float{scoreSigmoid + float{biasVal}};\n"


@pytest.mark.parametrize(
    "source",
    [
        "",
        _ROUTING + _ROUTING,
        # The FP32 form, but rounded through a BF16 declaration.
        _FP32_ROUTING.replace("auto biasVal", "OutputT biasVal"),
        _ROUTING.replace(
            _KERNEL_END,
            _KERNEL_END
            + "    float again = loadScalar(params.mPtrRoutingBias, 0, 0);\n",
        ),
        _ROUTING.replace(_KERNEL_END, _KERNEL_END + "    auto low = invalidScore;\n"),
        _ROUTING.replace(_INVALID_SCORE, ""),
        _ROUTING.replace(_STOCK_SIGMOID, "    auto scoreBias"),
        _ROUTING.replace(
            _KERNEL_END,
            "    auto scoreSigmoid = sigmoid_accurate(score);\n" + _KERNEL_END,
        ),
        _ROUTING.replace(
            _KERNEL_END,
            "    auto scoreSigmoid = 1.0f / (1.0f + expf(-score));\n" + _KERNEL_END,
        ),
    ],
    ids=[
        "empty",
        "kernel-twice",
        "bias-declared-bf16",
        "bias-loaded-twice",
        "invalid-score-reused",
        "no-invalid-score",
        "no-sigmoid",
        "sigmoid-twice",
        "stock-and-edited-sigmoid",
    ],
)
def test_unrecognized_routing_fails_closed(source):
    with pytest.raises(RuntimeError, match="review the BF16 MoE adapter"):
        adapter._keep_routing_bias_fp32(source)


def _installed_routing() -> str:
    jit_env = pytest.importorskip("flashinfer.jit.env")
    path = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "trtllm_backend"
    path = path / "trtllm_fused_moe_routing_deepseek.cu"
    if not path.exists():
        pytest.skip("installed FlashInfer ships no DeepSeekV3 routing source")
    return path.read_text()


def test_installed_routing_gets_exactly_the_fp32_bias_edit():
    import difflib

    stock = _installed_routing()
    edited = adapter._keep_routing_bias_fp32(stock)
    assert adapter._keep_routing_bias_fp32(edited) == edited
    changed = [
        line[0]
        for line in difflib.unified_diff(stock.splitlines(), edited.splitlines())
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]
    # flashinfer-ai/flashinfer#5557's +4/-5, or nothing once FlashInfer has it.
    assert sorted(changed) in (["+"] * 4 + ["-"] * 5, [])
    assert adapter.stock_routing_keeps_fp32_bias() == (not changed)


def test_stock_routing_detection_reads_the_installed_source(tmp_path, monkeypatch):
    jit_env = pytest.importorskip("flashinfer.jit.env")
    routing = tmp_path / "fused_moe" / "trtllm_backend"
    routing.mkdir(parents=True)
    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", tmp_path)
    try:
        for source, keeps in ((_ROUTING, False), (_FP32_ROUTING, True)):
            (routing / "trtllm_fused_moe_routing_deepseek.cu").write_text(source)
            adapter.stock_routing_keeps_fp32_bias.cache_clear()
            assert adapter.stock_routing_keeps_fp32_bias() is keeps
    finally:
        adapter.stock_routing_keeps_fp32_bias.cache_clear()


@pytest.mark.parametrize("gated_alignment", [64, 128])
def test_fp32_routing_bias_module_is_named_after_its_sources(
    tmp_path, monkeypatch, gated_alignment
):
    import hashlib
    from dataclasses import dataclass

    fused_moe = pytest.importorskip("flashinfer.jit.fused_moe")
    jit_env = pytest.importorskip("flashinfer.jit.env")
    stock = tmp_path / "csrc"
    stock.mkdir()
    names = (
        "trtllm_fused_moe_kernel_launcher.cu",
        "trtllm_fused_moe_runner.cu",
        "trtllm_fused_moe_routing_deepseek.cu",
    )
    for name, text in zip(names, (_LAUNCHER, "// runner\n", _ROUTING)):
        (stock / name).write_text(text)

    @dataclass(frozen=True)
    class Spec:
        name: str
        sources: list

    monkeypatch.setattr(
        fused_moe,
        "gen_trtllm_gen_fused_moe_sm100_module",
        lambda **kwargs: Spec("fused_moe_trtllm_sm100", [stock / n for n in names]),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "gen")
    monkeypatch.setattr(adapter, "gated_ispp_alignment", lambda: gated_alignment)

    relaxed = adapter._relax_bf16_intermediate_check(_LAUNCHER)
    launcher = adapter._relaxed_spec(enable_rubin=False)
    # The 64-aligned launcher keeps its single-source name.
    digest = hashlib.sha256(relaxed.encode()).hexdigest()[:16]
    assert launcher.name == f"tokenspeed_fused_moe_trtllm_sm100_bf16_ispp64_{digest}"

    spec = adapter._fp32_routing_bias_spec(enable_rubin=False)
    edited = relaxed if gated_alignment == 64 else _LAUNCHER
    digest = hashlib.sha256(
        ((relaxed if gated_alignment == 64 else "") + _FP32_ROUTING).encode()
    ).hexdigest()[:16]
    name = f"tokenspeed_fused_moe_trtllm_sm100_fp32_routing_bias_{digest}"
    directory = tmp_path / "gen" / name
    assert spec.name == name
    assert [path.read_text() for path in spec.sources] == [
        edited,
        "// runner\n",
        _FP32_ROUTING,
    ]
    assert spec.sources[1] == stock / names[1]
    assert spec.sources[2] == directory / names[2]
    (directory / names[2]).write_text(_ROUTING)
    with pytest.raises(RuntimeError, match="source-cache mismatch"):
        adapter._fp32_routing_bias_spec(enable_rubin=False)


def test_fp32_routing_bias_entrypoints_are_private():
    core = pytest.importorskip("flashinfer.fused_moe.core")
    before = dict(vars(core))
    args = (adapter._fp32_routing_bias_spec, adapter._FP32_BIAS_OPERATORS)
    private = adapter._entrypoints(*args)
    assert vars(core) == before
    assert adapter._entrypoints(*args) is private
    assert private is not adapter._entrypoints()
    assert private["trtllm_bf16_moe"].__globals__ is private
    assert private["gen_trtllm_gen_fused_moe_sm100_module"] is args[0]
    assert private["register_custom_op"].keywords == {"prefix": args[1]}
    factory = private["_get_trtllm_moe_sm100_module_impl"]
    assert factory is not adapter._entrypoints()["_get_trtllm_moe_sm100_module_impl"]


@pytest.mark.parametrize("case", ["already-fp32", "built", "refused", "build-fails"])
def test_fp32_routing_bias_ready(monkeypatch, caplog, case):
    builds = []

    def build():
        builds.append(1)
        if case == "build-fails":
            raise RuntimeError("nvcc failed")

    def stock_keeps_fp32_bias():
        if case == "refused":
            raise RuntimeError("unrecognized routing")
        return case == "already-fp32"

    monkeypatch.setattr(
        adapter, "_entrypoints", lambda *a: {"get_trtllm_moe_sm100_module": build}
    )
    monkeypatch.setattr(adapter, "stock_routing_keeps_fp32_bias", stock_keeps_fp32_bias)
    adapter.fp32_routing_bias_ready.cache_clear()
    try:
        with caplog.at_level("INFO", logger=adapter.__name__):
            ready = adapter.fp32_routing_bias_ready()
            # Once per process, whatever the outcome.
            assert adapter.fp32_routing_bias_ready() is ready
    finally:
        adapter.fp32_routing_bias_ready.cache_clear()
    assert ready is (case in ("already-fp32", "built"))
    assert len(builds) == (case in ("built", "build-fails"))
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == (not ready)


def _plan(platform, weight_dtype, routing_mode=None, **kwargs):
    from tokenspeed_kernel.platform import Platform
    from tokenspeed_kernel.registry import KernelRegistry

    real_platform = Platform.get()
    try:
        Platform.override(platform)
        KernelRegistry.get().clear_cache()
        return kernel_moe_plan(
            weight_dtype,
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode=routing_mode,
            ep_size=1,
            ispp=256,
            hidden=7168,
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            fp8_scale_block_shape=(128, 128) if weight_dtype == "fp8" else None,
            internal_activation_dtype="input",
            fast_math=True,
            combine_order="rank",
            **kwargs,
        )
    finally:
        Platform.override(real_platform)
        KernelRegistry.get().clear_cache()


@pytest.mark.parametrize(
    "weight_dtype, routing_mode, hook, expected",
    [
        ("unquant", None, True, "flashinfer_trtllm_unquant_moe_apply"),
        ("unquant", "kernel_routing", True, "flashinfer_trtllm_unquant_moe_apply"),
        ("unquant", None, False, "flashinfer_trtllm_unquant_routed_moe_apply"),
        ("unquant", "kernel_routing", False, ValueError),
        (
            "unquant",
            "precomputed_topk",
            None,
            "flashinfer_trtllm_unquant_routed_moe_apply",
        ),
        # In-kernel routers without the hook plan precomputed top-k.
        ("fp8", None, None, "flashinfer_cutlass_fp8_moe_apply"),
        ("nvfp4", None, None, "flashinfer_trtllm_nvfp4_routed_moe_apply"),
        ("mxfp4", None, None, "flashinfer_trtllm_mxfp4_moe_apply"),
    ],
)
def test_moe_plan_keeps_in_kernel_routing_only_with_an_fp32_bias(
    b200_platform, monkeypatch, weight_dtype, routing_mode, hook, expected
):
    from tokenspeed_kernel.registry import KernelRegistry

    pytest.importorskip("tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant")
    if KernelRegistry.get().get_by_name("flashinfer_trtllm_unquant_moe_apply") is None:
        pytest.skip("flashinfer_trtllm MoE kernels are not registered")
    calls = []
    # The registered function moe_plan calls; a module reload rebinds only the
    # module's name.
    kernel = KernelRegistry.get().get_impl("flashinfer_trtllm_unquant_moe_apply")
    monkeypatch.setattr(
        kernel, "_tokenspeed_fp32_correction_bias", lambda: calls.append(1) or hook
    )
    default = _plan(b200_platform, weight_dtype, routing_mode)
    off = _plan(b200_platform, weight_dtype, routing_mode, fp32_correction_bias=False)
    assert off == {**default, "fp32_correction_bias": False} and not calls
    if expected is ValueError:
        with pytest.raises(ValueError, match="cannot add the correction bias"):
            _plan(b200_platform, weight_dtype, routing_mode, fp32_correction_bias=True)
        return
    plan = _plan(b200_platform, weight_dtype, routing_mode, fp32_correction_bias=True)
    assert plan["apply_kernel_name"] == expected
    assert plan["fp32_correction_bias"] is True
    kept = expected == "flashinfer_trtllm_unquant_moe_apply"
    assert plan["support_routing"] is kept
    assert plan["supports_precomputed_topk"] is not kept
    assert len(calls) == (hook is not None)


def _deepseek_v3_routing(logits, bias, top_k, n_group, topk_group, scale):
    """DeepSeek-V3's reference router: FP32 sigmoid scores plus an FP32 bias."""
    scores = torch.sigmoid(logits.float())
    biased = scores + bias.float()
    tokens, experts = logits.shape
    groups = biased.view(tokens, n_group, -1).topk(2, dim=-1).values.sum(dim=-1)
    kept = torch.zeros_like(groups, dtype=torch.bool)
    kept.scatter_(1, groups.topk(topk_group, dim=-1).indices, True)
    kept = kept.repeat_interleave(experts // n_group, dim=1)
    ids = biased.masked_fill(~kept, float("-inf")).topk(top_k, dim=-1).indices
    weights = scores.gather(1, ids)
    return ids, weights / weights.sum(dim=-1, keepdim=True) * scale


def _gpu_bf16_moe_layer(intermediate_size, num_experts=256, hidden=1024):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("TRT-LLM BF16 MoE kernels need SM100 or SM103")
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_unquant as unquant

    if intermediate_size % unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT:
        pytest.skip("the installed FlashInfer launcher cannot be relaxed")
    generator = torch.Generator(device="cuda").manual_seed(intermediate_size)

    def randn(*shape, scale):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(
            torch.bfloat16
        )

    w = torch.nn.Module()
    w.w13_weight = torch.nn.Parameter(
        randn(num_experts, 2 * intermediate_size, hidden, scale=hidden**-0.5),
        requires_grad=False,
    )
    w.w2_weight = torch.nn.Parameter(
        randn(num_experts, hidden, intermediate_size, scale=intermediate_size**-0.5),
        requires_grad=False,
    )
    w.num_experts = w.num_local_experts = num_experts
    w.top_k, w.intermediate_size, w.tp_size, w.ep_rank = 8, intermediate_size, 1, 0
    return w, randn(1, hidden, scale=1.0)


# flashinfer-ai/flashinfer#5557's cases: bias steps below the BF16 spacing of
# the bias, and logits whose tanh-form sigmoid is 0.
_BIAS_CASES = ["fp32-bias-steps", "bf16-bias-steps", "negative-logits"]


def _routing_case(case, num_tokens):
    expert = torch.arange(256, device="cuda", dtype=torch.float32)
    if case == "negative-logits":
        return torch.full((num_tokens, 256), -20.0, device="cuda"), -expert / 8
    logits = (-5 + expert * 0.00004).repeat(num_tokens, 1)
    if case == "fp32-bias-steps":
        return logits, 20.14 - expert * 0.00002
    return logits, (20.0 - expert * 0.125).to(torch.bfloat16)


@pytest.mark.parametrize("intermediate_size", [64, 128])
@pytest.mark.parametrize("num_tokens", [1, 128, 1025])
@pytest.mark.parametrize("case", _BIAS_CASES)
def test_fp32_bias_routing_matches_the_fp32_reference(
    case, num_tokens, intermediate_size
):
    w, x = _gpu_bf16_moe_layer(intermediate_size)
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_unquant as unquant

    assert adapter.fp32_routing_bias_ready()
    unquant.flashinfer_trtllm_unquant_moe_weights({}, w)
    logits, bias = _routing_case(case, num_tokens)
    replay = torch.full((num_tokens, 8), -1, dtype=torch.int16, device="cuda")
    # The module trtllm_unquant.py routes an FP32-bias plan on.
    moe = adapter.trtllm_bf16_fp32_routing_bias_moe
    if adapter.stock_routing_keeps_fp32_bias():
        moe = unquant.trtllm_bf16_moe
        if intermediate_size % adapter.STOCK_ISPP_ALIGNMENT:
            moe = adapter.trtllm_bf16_moe
    _, weights, _ = moe(
        routing_logits=logits,
        routing_bias=bias,
        hidden_states=x.expand(num_tokens, -1).contiguous(),
        gemm1_weights=w.w13_weight,
        gemm2_weights=w.w2_weight,
        num_experts=256,
        top_k=8,
        n_group=8,
        topk_group=4,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=256,
        routed_scaling_factor=2.5,
        routing_method_type=2,
        do_finalize=False,
        routing_replay_out=replay,
    )
    if weights.dtype == torch.float32:
        weights = weights.view(torch.bfloat16).view(-1, 8)[:num_tokens]
    ids, expected = _deepseek_v3_routing(logits, bias, 8, 8, 4, 2.5)
    order = replay.long().argsort(dim=1)
    assert torch.equal(replay.long().gather(1, order), ids.sort(dim=1).values)
    got = weights.float().gather(1, order)
    want = expected.gather(1, ids.argsort(dim=1))
    torch.testing.assert_close(got, want, rtol=2**-8, atol=0)
    if case == "negative-logits":
        assert torch.all(got == 0.3125)


@pytest.mark.parametrize("intermediate_size", [64, 128])
def test_fp32_bias_plan_routes_like_precomputed_fp32_topk(
    monkeypatch, intermediate_size
):

    w, x = _gpu_bf16_moe_layer(intermediate_size)
    reached = []
    fp32_moe = adapter.trtllm_bf16_fp32_routing_bias_moe
    monkeypatch.setattr(
        adapter,
        "trtllm_bf16_fp32_routing_bias_moe",
        lambda *a, **k: reached.append(1) or fp32_moe(*a, **k),
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits, bias = _routing_case("fp32-bias-steps", 37)
    logits = logits + 0.01 * torch.randn(
        logits.shape, device="cuda", generator=generator
    )
    w.routing_config = dict(
        routing_method_type=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        correction_bias=bias,
    )

    def plan(**kwargs):
        return kernel_moe_plan(
            "unquant",
            input_dtype=torch.bfloat16,
            activation="silu",
            ep_size=1,
            ispp=intermediate_size,
            hidden=x.shape[1],
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            internal_activation_dtype="input",
            solution="flashinfer_trtllm",
            fast_math=True,
            combine_order="rank",
            **kwargs,
        )

    in_kernel = plan(fp32_correction_bias=True)
    routed = plan(routing_mode="precomputed_topk")
    assert in_kernel["support_routing"] and not routed["support_routing"]
    kernel_moe_process_weights(in_kernel, w)
    hidden = torch.randn((37, x.shape[1]), device="cuda", generator=generator)
    hidden = hidden.to(torch.bfloat16)
    ids, weights = _deepseek_v3_routing(logits, bias, 8, 8, 4, 2.5)
    got = kernel_moe_apply(in_kernel, hidden, w, logits)
    assert reached or adapter.stock_routing_keeps_fp32_bias()
    want = kernel_moe_apply(
        routed, hidden, w, logits, topk_weights=weights, topk_ids=ids.to(torch.int32)
    )
    torch.testing.assert_close(got, want, rtol=2**-6, atol=2**-10)
