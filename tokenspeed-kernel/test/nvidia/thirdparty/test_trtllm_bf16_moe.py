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
import inspect

import pytest
import torch
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


def test_gated_alignment_falls_back_to_the_stock_launcher(monkeypatch):
    _installed_launcher()

    def refuse(source):
        raise RuntimeError("unrecognized launcher")

    adapter.gated_ispp_alignment.cache_clear()
    try:
        assert adapter.gated_ispp_alignment() == adapter.GATED_ISPP_ALIGNMENT
        adapter.gated_ispp_alignment.cache_clear()
        monkeypatch.setattr(adapter, "_relax_bf16_intermediate_check", refuse)
        assert adapter.gated_ispp_alignment() == adapter.STOCK_ISPP_ALIGNMENT
    finally:
        adapter.gated_ispp_alignment.cache_clear()


@pytest.mark.parametrize("drift", ["wrapped-entry-point", "no-csrc-dir"])
def test_gated_alignment_falls_back_on_unrecognized_flashinfer(monkeypatch, drift):
    _installed_launcher()
    core = pytest.importorskip("flashinfer.fused_moe.core")
    jit_env = pytest.importorskip("flashinfer.jit.env")
    if drift == "wrapped-entry-point":
        routed = functools.partial(core.trtllm_bf16_routed_moe)
        monkeypatch.setattr(core, "trtllm_bf16_routed_moe", routed)
    else:
        monkeypatch.delattr(jit_env, "FLASHINFER_CSRC_DIR")
    adapter._entrypoints.cache_clear()
    adapter.gated_ispp_alignment.cache_clear()
    try:
        assert adapter.gated_ispp_alignment() == adapter.STOCK_ISPP_ALIGNMENT
    finally:
        monkeypatch.undo()
        adapter._entrypoints.cache_clear()
        adapter.gated_ispp_alignment.cache_clear()


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
    import tokenspeed_kernel
    from tokenspeed_kernel.platform import Platform
    from tokenspeed_kernel.registry import KernelRegistry

    unquant = pytest.importorskip("tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant")
    registry = KernelRegistry.get()
    if registry.get_by_name("flashinfer_trtllm_unquant_moe_apply") is None:
        pytest.skip("flashinfer_trtllm unquant MoE kernels are not registered")
    assert unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT == adapter.gated_ispp_alignment()

    real_platform = Platform.get()
    try:
        Platform.override(b200_platform)
        registry.clear_cache()
        plan = tokenspeed_kernel.moe_plan(
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
    import tokenspeed_kernel
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_unquant as unquant

    if unquant.TRTLLM_UNQUANT_ISPP_ALIGNMENT != adapter.GATED_ISPP_ALIGNMENT:
        pytest.skip("the installed FlashInfer launcher cannot be relaxed")

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
        plan = tokenspeed_kernel.moe_plan(
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
        tokenspeed_kernel.moe_process_weights(plan, w)
        return tokenspeed_kernel.moe_apply(
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
