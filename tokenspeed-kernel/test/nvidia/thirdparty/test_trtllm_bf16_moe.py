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
    import tokenspeed_kernel
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
        return tokenspeed_kernel.moe_plan(
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
    import tokenspeed_kernel
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
