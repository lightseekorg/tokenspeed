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

"""Compatibility and isolation checks for the private MoE tactic policy."""

import functools
import inspect

import pytest
import torch
from tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe import (
    _clone,
    _entrypoints,
    _prefer_qwen38_decode_tile_32,
    _register_private,
    _require_runner_rebinding,
    _require_tactic_hooks,
)

_CLONE_VALUE = object()


def test_function_rebinding_does_not_mutate_upstream():
    sentinel = object()

    def original(x, *, value):
        return x, value, _CLONE_VALUE

    clone = _clone(original, {**original.__globals__, "_CLONE_VALUE": sentinel})
    assert clone(3, value=4) == (3, 4, sentinel)
    assert inspect.signature(clone) == inspect.signature(original)
    assert original(3, value=4) == (3, 4, _CLONE_VALUE)
    assert original.__globals__["_CLONE_VALUE"] is not sentinel


def test_operator_names_are_private():
    def register(name, *, mutates_args):
        return name, mutates_args

    assert _register_private(register, "flashinfer::moe", mutates_args=("out",)) == (
        "tokenspeed_flashinfer_route_init::moe",
        ("out",),
    )
    with pytest.raises(RuntimeError, match="Unexpected FlashInfer operator"):
        _register_private(register, "another::moe", mutates_args=())


def test_qwen38_decode_tactic_filter_accepts_ffi_arrays():
    tvm_ffi = pytest.importorskip("tvm_ffi")
    tactics = [
        tvm_ffi.Array([8, 1]),
        tvm_ffi.Array([32, 2]),
        tvm_ffi.Array([16, 3]),
        tvm_ffi.Array([32, 4]),
    ]
    assert _prefer_qwen38_decode_tile_32(tactics) == [tactics[1], tactics[3]]
    without_tile_32 = [tactics[0], tactics[2]]
    assert _prefer_qwen38_decode_tile_32(without_tile_32) is without_tile_32


def _moe_runner_and_inputs(num_tokens, hidden_size, weight_dtype):
    core = pytest.importorskip("flashinfer.fused_moe.core")
    inputs_module = pytest.importorskip("flashinfer.fused_moe.shared.inputs")
    enums = pytest.importorskip("flashinfer.tllm_enums")
    tvm_ffi = pytest.importorskip("tvm_ffi")
    tactics = [tvm_ffi.Array([8, 1]), tvm_ffi.Array([32, 2])]

    class MoeOp:
        def trtllm_get_valid_moe_configs(self, *query_key):
            return tactics

    runner = _entrypoints()["TrtllmMoERunner"](
        MoeOp(),
        top_k=10,
        num_local_experts=128,
        dtype_act=enums.DtypeTrtllmGen.Bfloat16,
        dtype_weights=enums.DtypeTrtllmGen[weight_dtype],
        fp8_quantization_type=enums.Fp8QuantizationType.NoneFp8,
        hidden_size=hidden_size,
        intermediate_size=640,
        num_experts=512,
    )
    inputs = [None] * len(inputs_module.MoeRunnerInputs._FIELDS)
    hidden_index = inputs_module.MoeRunnerInputs._FIELDS.index("hidden_states")
    inputs[hidden_index] = torch.empty((num_tokens, hidden_size))
    return core, runner, inputs, tactics, hidden_index


def _input_shapes(inputs):
    return tuple(tuple(tensor.shape) if tensor is not None else () for tensor in inputs)


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "weight_dtype", "targeted"),
    [
        (4, 2560, "E2m1", True),
        (32, 2560, "E2m1", True),
        (33, 2560, "E2m1", False),
        (4, 4096, "E2m1", False),
        (4, 2560, "Bfloat16", False),
    ],
)
def test_tactic_cache_key_changes_only_for_target_shape(
    num_tokens, hidden_size, weight_dtype, targeted
):
    core, runner, inputs, tactics, _ = _moe_runner_and_inputs(
        num_tokens, hidden_size, weight_dtype
    )
    autotuner = pytest.importorskip("flashinfer.autotuner")
    upstream_extras = core.TrtllmMoERunner.get_cache_key_extras(runner, inputs)
    # Tensor properties stay invariant; the target profile adds the policy tag.
    assert runner.get_cache_key_extras(inputs) == upstream_extras
    key = core.AutoTuner._get_cache_key(
        "test_moe",
        runner,
        _input_shapes(inputs),
        autotuner.TuningConfig(),
        upstream_extras,
    )
    if targeted:
        assert key.extras == (*upstream_extras, "tokenspeed-qwen38-tile32-v1")
        assert runner.get_valid_tactics(inputs, None) == [tactics[1]]
    else:
        assert key.extras == upstream_extras
        assert runner.get_valid_tactics(inputs, None) is tactics


@pytest.mark.parametrize(
    "caller_tokens,profile_tokens", [(4, 64), (8192, 1), (8192, 32)]
)
def test_profile_cache_lookup_matches_stored_winner(caller_tokens, profile_tokens):
    core, runner, caller_inputs, tactics, hidden_index = _moe_runner_and_inputs(
        caller_tokens, 2560, "E2m1"
    )
    autotuner = pytest.importorskip("flashinfer.autotuner")
    profile_inputs = list(caller_inputs)
    profile_inputs[hidden_index] = torch.empty((profile_tokens, 2560))
    profile_shapes = _input_shapes(profile_inputs)
    config = autotuner.TuningConfig()
    tuner = core.AutoTuner()
    stored = core.AutoTuner._get_cache_key(
        "test_moe",
        runner,
        profile_shapes,
        config,
        runner.get_cache_key_extras(profile_inputs),
    )
    winner = tactics[1] if profile_tokens <= 32 else tactics[0]
    tuner.profiling_cache[stored] = (winner, None)
    # This is choose_one's lookup: target profile shapes plus caller tensors.
    hit, _, actual, _ = tuner.search_cache(
        "test_moe", [runner], profile_shapes, config, inputs=caller_inputs
    )
    assert hit
    assert actual is winner
    assert ("tokenspeed-qwen38-tile32-v1" in stored.extras) == (profile_tokens <= 32)
    other = core.AutoTuner._get_cache_key(
        "test_moe",
        runner,
        _input_shapes(caller_inputs),
        config,
        runner.get_cache_key_extras(caller_inputs),
    )
    assert stored != other  # Different token profiles still select independently.


def test_serving_lookup_uses_mapped_profile_for_policy_tag():
    core, runner, inputs, _, hidden_index = _moe_runner_and_inputs(33, 2560, "E2m1")
    autotuner = pytest.importorskip("flashinfer.autotuner")
    config = autotuner.TuningConfig(
        dynamic_tensor_specs=(
            autotuner.DynamicTensorSpec(
                input_idx=(hidden_index,),
                dim_idx=(0,),
                gen_tuning_buckets=(32, 64),
                map_to_tuning_buckets=lambda tokens: 32,
            ),
        )
    )
    key = core.AutoTuner._get_cache_key(
        "test_moe",
        runner,
        _input_shapes(inputs),
        config,
        runner.get_cache_key_extras(inputs),
    )
    assert key.nearest_profile[hidden_index][0] == 32
    assert key.extras[-1] == "tokenspeed-qwen38-tile32-v1"


def test_cache_key_hook_preserves_stock_runner_keys():
    core = pytest.importorskip("flashinfer.fused_moe.core")
    autotuner = pytest.importorskip("flashinfer.autotuner")
    stock_runner = object()
    shapes = ((4, 2560),)
    config = autotuner.TuningConfig()
    expected = core.AutoTuner._get_cache_key(
        "test_stock", stock_runner, shapes, config, ("explicit-extra",)
    )
    _entrypoints()
    actual = core.AutoTuner._get_cache_key(
        "test_stock", stock_runner, shapes, config, ("explicit-extra",)
    )
    assert actual == expected
    assert actual.extras == ("explicit-extra",)


def test_tactic_hooks_must_exist_on_upstream_runner():
    class CacheOnly:
        def get_cache_key_extras(self, inputs):
            return ()

    class TacticsOnly:
        def get_valid_tactics(self, inputs, profile):
            return []

    class InheritedHooks(CacheOnly, TacticsOnly):
        pass

    _require_tactic_hooks(InheritedHooks)
    with pytest.raises(RuntimeError, match="get_cache_key_extras"):
        _require_tactic_hooks(TacticsOnly)
    with pytest.raises(RuntimeError, match="get_valid_tactics"):
        _require_tactic_hooks(CacheOnly)


def test_runner_rebinding_requires_cloned_global_reference():
    def uses_runner():
        return TrtllmMoERunner

    def omits_runner():
        return None

    namespace = {"trtllm_fp4_block_scale_moe": uses_runner}
    with pytest.raises(RuntimeError, match="through cloned globals"):
        _require_runner_rebinding(namespace)

    namespace["trtllm_fp4_block_scale_moe"] = _clone(uses_runner, namespace)
    _require_runner_rebinding(namespace)

    namespace["trtllm_fp4_block_scale_moe"] = _clone(omits_runner, namespace)
    with pytest.raises(RuntimeError, match="no longer construct"):
        _require_runner_rebinding(namespace)


def test_upstream_dispatch_and_caches_are_unchanged():
    core = pytest.importorskip("flashinfer.fused_moe.core")
    before = dict(vars(core))
    private = _entrypoints()
    assert vars(core) == before
    assert (
        private["get_trtllm_moe_sm100_module"] is not core.get_trtllm_moe_sm100_module
    )
    assert issubclass(private["TrtllmMoERunner"], core.TrtllmMoERunner)
    assert private["TrtllmMoERunner"] is not core.TrtllmMoERunner
    for name in ("trtllm_fp4_block_scale_moe", "trtllm_fp4_block_scale_routed_moe"):
        assert private[name].__globals__ is private
        assert inspect.signature(private[name]) == inspect.signature(
            getattr(core, name)
        )
    factory = private.get("_get_trtllm_moe_sm100_module_impl")
    if factory is not None:
        assert isinstance(factory, functools._lru_cache_wrapper)
        assert factory is not core._get_trtllm_moe_sm100_module_impl


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_private_fp4_dispatch_matches_upstream_output():
    """Factory/builder splits must retain working operators and their arguments."""
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe import (
        trtllm_fp4_block_scale_routed_moe,
    )

    if not current_platform().is_nvidia or current_platform().arch_version.major != 10:
        pytest.skip("requires Blackwell or Rubin")
    core = pytest.importorskip("flashinfer.fused_moe.core")
    from flashinfer import fp4_quantize

    torch.manual_seed(6604)
    batch, experts, hidden, intermediate, top_k = 8, 16, 256, 256, 2

    def codes(*shape):
        return torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)

    def scales(*shape):
        return torch.full(shape, 0.015625, device="cuda").to(torch.float8_e4m3fn)

    ids = torch.arange(batch * top_k, device="cuda", dtype=torch.int32)
    ids = ids.reshape(batch, top_k) % experts
    hidden_states, hidden_states_scale = fp4_quantize(
        torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16) * 0.1,
        torch.ones(1, device="cuda"),
        is_sf_swizzled_layout=False,
    )
    kwargs = dict(
        topk_ids=(
            ids,
            torch.full((batch, top_k), 0.5, device="cuda", dtype=torch.bfloat16),
        ),
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale.view(torch.float8_e4m3fn),
        gemm1_weights=codes(experts, 2 * intermediate, hidden // 2),
        gemm1_weights_scale=scales(experts, 2 * intermediate, hidden // 16),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=codes(experts, hidden, intermediate // 2),
        gemm2_weights_scale=scales(experts, hidden, intermediate // 16),
        gemm2_bias=None,
        output1_scale_scalar=torch.ones(experts, device="cuda"),
        output1_scale_gate_scalar=torch.ones(experts, device="cuda"),
        output2_scale_scalar=torch.ones(experts, device="cuda"),
        num_experts=experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate,
        local_expert_offset=0,
        local_num_experts=experts,
        routed_scaling_factor=1.0,
        routing_method_type=0,
        do_finalize=True,
        enable_pdl=False,
    )
    expected = core.trtllm_fp4_block_scale_routed_moe(**kwargs)[0].clone()
    actual = trtllm_fp4_block_scale_routed_moe(**kwargs)[0]
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
