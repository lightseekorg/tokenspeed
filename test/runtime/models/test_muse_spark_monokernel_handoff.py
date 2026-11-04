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

"""The fused Muse Spark layers' launch arguments, built as serving builds
them, checked without a GPU.

A launch whose argument types differ from the ones prepare compiled for
fails only on the GPU, after the server has loaded. This builds the real
model at Muse Spark's dimensions (MXFP4 experts processed by the loader's
hooks) on the meta device, prepares its fused layers through the layer
stack, and drives a decode step of every served shape through the real
decoder layers, the real attention backend's decode view and prologue
target, and the leaf's own launch code. At the launch, every argument's
type is checked against the compiled signature, and every weight against
the shape and dtype the leaf requires; all mismatches are reported at once.

Only what has no CPU form is replaced: the process groups, the Iris heap and
uncached allocations, the residual-gate and router-weight kernels (by their
torch equivalents), the split-schedule kernel, and the compile and the
launch themselves.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest import mock

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel.platform import ArchVersion, current_platform  # noqa: E402

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed_kernel._triton import triton  # noqa: E402
from tokenspeed_kernel.ops.monokernel import (  # noqa: E402
    PlanKey,
    prepare_monokernel_state,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark import (  # noqa: E402
    layer as composer,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark import (  # noqa: E402
    prepared,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (  # noqa: E402
    deadline_ticks,
)

from tokenspeed.runtime.configs.muse_spark_config import (  # noqa: E402
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    MuseSparkConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.layers.attention.backends.paged import mha  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (  # noqa: E402
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import (  # noqa: E402
    CacheGroupRouter,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.kv_cache.mha import (  # noqa: E402
    MHATokenToKVPool,
)
from tokenspeed.runtime.layers.monokernel import WAIT_DEADLINE_SECONDS  # noqa: E402
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config  # noqa: E402
from tokenspeed.runtime.models import muse_spark  # noqa: E402
from tokenspeed.runtime.models.muse_spark_layer_stack import (  # noqa: E402
    FUSED_REQUESTS,
    MuseSparkLayerStack,
)

WORLD = 8
VERIFY_WIDTH = 8
PAGE = 64
# Muse Spark's sliding layers keep a 2048-token window.
WINDOW = 2048
SERVER_ARGS = {
    "speculative_algorithm": "DFLASH",
    "speculative_num_draft_tokens": VERIFY_WIDTH,
    "kv_cache_dtype": "fp8_e4m3",
}


@pytest.fixture(scope="module", autouse=True)
def gfx950():
    platform = current_platform()
    if not platform.is_amd or platform.arch_version != ArchVersion(9, 5):
        pytest.skip("the fused Muse Spark layer is a gfx950 kernel")


def muse_spark_text_config() -> dict:
    """Muse Spark's text model (the published checkpoint's values)."""
    layers = 62
    layer_types = [
        FULL_ATTENTION if (layers - layer - 1) % 4 == 0 else SLIDING_ATTENTION
        for layer in range(layers)
    ]
    return {
        "vocab_size": 202048,
        "hidden_size": 8192,
        "head_dim": 64,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "num_hidden_layers": layers,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "final_logit_softcapping": 20.0,
        "qk_scale_factor": 5.4730064863875,
        "output_multiplier": 0.17677669529663687,
        "residual_gate_temperature": 0.3,
        "sliding_window": WINDOW,
        "layer_types": layer_types,
        "layer_rope_theta": [
            0.0 if kind == FULL_ATTENTION else 500000.0 for kind in layer_types
        ],
        "mlp_layer_types": ["sparse"] * layers,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "moe_hidden_size": 4096,
        "moe_intermediate_size": 4096,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
        "routed_scaling_factor": 1.0,
    }


def muse_spark_config() -> MuseSparkConfig:
    """The text model above, with small vision and audio towers (the fused
    layers never see them)."""
    return MuseSparkConfig(
        text_config=muse_spark_text_config(),
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "layer_types": ["full_attention"],
            "layer_norm_eps": 1e-5,
            "merge_size": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
            "max_position_embeddings": 4,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "hidden_act": "gelu",
        },
        audio_config={
            "conv_kernel_size": 7,
            "depths": [1, 1, 1, 1],
            "hidden_act": "gelu",
            "hidden_sizes": [8, 16, 32, 64],
            "mlp_ratio": 4.0,
            "num_hidden_layers": 4,
            "out_hidden_size": 8192,
            "patch_sizes": [16, 6, 5, 4],
            "rms_norm_eps": 1e-6,
        },
        image_token_id=200,
        video_token_id=201,
        audio_token_id=202,
        out_hidden_size=128,
        projector_hidden_size=64,
        projector_hidden_act="gelu",
    )


def muse_spark_quant_config(num_layers: int) -> Mxfp4Config:
    """The checkpoint's Quark MXFP4 export: MXFP4 experts with dynamic MXFP4
    activations; attention, router, the expert projections around the
    experts, and the first and last layers' experts stay BF16."""
    mx = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "ch_axis": -1,
        "group_size": 32,
        "scale_format": "e8m0",
        "scale_calculation_mode": "even",
    }
    exclude = ["lm_head"]
    for layer in range(num_layers):
        prefix = f"model.language_model.layers.{layer}"
        exclude += [
            f"{prefix}.self_attn.{name}_proj" for name in ("q", "k", "v", "o", "gate")
        ]
        exclude += [
            f"{prefix}.mlp.{name}"
            for name in ("gate", "pre_expert_proj", "post_expert_proj")
        ]
        if layer in (0, num_layers - 1):
            exclude += [
                f"{prefix}.mlp.experts.{expert}.{name}_proj"
                for expert in range(256)
                for name in ("gate", "up", "down")
            ]
    return Mxfp4Config.from_config(
        {
            "quant_method": "quark",
            "global_quant_config": {
                "input_tensors": {**mx, "is_dynamic": True},
                "weight": {**mx, "is_dynamic": False},
                "output_tensors": None,
                "bias": None,
            },
            "export": {"pack_method": "reorder", "weight_format": "real_quantized"},
            "exclude": exclude,
            "layer_quant_config": {},
        }
    )


def tp8_ep1_mapping(rank: int) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=WORLD,
        attn_tp_size=WORLD,
        attn_dp_size=1,
        dense_tp_size=WORLD,
        dense_dp_size=1,
        moe_tp_size=WORLD,
        moe_ep_size=1,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=WORLD,
        linear_attn_tp_size=WORLD,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=WORLD,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def _gate_coefficients(gate, temperature, eps):
    """The residual-gate kernel's math and output layout, in torch."""
    scaled = gate.float() / temperature
    beta = torch.sigmoid(scaled)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1.0 + beta), min=eps))
    return alpha, beta


@contextmanager
def no_gpu() -> Iterator[None]:
    """Replace what has no CPU form (see the module docstring)."""

    class Peers:
        """An 8-rank Iris group whose symmetric heap is meta memory."""

        def __init__(self, ctx, group, rank, device):
            self.ctx, self.rank, self.device = ctx, rank, device
            self.world_size = WORLD
            self.addresses = [(rank + 1) << 40 for rank in range(WORLD)]

        def finish_prepare(self):
            pass

    class Heap:
        def empty(self, shape, *, dtype):
            return torch.empty(shape, dtype=dtype, device="meta")

        def zeros(self, shape, *, dtype):
            return torch.zeros(shape, dtype=dtype, device="meta")

    class Uncached:
        """An uncached allocation as kernels take it: a pointer and a dtype."""

        def __init__(self, numel, dtype, device):
            self.dtype, self.device = dtype, torch.device(device)
            self.nbytes = numel * torch.empty((), dtype=dtype).element_size()

        def data_ptr(self):
            return 1 << 40

        def ptr_range(self):
            return self.nbytes

    with ExitStack() as stack:
        patch = stack.enter_context
        patch(
            mock.patch.object(
                muse_spark.process_group_manager,
                "get_device_process_group",
                lambda group: None,
            )
        )
        patch(
            mock.patch.object(
                muse_spark, "residual_gate_coefficients", _gate_coefficients
            )
        )
        patch(
            mock.patch.object(
                muse_spark, "fp32_router_gemv_process_weight", lambda plan, w: w
            )
        )
        patch(mock.patch.object(mha, "mha_decode_schedule", lambda **kwargs: None))
        patch(
            mock.patch.object(
                prepared, "get_or_create_context", lambda heap_size: Heap()
            )
        )
        patch(mock.patch.object(prepared, "IrisGroup", Peers))
        patch(mock.patch.object(prepared, "UncachedBuffer", Uncached))
        patch(mock.patch.object(prepared, "FailureMirror", lambda errors, path: None))
        # Offline compiles of every variant are the codegen tests' concern.
        patch(
            mock.patch.object(
                prepared.MuseSparkMonokernelLayer, "_compile", lambda self, key: []
            )
        )
        yield


def build_model(config: MuseSparkConfig, quant_config, rank: int):
    """The served model on the meta device, its weights processed by the
    loader's post-load hooks (the experts into the gfx950 MXFP4 layout)."""
    model_class = muse_spark.MuseSparkForConditionalGeneration
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = model_class(
                config,
                tp8_ep1_mapping(rank),
                quant_config,
                is_multimodal_active=False,
                mm_attention_backend=None,
            )
    finally:
        torch.set_default_dtype(previous)
    # Vectors (norms, gates, biases) are read on the host as well: give them
    # memory, as loading does. Matrices stay meta.
    for module in model.modules():
        for tensors in (module._parameters, module._buffers):
            for name, tensor in tensors.items():
                if tensor is not None and tensor.is_meta and tensor.dim() <= 1:
                    vector = torch.ones(tensor.shape, dtype=tensor.dtype)
                    tensors[name] = (
                        torch.nn.Parameter(vector, requires_grad=False)
                        if isinstance(tensor, torch.nn.Parameter)
                        else vector
                    )
    for _, module in model.model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
        process = getattr(module, "process_weights_after_loading", None)
        if process is not None:
            process(module)
    return model.model


class DecodeStep:
    """A verify step's attention metadata as the runtime builds it: the
    cache-group router over one MHA leaf per group, an FP8 KV pool, and the
    forward context the layers read."""

    def __init__(self, model, config: MuseSparkConfig, max_bs: int):
        text = config.get_text_config()
        heads = text.num_attention_heads
        spec = MHAConfig(
            backend_name="mha",
            num_attention_heads=heads,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            attn_tp_size=WORLD,
        )
        attn_config = AttnConfig(
            device="cpu",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.float8_e4m3fn,
            kv_cache_quant_method="none",
            context_len=text.max_position_embeddings,
            max_bs=max_bs,
            speculative_num_draft_tokens=VERIFY_WIDTH,
            is_draft=False,
            draft_block_decode=False,
            components=(spec,),
        )
        groups = ("full_attention", "sliding_attention")
        self.leaves = {
            group: mha.MHAAttnBackend(attn_config, spec, kernel_page_size=PAGE)
            for group in groups
        }
        self.router = CacheGroupRouter(
            None,
            is_draft=False,
            spec_num_tokens=VERIFY_WIDTH,
            device="cpu",
            consumed_group_ids=None,
        )
        self.router.bind(
            CacheGroupGeometry(
                granularities=dict.fromkeys(groups, PAGE),
                attn_kinds=dict(zip(groups, ("full_history", "sliding"))),
                full_history_group_id=groups[0],
                row_geometry=dict.fromkeys(groups, (PAGE, 1)),
                windows=dict(zip(groups, (None, WINDOW))),
            ),
            self.leaves,
        )
        self.router.init_cuda_graph_state(max_bs)
        for layer in model.layers:
            attention = layer.self_attn.attn
            attention.bind_cache_group(
                groups[0] if attention.sliding_window_size < 0 else groups[1]
            )
        self.pool = self._fp8_pool(len(model.layers), text.num_key_value_heads // WORLD)
        self.max_pages = text.max_position_embeddings // PAGE

    @staticmethod
    def _fp8_pool(layers: int, kv_heads: int) -> MHATokenToKVPool:
        """The FP8 pool's per-layer planes (stored as bytes, read as FP8),
        without the arena that would size them from device memory."""
        pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        pool.dtype = torch.float8_e4m3fn
        pool.store_dtype = torch.uint8
        pool.head_num = kv_heads
        pool.head_dim = 64
        pool.layer_num = layers
        pool.layerwise_load_tracker = None
        pool._layer_kv_head_counts = None
        pool._kv_alloc_head_count = None
        rows = 4096 * PAGE
        pool.k_buffer, pool.v_buffer = (
            [
                torch.empty((rows, kv_heads, 64), dtype=torch.uint8, device="meta")
                for _ in range(layers)
            ]
            for _ in range(2)
        )
        return pool

    def context(self, requests: int) -> ForwardContext:
        """Refresh the decode metadata for ``requests`` verify requests and
        return the step's forward context."""
        seq_lens = torch.tensor(
            [5000 + 777 * request for request in range(requests)], dtype=torch.int32
        )
        tables = torch.arange(requests * self.max_pages, dtype=torch.int32)
        tables = tables.view(requests, self.max_pages)
        self.router.refresh_decode_metadata(
            requests,
            requests,
            torch.arange(requests, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables={group: tables for group in self.leaves},
        )
        rows = requests * VERIFY_WIDTH
        return ForwardContext(
            attn_backend=self.router,
            token_to_kv_pool=self.pool,
            bs=requests,
            num_extends=0,
            input_num_tokens=rows,
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, requests, VERIFY_WIDTH),
        )


def _scalar_problems(args: tuple, constexprs: dict) -> list[str]:
    """Scalar arguments whose launch type differs from the compiled one (the
    leaf's own check covers the tensors)."""
    names = [
        n
        for n in composer.gluon_monokernel_muse_spark_gfx950.arg_names
        if n not in constexprs
    ]
    expected = composer.offline_signature(constexprs)
    problems = []
    for name, arg in zip(names, args):
        if hasattr(arg, "data_ptr"):
            continue
        values = arg if isinstance(arg, tuple) else (arg,)
        offline = expected[name] if isinstance(arg, tuple) else (expected[name],)
        launched = tuple(triton.runtime.jit.mangle_type(value) for value in values)
        if launched != tuple(offline):
            problems.append(f"{name}: launched {launched}, offline {tuple(offline)}")
    return problems


class LaunchRecorder:
    """Stands in for the compile check and the launch: records every
    ``_require`` failure and every argument mismatch the launch code finds,
    and lets the step continue so all are reported at once."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.launched: list[PlanKey] = []
        # The wait deadline (clock ticks) every launch passed.
        self.deadlines: set[int] = set()
        self.layer: str = ""

    @contextmanager
    def recording(self) -> Iterator[None]:
        real_require = prepared._require
        real_signature_problems = prepared.signature_problems

        def require(name, tensor, shape, dtype):
            try:
                return real_require(name, tensor, shape, dtype)
            except ValueError as error:
                self.problems.append(f"{self.layer}: {error}")
                return tensor

        def signature_problems(args, constexprs):
            for problem in real_signature_problems(args, constexprs) + _scalar_problems(
                args, constexprs
            ):
                self.problems.append(f"{self.layer}: {problem}")
            return []

        recorder = self

        class Kernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    names = [
                        n
                        for n in composer.gluon_monokernel_muse_spark_gfx950.arg_names
                        if n not in kwargs
                    ]
                    recorder.deadlines.add(args[names.index("deadline_ticks")])
                    next_kinds = {v: k for k, v in composer._NEXT_KINDS.items()}
                    recorder.launched.append(
                        PlanKey(
                            kwargs["REQUESTS"],
                            "sliding" if kwargs["SLIDING"] else "full",
                            next_kinds[kwargs["NEXT_KIND"]],
                        )
                    )

                return launch

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(prepared, "_require", require))
            stack.enter_context(
                mock.patch.object(prepared, "signature_problems", signature_problems)
            )
            stack.enter_context(
                mock.patch.object(
                    prepared, "gluon_monokernel_muse_spark_gfx950", Kernel()
                )
            )
            stack.enter_context(
                mock.patch.object(prepared, "check_compiled", lambda compiled: [])
            )
            yield


def check_handoff(config: MuseSparkConfig, quant_config, rank: int) -> LaunchRecorder:
    """Prepare rank ``rank``'s fused layers and run one decode step of every
    served shape through them; the recorder holds what the launches found."""
    recorder = LaunchRecorder()
    with no_gpu(), recorder.recording():
        model = build_model(config, quant_config, rank)
        stack = model.layer_stack
        assert isinstance(stack, MuseSparkLayerStack)
        fused = stack._candidate_indices(model)
        assert fused, "no layer can run fused"
        state = prepare_monokernel_state(
            recipe="muse_spark",
            group=None,
            rank_in_group=rank,
            max_requests=max(FUSED_REQUESTS),
            failure_record_path="/dev/null",
            device=torch.device("meta"),
        )
        traits = stack.selection_traits(SERVER_ARGS, model.mapping)
        recorder.layer = "prepare"
        layers = stack.prepare_layers(model, state, traits, WAIT_DEADLINE_SECONDS)
        assert all(layer is not None for layer in layers), "no leaf was selected"
        step = DecodeStep(model, config, max(FUSED_REQUESTS))
        for requests in FUSED_REQUESTS:
            ctx = step.context(requests)
            rows = requests * VERIFY_WIDTH
            positions = torch.arange(rows, dtype=torch.int64)
            hidden = torch.empty((rows, model.config.hidden_size), dtype=torch.float32)
            # The first fused layer's query as its attention's regular
            # prologue hands it over: the prepared query, and the output
            # gate as a view of the projection's rows.
            attention = model.layers[fused.start].self_attn
            target = attention.attn.prologue_target(
                positions, rows, attention.qkv_proj.weight.dtype, ctx
            )
            projected = torch.empty(
                (rows, attention.qkv_proj.weight.shape[0] // WORLD),
                dtype=attention.qkv_proj.weight.dtype,
            )
            gate = projected.split(
                (
                    attention.q_size,
                    attention.kv_size,
                    attention.kv_size,
                    attention.q_size,
                ),
                dim=-1,
            )[3]
            query = (
                torch.empty(
                    (rows, attention.num_heads, attention.head_dim),
                    dtype=target.query_dtype,
                ),
                gate,
            )
            for index in fused:
                recorder.layer = f"layer {index} at {requests} x {VERIFY_WIDTH} rows"
                layer = model.layers[index]
                hidden, query, _ = layer.forward_fused(
                    layers[index - fused.start],
                    positions,
                    query,
                    hidden,
                    ctx,
                    model.layers[index + 1],
                    index + 1 in fused,
                )
    return recorder


@pytest.mark.parametrize("rank", [0, WORLD - 1])
def test_every_fused_launch_matches_its_compiled_signature(rank):
    """Every launch of every plan key a decode step reaches passes the same
    tensors, in the same types and shapes, that the variant was compiled
    for; each mismatch is listed."""
    config = muse_spark_config()
    recorder = check_handoff(config, muse_spark_quant_config(62), rank)
    assert not recorder.problems, f"{len(recorder.problems)} mismatches:\n" + "\n".join(
        recorder.problems
    )
    keys = {
        PlanKey(requests, layer, next_kind)
        for requests in FUSED_REQUESTS
        for layer, next_kind in (
            ("full", "sliding"),
            ("sliding", "sliding"),
            ("sliding", "full"),
            ("sliding", "regular"),
        )
    }
    assert set(recorder.launched) == keys
    # The serving deadline reaches every launch.
    assert recorder.deadlines == {deadline_ticks(WAIT_DEADLINE_SECONDS)}
    # Layers 1-60 run fused at each of the two shapes.
    assert len(recorder.launched) == 2 * 60
