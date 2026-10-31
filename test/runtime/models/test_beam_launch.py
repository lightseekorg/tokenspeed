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

"""What a Beam launch resolves before it serves, on CPU.

The config and its refusals, the KV cache layout, the attention and MoE
backend defaults, TF32, the all-reduce fusion switch, the tokenizer, and the
DFlash2 drafter: its RoPE range, the contexts it serves and what Beam hands
it. Launches resolve through ``ServerArgs`` and ``ModelConfig`` on tiny
checkpoints, as the runtime does.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform

from tokenspeed.runtime.configs import model_config as model_config_module
from tokenspeed.runtime.configs.beam_config import (
    BEAM_OBSOLETE_CONFIG_FIELDS,
    BeamConfig,
)
from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.entrypoints import engine as engine_module
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.moe import utils as moe_utils
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.beam import BeamDraftEmbedding, BeamForCausalLM
from tokenspeed.runtime.models.beam_dflash2 import DFlash2BeamDraftModel
from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel
from tokenspeed.runtime.utils import server_args as server_args_module
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import kernel_override_env_key

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.models.beam_fixture import tiny_config, write_tiny_checkpoint
from test.runtime.models.conftest import model_config as _model_config
from test.runtime.models.conftest import server_args as _server_args
from test.runtime.models.conftest import switch_platform as _switch_platform

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu", disabled_on_runners=["amd-*"])


@pytest.fixture
def platform():
    """Switch to a platform of ``conftest.PLATFORMS``; restored afterwards."""
    original = current_platform()
    yield _switch_platform
    Platform.override(original)


# ---------------------------------------------------------------- config
def test_the_config_round_trips(tmp_path):
    config = tiny_config()
    raw = config.to_dict()
    assert not BEAM_OBSOLETE_CONFIG_FIELDS.intersection(raw)
    assert (raw["model_type"], raw["architectures"]) == ("beam", ["BeamForCausalLM"])
    assert config.no_rope_layers == [1, 0] and config.mlp_only_layers == [0]
    assert BeamConfig.from_dict(raw).to_dict() == raw
    config.save_pretrained(tmp_path)
    restored = BeamConfig.from_pretrained(tmp_path)
    assert restored.rope_parameters == config.rope_parameters
    assert restored.mlp_layer_types == ["dense", "sparse"]


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda raw: raw.pop("num_experts"), "missing required"),
        (lambda raw: raw.update(ssmax=True), "obsolete or unsupported"),
        (lambda raw: raw.update(seq_length=4096), "obsolete or unsupported"),
        (lambda raw: raw.update(model_type="llama"), "model_type must be 'beam'"),
        (lambda raw: raw.update(mlp_layer_types=["dense"]), "one entry per layer"),
        (lambda raw: raw.update(routed_scaling_factor=float("nan")), "finite"),
        (lambda raw: raw.update(num_experts_per_tok=8), "selected expert groups"),
        (lambda raw: raw.update(attention_dropout=0.1), "attention_dropout=0"),
        (
            lambda raw: raw["rope_parameters"].update(
                full_attention={"rope_type": "x"}
            ),
            "NoPE",
        ),
    ],
    ids=[
        "missing",
        "fixed-field",
        "context-alias",
        "model-type",
        "layer-count",
        "scale",
        "topk",
        "dropout",
        "nope",
    ],
)
def test_the_config_refuses_what_beam_does_not_serve(edit, match):
    raw = tiny_config().to_dict()
    edit(raw)
    with pytest.raises(ValueError, match=match):
        BeamConfig.from_dict(raw)


def test_an_override_cannot_set_a_field_beam_fixes():
    config = tiny_config()
    raw = config.to_dict()
    with pytest.raises(ValueError, match="cannot be overridden"):
        config.update({"lm_head_dtype": "fp32"})
    with pytest.raises(ValueError, match="cannot be overridden"):
        BeamConfig.from_dict(raw, seq_length=4096)
    assert config.to_dict() == raw


# ---------------------------------------------------------------- KV cache layout
# Beam's released stack: a full-attention layer after every five sliding
# layers, plus a final full layer (43 sliding + 9 full).
_FULL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 51}
_LAYER_TYPES = [
    "full_attention" if i in _FULL_LAYERS else "sliding_attention" for i in range(52)
]


def _released_config() -> BeamConfig:
    raw = tiny_config().to_dict()
    raw.update(
        num_hidden_layers=52,
        layer_types=_LAYER_TYPES,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 49,
    )
    return BeamConfig.from_dict(raw)


def _attn_config(layer_types, window) -> AttnConfig:
    # TP8 over 8 KV heads: one 128-wide BF16 KV head per rank and layer.
    spec = MHAConfig(
        backend_name="trtllm",
        num_attention_heads=48,
        num_kv_heads=8,
        head_dim=128,
        attn_tp_size=8,
        cache_layer_types=tuple(layer_types),
        sliding_window_tokens=window,
    )
    return AttnConfig(
        components=(spec,),
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=64,
        context_len=262144,
        max_bs=64,
    )


def test_sliding_layers_split_into_groups_no_larger_than_the_full_group():
    config = _released_config()
    labels = config.cache_layer_types
    assert [i for i, label in enumerate(labels) if label == "full_attention"] == (
        sorted(_FULL_LAYERS)
    )
    counts = {label: labels.count(label) for label in set(labels)}
    assert counts == {
        "full_attention": 9,
        "sliding_attention_0": 9,
        "sliding_attention_1": 9,
        "sliding_attention_2": 9,
        "sliding_attention_3": 8,
        "sliding_attention_4": 8,
    }
    # Compute labels and the serialized checkpoint stay as they were.
    assert config.layer_types == _LAYER_TYPES
    assert "cache_layer_types" not in config.to_dict()


@pytest.mark.parametrize(
    ("draft_layers", "layer_slots"),
    # Without a drafter a page holds the 9 full layers; a DFlash drafter's
    # layers ride the full-attention group, so a page then holds 9 + 5.
    ((0, 9), (5, 14)),
)
def test_a_page_holds_only_the_layers_its_group_keeps(draft_layers, layer_slots):
    config = _released_config()
    budget = 1 << 30
    setup = prepare_cache_setup(
        family="mha",
        server_args=SimpleNamespace(max_total_tokens=None, kv_offload_config=None),
        model_config=SimpleNamespace(num_attention_layers=52, hf_config=None),
        attn_config=_attn_config(config.cache_layer_types, config.sliding_window),
        draft_model_config=(
            SimpleNamespace(num_attention_layers=draft_layers, hf_config=None)
            if draft_layers
            else None
        ),
        draft_attn_config=(
            _attn_config(("full_attention",) * draft_layers, None)
            if draft_layers
            else None
        ),
        cache_budget_bytes=budget,
        probe_batch_rows=None,
        decode_input_tokens=16,
        overlap_schedule_depth=1,
    )
    plan = setup.spec.memory_plan
    # K and V, 128 BF16 values each, per token and layer slot.
    assert plan.lcm_block_bytes == 64 * layer_slots * 2 * 128 * 2
    assert {group.group_id for group in plan.groups} == {
        "full_attention",
        *(f"sliding_attention_{k}" for k in range(5)),
    }
    # The budget sizes the pages the plan lays out: none of it is left over
    # beyond one page and the reserved null page.
    assert budget - plan.arena_bytes < 2 * plan.lcm_block_bytes


# ---------------------------------------------------------------- backend defaults
ATTENTION_APPLIED = "Beam default attention_backend=trtllm; pass --attention-backend"
ATTENTION_NOT_APPLIED = (
    "Beam default attention_backend=trtllm not applied, left unset: "
)
MOE_APPLIED = (
    "Beam default moe_backend=flashinfer_trtllm; pass --moe-backend to choose another."
)
MOE_NOT_APPLIED = "Beam default moe_backend=flashinfer_trtllm not applied, left auto: "
DECODE_KEY = kernel_override_env_key("attention", "mha_decode_with_kvcache")
# Beam's routed-expert width; the tiny checkpoint's is 64.
BEAM_WIDTH = json.dumps({"moe_intermediate_size": 2560})
NOT_SM100 = "platform nvidia 9.0 is not NVIDIA SM100 to SM103"


@pytest.fixture
def sm100(monkeypatch):
    """Synthetic SM100, the multiple of 64 the BF16 MoE kernels take where
    FlashInfer's launcher is relaxed (they read it from the host GPU at
    import), no MHA override variable; the platform restored afterwards."""
    import tokenspeed_kernel.ops.moe.flashinfer as moe_flashinfer

    monkeypatch.setattr(moe_flashinfer, "TRTLLM_UNQUANT_ISPP_ALIGNMENT", 64)
    prefix = kernel_override_env_key("attention", "mha_")
    for key in [key for key in os.environ if key.startswith(prefix)]:
        monkeypatch.delenv(key)
    original = current_platform()
    Platform.override(dataclasses.replace(original, arch_version=ArchVersion(10, 0)))
    yield
    Platform.override(original)


def _lines(caplog, kind):
    prefix = f"Beam default {kind}_backend"
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def _resolve(path, caplog, platform_case=None, override_args="{}", **launch):
    """One launch's ServerArgs after ModelConfig applied the model defaults."""
    caplog.set_level(logging.INFO, logger=model_config_module.__name__)
    # ServerArgs resolves its own platform-dependent fields on SM100 first.
    args = _server_args(path, **launch)
    if platform_case is not None:
        _switch_platform(platform_case)
    _model_config(args, override_args)
    return args


@pytest.mark.parametrize(
    "launch, platform_case, env, reason",
    [
        ({}, None, None, None),
        ({}, "sm103", None, None),
        (
            {"kv_cache_dtype": "fp8"},
            None,
            None,
            "--kv-cache-dtype fp8_e4m3 is not BF16",
        ),
        (
            {"numerics": "rl-bitwise"},
            None,
            None,
            "--numerics rl-bitwise: the TRT-LLM decode splits KV by batch bucket",
        ),
        ({}, "sm90", None, NOT_SM100),
        (
            {},
            None,
            ("attention", "mha_decode_with_kvcache"),
            f"MHA kernel override {DECODE_KEY}=triton_mha_decode_with_kvcache",
        ),
    ],
    ids=[
        "default",
        "sm103",
        "fp8-kv",
        "rl-bitwise",
        "sm90",
        "mha-override",
    ],
)
def test_the_attention_default(
    checkpoints, caplog, monkeypatch, sm100, launch, platform_case, env, reason
):
    """An unset --attention-backend becomes the TRT-LLM leaf on NVIDIA SM100 to
    SM103 for a BF16 KV cache, --numerics auto and no attention.mha_* override;
    elsewhere it stays unset, with one line naming why."""
    if env is not None:
        monkeypatch.setenv(kernel_override_env_key(*env), f"triton_{env[1]}")
    args = _resolve(checkpoints["beam"], caplog, platform_case, **launch)
    if reason is None:
        assert args.attention_backend == "trtllm"
        assert [
            line[: len(ATTENTION_APPLIED)] for line in _lines(caplog, "attention")
        ] == [ATTENTION_APPLIED]
    else:
        assert args.attention_backend is None
        assert _lines(caplog, "attention") == [ATTENTION_NOT_APPLIED + reason]


@pytest.mark.parametrize(
    "launch, platform_case, override_args, reason",
    [
        ({}, None, "{}", None),
        ({"world_size": 8}, None, BEAM_WIDTH, None),
        ({"world_size": 8, "ep_size": 8}, None, BEAM_WIDTH, None),
        ({}, "sm90", "{}", NOT_SM100),
        ({"dtype": "float16"}, None, "{}", "model dtype float16 is not BF16"),
        (
            {"quantization": "fp8"},
            None,
            "{}",
            "quantization fp8 (the vendor MoE runs BF16 experts)",
        ),
        ({"all2all_backend": "deepep"}, None, "{}", "--all2all-backend deepep"),
        (
            {"world_size": 16, "ep_size": 16},
            None,
            "{}",
            "moe ep_size 16 does not divide num_experts 8",
        ),
        (
            {"world_size": 16},
            None,
            BEAM_WIDTH,
            "intermediate size per MoE rank 160 (moe_intermediate_size 2560 / "
            "moe tp_size 16) is not a multiple of 64",
        ),
    ],
    ids=[
        "tiny-tp1",
        "beam-tp8",
        "beam-ep8",
        "sm90",
        "float16",
        "quantized",
        "deepep",
        "ep16",
        "tp16",
    ],
)
def test_the_moe_default(
    checkpoints, caplog, sm100, launch, platform_case, override_args, reason
):
    """An auto --moe-backend becomes the vendor MoE (Beam's 2,560 is 320 per TP8
    rank and whole experts under EP8) where its policy starts; elsewhere auto
    stays (the Triton MoE), with one line naming why."""
    args = _resolve(checkpoints["beam"], caplog, platform_case, override_args, **launch)
    if reason is None:
        assert args.moe_backend == "flashinfer_trtllm"
        assert moe_utils.MoeBackend(args.moe_backend) is (
            moe_utils.MoeBackend.FLASHINFER_TRTLLM
        )
        assert _lines(caplog, "moe") == [MOE_APPLIED]
    else:
        assert args.moe_backend == "auto"
        assert _lines(caplog, "moe") == [MOE_NOT_APPLIED + reason]


def test_a_128_aligned_launcher_keeps_the_triton_moe_at_tp8(
    checkpoints, caplog, monkeypatch, sm100
):
    """Where FlashInfer's launcher cannot be relaxed the BF16 kernels take
    multiples of 128 only: Beam's 320 per TP8 rank keeps auto; the 2,560 per
    EP8 rank still takes the vendor MoE."""
    import tokenspeed_kernel.ops.moe.flashinfer as moe_flashinfer

    monkeypatch.setattr(moe_flashinfer, "TRTLLM_UNQUANT_ISPP_ALIGNMENT", 128)
    tp8 = _resolve(checkpoints["beam"], caplog, None, BEAM_WIDTH, world_size=8)
    ep8 = _resolve(
        checkpoints["beam"], caplog, None, BEAM_WIDTH, world_size=8, ep_size=8
    )
    assert (tp8.moe_backend, ep8.moe_backend) == ("auto", "flashinfer_trtllm")


def test_explicit_backends_and_rl_bitwise_keep_their_choice(checkpoints, caplog, sm100):
    """An explicit backend is never replaced, and --numerics rl-bitwise folds
    auto available for capability-based expert planning."""
    args = _resolve(
        checkpoints["beam"], caplog, attention_backend="triton", moe_backend="triton"
    )
    assert (args.attention_backend, args.moe_backend) == ("triton", "triton")
    assert _lines(caplog, "attention") == _lines(caplog, "moe") == []
    args = _resolve(checkpoints["beam"], caplog, numerics="rl-bitwise")
    assert args.moe_backend == "auto" and _lines(caplog, "moe") == []


def test_other_models_and_the_drafter_are_not_asked(checkpoints, caplog, sm100):
    args = _resolve(checkpoints["other"], caplog)
    assert (args.attention_backend, args.moe_backend) == (None, "auto")
    args = _server_args(checkpoints["beam"])
    _model_config(args, is_draft_worker=True)
    assert args.moe_backend == "auto"
    assert _lines(caplog, "attention") == _lines(caplog, "moe") == []


# ---------------------------------------------------------------- TF32
TF32_KEYS = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")


@pytest.fixture
def environ(monkeypatch):
    """Engine setup on a private copy of the environment without TF32
    variables; its process-wide signal handler, start method, file limit and
    socket interface are left alone."""
    env = {key: value for key, value in os.environ.items() if key not in TF32_KEYS}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(engine_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(engine_module.mp, "set_start_method", lambda *a, **k: None)
    monkeypatch.setattr(engine_module, "set_ulimit", lambda: None)
    monkeypatch.setattr(engine_module, "_set_socket_interface", lambda args: None)
    return env


def _setup(environ, caplog, path, caller, **overrides):
    """The environment engine setup leaves for ``path``, and its warnings."""
    environ.update(caller)
    args = _server_args(path, **overrides)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=engine_module.__name__):
        engine_module._set_envs_and_config(args)
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == engine_module.__name__ and r.levelno == logging.WARNING
    ]
    return args, warnings


@pytest.mark.parametrize(
    "caller",
    [{}, dict.fromkeys(TF32_KEYS, "1"), dict.fromkeys(TF32_KEYS, "0")],
    ids=["unset", "tf32-on", "tf32-off"],
)
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "disable-tf32"])
def test_beam_turns_tf32_off_for_its_processes(
    checkpoints, environ, caplog, caller, explicit
):
    """A Beam checkpoint sets --disable-tf32, and engine setup sets both TF32
    variables to 0 for the processes it starts, replacing a caller's 1 with a
    warning."""
    overrides = {"disable_tf32": True} if explicit else {}
    args, warnings = _setup(environ, caplog, checkpoints["beam"], caller, **overrides)
    assert args.disable_tf32 and args._beam_checkpoint
    assert {key: environ.get(key) for key in TF32_KEYS} == dict.fromkeys(TF32_KEYS, "0")
    assert warnings == [
        f"Beam (no TF32) replaces {key}=1 with 0"
        for key in TF32_KEYS
        if caller.get(key) == "1"
    ]


@pytest.mark.parametrize(
    "overrides, expected",
    [({}, dict.fromkeys(TF32_KEYS, "1")), ({"disable_tf32": True}, {})],
    ids=["default", "disable-tf32"],
)
def test_another_model_keeps_its_tf32_setup(
    checkpoints, environ, caplog, overrides, expected
):
    args, warnings = _setup(environ, caplog, checkpoints["other"], {}, **overrides)
    assert not args._beam_checkpoint and args.disable_tf32 == bool(overrides)
    assert {key: environ.get(key) for key in TF32_KEYS} == {
        key: expected.get(key) for key in TF32_KEYS
    }
    assert warnings == []


def test_a_beam_checkpoint_in_the_hugging_face_cache(
    checkpoints, tmp_path, monkeypatch, environ
):
    """A hub id is read from the local cache, at refs/main or a revision; a hub
    id the cache does not hold is left to engine setup."""
    from huggingface_hub import constants

    commit = "0" * 40
    repo = tmp_path / "models--org--beam"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    shutil.copy(checkpoints["beam"] / "config.json", snapshot / "config.json")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    for revision in (None, commit):
        args = server_args_module.ServerArgs(model="org/beam", revision=revision)
        assert args.disable_tf32 and args._beam_checkpoint
    engine_module._set_envs_and_config(_server_args("org/beam"))
    assert {key: environ.get(key) for key in TF32_KEYS} == dict.fromkeys(TF32_KEYS, "0")
    args = server_args_module.ServerArgs(model="org/other")
    assert not args.disable_tf32 and args._beam_checkpoint is None


# ---------------------------------------------------------------- all-reduce fusion
def _fusion_launch(path, nnodes, **overrides):
    return _server_args(
        path,
        world_size=8,
        attn_tp_size=8,
        nnodes=nnodes,
        dist_init_addr="127.0.0.1:5000",
        **overrides,
    )


@pytest.mark.parametrize(
    "model, nnodes, case, expected",
    [
        ("beam", 1, "sm103", True),
        ("beam", 2, "sm103", True),
        ("beam", 2, "sm90", True),
        ("other", 1, "sm103", True),
        ("other", 2, "sm103", False),
    ],
)
def test_the_fusion_is_auto_enabled_for_beam_at_any_node_count(
    checkpoints, platform, model, nnodes, case, expected
):
    """Beam on Hopper or newer: on at any node count (the boundaries fuse where
    the MNNVL workspace serves them). Another model: on one node only."""
    platform(case)
    args = _fusion_launch(checkpoints[model], nnodes)
    assert args.enable_allreduce_fusion is expected


@pytest.mark.parametrize(
    "overrides",
    [{"layer_boundary_norm": "unfused"}, {"ep_size": 8, "moe_combine_order": "slot"}],
    ids=["unfused-boundary-norm", "slot-combine"],
)
def test_trainer_order_switches_still_turn_it_off_for_beam(
    checkpoints, platform, overrides
):
    platform("sm103")
    assert not _fusion_launch(
        checkpoints["beam"], 2, **overrides
    ).enable_allreduce_fusion


class _Hub:
    """The Hugging Face Hub, online: serves ``repos[repo_id]/<file>`` and
    records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.repos: dict[str, Path] = {}

    def hf_hub_download(
        self, repo_id, filename, *, revision=None, local_files_only=False
    ):
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.calls.append((repo_id, filename, revision, local_files_only))
        if local_files_only or repo_id not in self.repos:
            raise LocalEntryNotFoundError(f"{repo_id}/{filename} is not available")
        return str(self.repos[repo_id] / filename)


@pytest.mark.parametrize("model", ["beam", "other"])
def test_a_first_launch_from_a_hub_id_resolves_as_from_the_directory(
    checkpoints, platform, environ, tmp_path, monkeypatch, model
):
    """A hub id the local cache does not hold: the arguments resolve without
    the network; engine setup downloads config.json alone before any process
    starts, then resolves TF32 and the fusion across nodes as from the
    directory, so every node of a launch decides from the same file."""
    import huggingface_hub
    from huggingface_hub import constants

    hub = _Hub()
    hub.repos["org/model"] = checkpoints[model]
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.hf_hub_download)
    platform("sm103")
    directory = _fusion_launch(checkpoints[model], 2)
    args = _fusion_launch("org/model", 2)
    assert args._beam_checkpoint is None and not args.disable_tf32
    assert not args.enable_allreduce_fusion and hub.calls == []
    engine_module._set_envs_and_config(args)
    assert hub.calls == [("org/model", "config.json", None, False)]
    beam = model == "beam"
    assert args._beam_checkpoint is beam and args.disable_tf32 is beam
    assert args.enable_allreduce_fusion is directory.enable_allreduce_fusion is beam
    assert {key: environ.get(key) for key in TF32_KEYS} == dict.fromkeys(
        TF32_KEYS, "0" if beam else "1"
    )


# ---------------------------------------------------------------- tokenizer
def test_the_registered_tokenizer_loads_local_canonical_metadata(tmp_path):
    from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer

    checkpoint = write_tiny_checkpoint(tmp_path / "beam")
    tokenizer = get_tokenizer(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        architectures=["BeamForCausalLM"],
    )
    assert tokenizer.encode("t4 t5", add_special_tokens=False) == [4, 5]


# ---------------------------------------------------------------- DFlash2 drafter
def _drafter_config(max_position_embeddings):
    """Drafter config (RoPE, window, block) on one small layer."""
    return Qwen3Config(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=max_position_embeddings,
        rope_theta=10000.0,
        use_sliding_window=True,
        sliding_window=4096,
        layer_types=["sliding_attention"],
        dflash_config={
            "block_size": 16,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": [0],
        },
    )


@pytest.mark.parametrize(
    ("positions", "allow_override"),
    [(262144, False), (262144, True), (1048576, False)],
    ids=["synthetic-mismatch", "synthetic-override", "released-no-override"],
)
def test_the_drafter_context_limit_at_launch(
    monkeypatch, tmp_path, positions, allow_override
):
    """Validate 1M serving from metadata without allocating RoPE tables."""
    monkeypatch.delenv("TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", raising=False)
    if allow_override:
        monkeypatch.setenv("TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    config = _drafter_config(positions)
    config.architectures = ["DFlash2BeamDraftModel"]
    config.save_pretrained(tmp_path)
    args = _server_args(tmp_path, max_model_len=1048576)
    if positions < args.max_model_len and not allow_override:
        with pytest.raises(ValueError, match="context_length.*1048576.*262144"):
            _model_config(args, is_draft_worker=True)
    else:
        loaded = _model_config(args, is_draft_worker=True)
        assert loaded.context_len == 1048576
        assert loaded.hf_config.max_position_embeddings == positions


def _beam(monkeypatch, max_model_len, draft_tokens=None, device="cpu"):
    """A tiny Beam (a 2048-token context) under --max-model-len and, with
    ``draft_tokens``, the DFlash2 drafter's verify width."""
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", moe_utils.MoeBackend.TRITON)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", max_model_len)
    monkeypatch.setitem(
        global_server_args_dict,
        "speculative_algorithm",
        None if draft_tokens is None else "DFLASH",
    )
    monkeypatch.setitem(
        global_server_args_dict, "speculative_num_draft_tokens", draft_tokens
    )
    with torch.device(device):
        return BeamForCausalLM(tiny_config(), Mapping(rank=0, world_size=1), None)


def _rope_table(model):
    return model.model.layers[0].self_attn.rotary_emb.cos_sin_cache


def test_the_draft_rope_table_covers_speculation_past_the_context(monkeypatch):
    """The kernels index the table by position and speculation writes three
    blocks past the context: the table covers them and one more block, and
    keeps the checkpoint's rows bit for bit (built on one CPU thread, as the
    vectorized cos/sin round by the thread split)."""
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 128)
    import tokenspeed.runtime.layers.rotary_embedding as rotary_module

    # get_rope caches a table by its length only: build both tables here.
    monkeypatch.setattr(rotary_module, "_ROPE_DICT", {})
    mapping = Mapping(rank=0, world_size=1)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint = DFlash2DraftModel(_drafter_config(128), mapping)
        draft = DFlash2BeamDraftModel(_drafter_config(128), mapping)
    finally:
        torch.set_num_threads(threads)
    attention = draft.layers[0].self_attn
    table = attention.rotary_emb.cos_sin_cache
    assert table.shape[0] == 128 + 4 * 16
    assert attention.attn.rotary_emb is attention.rotary_emb
    reference = checkpoint.layers[0].self_attn.rotary_emb.cos_sin_cache
    assert torch.equal(table[:128], reference)
    assert draft.config.max_position_embeddings == 128


def test_without_max_model_len_the_drafter_refuses_a_longer_target_context(
    monkeypatch,
):
    """A synthetic shorter drafter refuses the target context; a covered
    context is served. Small tables exercise the same bounds as 1M serving."""
    monkeypatch.setitem(global_server_args_dict, "max_model_len", None)
    draft = DFlash2BeamDraftModel(_drafter_config(128), Mapping(rank=0, world_size=1))
    target = _beam(monkeypatch, None)
    raw = tiny_config().to_dict()
    with pytest.raises(
        ValueError,
        match="up to 128 tokens of context.*--max-model-len 128 or less",
    ):
        draft.configure_target(
            target, BeamConfig.from_dict(raw | {"max_position_embeddings": 256})
        )
    assert target.model.layers_to_capture == []
    draft.configure_target(
        target, BeamConfig.from_dict(raw | {"max_position_embeddings": 128})
    )
    assert target.model.layers_to_capture == [1]


# Small synthetic tables cover the native limit and an accepted override.
@pytest.mark.parametrize(("max_model_len", "rows"), [(128, 192), (160, 224)])
def test_the_drafter_serves_the_max_model_len_its_table_covers(
    monkeypatch, max_model_len, rows
):
    target = _beam(monkeypatch, max_model_len)
    draft = DFlash2BeamDraftModel(_drafter_config(128), Mapping(rank=0, world_size=1))
    assert draft.layers[0].self_attn.rotary_emb.cos_sin_cache.shape[0] == rows
    draft.configure_target(
        target,
        BeamConfig.from_dict(
            tiny_config().to_dict() | {"max_position_embeddings": 256}
        ),
    )
    assert target.model.layers_to_capture == [1]


# Below the context, the old --max-model-len for the drafter (48 under it), and
# the context itself: the table covers the context, the drafter's three-block
# pad and one more block, or the context alone without the drafter.
@pytest.mark.parametrize("max_model_len", [1024, 2000, 2048])
@pytest.mark.parametrize(("draft_tokens", "rows"), [(None, 2048), (16, 2048 + 64)])
def test_beams_rope_table_covers_what_speculation_writes_past_the_context(
    monkeypatch, max_model_len, draft_tokens, rows
):
    target = _beam(monkeypatch, max_model_len, draft_tokens)
    assert _rope_table(target).shape[0] == rows
    assert target.rope_positions == rows


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_beams_longer_rope_table_keeps_the_rows_of_the_context(monkeypatch, device):
    # One thread on the CPU, as for the drafter's table above.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        context = _rope_table(_beam(monkeypatch, 2048, None, device))
        longer = _rope_table(_beam(monkeypatch, 2048, 16, device))
    finally:
        torch.set_num_threads(threads)
    assert context.shape[0] == 2048
    assert torch.equal(longer[:2048], context)


def test_the_draft_lookup_is_a_tp_partial_of_the_replicated_rows():
    """A drafter all-reduces its block's rows over the TP group: rank 0 hands
    on Beam's replicated rows, the other ranks zeros, so the sum is the rows."""
    table = VocabParallelEmbedding(
        16, 8, params_dtype=torch.bfloat16, org_num_embeddings=16, padding_size=1
    )
    with torch.no_grad():
        table.weight.normal_()
    ids = torch.tensor([0, 5, 15, 5])
    rows = table(ids)
    partials = [
        BeamDraftEmbedding(table, rank, 4)(ids, reduce_results=False)
        for rank in range(4)
    ]
    assert torch.equal(partials[0], rows)
    assert not any(partial.any() for partial in partials[1:])
    assert torch.equal(sum(partials), rows)
    for rank in range(4):
        assert torch.equal(BeamDraftEmbedding(table, rank, 4)(ids), rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
