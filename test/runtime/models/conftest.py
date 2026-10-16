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

"""Shared fixtures and CPU-launch helpers of the Beam runtime tests."""

import dataclasses
import json

import pytest
import torch


@pytest.fixture
def server(monkeypatch):
    """Server arguments plus the vendor MoE's FP32-bias routing hook, faked.

    ``set_args(kernel_routing=...)`` is what the hook answers: whether the
    TRT-LLM kernel can route with the FP32 correction bias (nothing is built
    on CPU).
    """
    from tokenspeed_kernel.registry import KernelRegistry

    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.utils.env import global_server_args_dict

    routing = {"kernel": False}
    impl = KernelRegistry.get().get_impl("flashinfer_trtllm_unquant_moe_apply")
    if impl is not None:
        monkeypatch.setattr(
            impl, "_tokenspeed_fp32_correction_bias", lambda: routing["kernel"]
        )
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    def set_args(backend="auto", numerics="auto", kernel_routing=False):
        monkeypatch.setattr(moe_utils, "MOE_BACKEND", moe_utils.MoeBackend(backend))
        monkeypatch.setitem(global_server_args_dict, "numerics", numerics)
        routing["kernel"] = kernel_routing

    yield set_args
    torch.set_default_dtype(original_dtype)


# (vendor, major, minor) of the platforms a test switches to after ServerArgs.
PLATFORMS = {
    "sm103": ("nvidia", 10, 3),
    "sm90": ("nvidia", 9, 0),
}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """A tiny Beam checkpoint, and the ``config.json`` of another model (a tiny
    Qwen3) for launches that must not take Beam's defaults."""
    from test.runtime.models import beam_fixture

    root = tmp_path_factory.mktemp("checkpoints")
    other = root / "other"
    other.mkdir()
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16",
    }
    (other / "config.json").write_text(json.dumps(config))
    return {"beam": beam_fixture.write_tiny_checkpoint(root / "beam"), "other": other}


def server_args(path, **overrides):
    """The ServerArgs of one CPU launch of ``path`` (rank 0)."""
    from tokenspeed.runtime.utils.server_args import ServerArgs

    launch = dict(
        model=str(path),
        device="cpu",
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=512,
        max_num_seqs=2,
        chunked_prefill_size=256,
        max_total_tokens=512,
        enforce_eager=True,
        disable_prefill_graph=True,
        seed=0,
    )
    args = ServerArgs(**{**launch, **overrides})
    args.mapping.rank = 0
    return args


def model_config(args, override_args="{}", is_draft_worker=False):
    """The target (or draft) ModelConfig the event loop builds for ``args``."""
    from tokenspeed.runtime.configs.model_config import ModelConfig

    return ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        context_length=args.max_model_len,
        model_override_args=override_args,
        dtype=args.dtype,
        quantization=args.quantization,
        server_args=args,
        is_draft_worker=is_draft_worker,
    )


def switch_platform(case):
    """Make ``PLATFORMS[case]`` the current platform (the caller restores it)."""
    from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform

    vendor, major, minor = PLATFORMS[case]
    Platform.override(
        dataclasses.replace(
            current_platform(), vendor=vendor, arch_version=ArchVersion(major, minor)
        )
    )
