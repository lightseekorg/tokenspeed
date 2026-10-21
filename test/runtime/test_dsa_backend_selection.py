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

"""One DSA launch name selects the device implementation and its constraints."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=1, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import AttentionArch  # noqa: E402
from tokenspeed.runtime.layers.attention import backends  # noqa: E402
from tokenspeed.runtime.layers.attention.configs import base as configs  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.registry import (  # noqa: E402
    _get_backend_cls,
    validate_attention_backend_name,
)


def _config(monkeypatch, device, backend_name, **overrides):
    # The NPU device spelling is registered only when torch_npu is installed.
    # Exercise the device-dependent config policy on CPU/CUDA CI as well.
    monkeypatch.setattr(
        configs,
        "torch",
        SimpleNamespace(
            device=lambda name: SimpleNamespace(type=name),
            bfloat16=torch.bfloat16,
            int8=torch.int8,
        ),
    )
    packed_width = overrides.pop("packed_width", 576)
    spec = DSAConfig(
        backend_name=backend_name,
        num_attention_heads=64,
        num_kv_heads=1,
        attn_tp_size=8,
        head_dim=192,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=packed_width,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=16,
        index_k_format="bf16",
    )
    fields = dict(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=128,
        context_len=4096,
        max_bs=4,
        kvp_size=8,
        kvp_rank=0,
        kvp_group=tuple(range(8)),
        components=(spec,),
    )
    fields.update(overrides)
    return configs.AttnConfig(**fields), spec


@pytest.mark.parametrize("name", [None, "dsa"])
def test_dsa_name_resolves_to_the_device_implementation(name):
    expected = (
        backends.ascend_dsa.AscendDSABackend
        if current_platform().is_npu
        else backends.dsa.DSABackend
    )
    validate_attention_backend_name(name, flag="--attention-backend")
    assert _get_backend_cls(name, AttentionArch.DSA) is expected


def test_model_specific_dsa_name_is_not_registered():
    with pytest.raises(ValueError, match="Unknown --attention-backend"):
        validate_attention_backend_name("longcat_dsa", flag="--attention-backend")


@pytest.mark.parametrize("name", [None, "dsa"])
@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_dsa_dcp_config_accepts_the_common_name(monkeypatch, device, name):
    config, _ = _config(monkeypatch, device, name)
    assert config.kvp_size == 8


def test_dsa_dcp_still_rejects_cpu(monkeypatch):
    with pytest.raises(ValueError, match="requires CUDA or Ascend NPU"):
        _config(monkeypatch, "cpu", "dsa")


def test_gpu_dsa_dcp_still_rejects_block_drafts(monkeypatch):
    with pytest.raises(ValueError, match="non-causal block drafts"):
        _config(monkeypatch, "cuda", "dsa", is_draft=True, draft_block_decode=True)


def test_ascend_draft_constraint_is_owned_by_the_backend(monkeypatch):
    config, spec = _config(monkeypatch, "npu", "dsa", is_draft=True)

    def initialize_base(self, config, spec, *, kernel_page_size):
        self.is_draft = config.is_draft

    monkeypatch.setattr(backends.dsa.DSABackend, "__init__", initialize_base)
    backend = _get_backend_cls("dsa", AttentionArch.DSA)
    # CUDA CI verifies the NPU guard directly, before native ops initialize.
    if not current_platform().is_npu:
        backend = backends.ascend_dsa.AscendDSABackend
    with pytest.raises(ValueError, match="draft attention is disabled"):
        backend(config, spec, kernel_page_size=128)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_ascend_bf16_kv_keeps_native_row_layout(monkeypatch):
    config, spec = _config(monkeypatch, "npu", "dsa")
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        _mla_layer_fields,
    )

    fields = _mla_layer_fields(config, 0, 0)
    assert fields[0].shape == (128, 1, 576)
    assert fields[0].dtype == "bfloat16"
    assert fields[0].payload_bytes == 128 * 576 * 2
    assert spec.cache_cell_size(config) == (576 + 128) * 2


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize("width", [576, 672])
def test_int8_cache_is_rejected_before_storage_planning(monkeypatch, device, width):
    with pytest.raises(ValueError, match="INT8 KV cache is not supported"):
        _config(
            monkeypatch, device, "dsa", kv_cache_dtype=torch.int8, packed_width=width
        )


def test_int8_cache_dtype_is_not_resolved():
    from tokenspeed.runtime.layers.attention.configs.base import resolve_dtype

    with pytest.raises(ValueError, match="Unsupported kv_cache_dtype"):
        resolve_dtype("int8")


def test_int8_cache_is_rejected_by_server_arguments():
    import argparse

    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--model", "x", "--kv-cache-dtype", "int8"])
    assert error.value.code == 2
