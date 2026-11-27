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


"""A loaded engine keeps serving after a plugin retires its checkpoint source."""

import json
import os
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import (
    PauseSchedulerReqInput,
    ResumeSchedulerReqInput,
    UpdateWeightFromDiskReqInput,
)
from tokenspeed.runtime.entrypoints.wire_driver import WireDriver, generate_request
from tokenspeed.runtime.utils.network import get_free_port

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _checkpoint(directory):
    torch.manual_seed(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=128,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        )
    )
    model.save_pretrained(directory)
    tokenizer = Tokenizer(
        models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", eos_token="t1", pad_token="t2"
    ).save_pretrained(directory)


def _draft_checkpoint(directory):
    directory.mkdir()
    LlamaConfig(
        architectures=["LlamaForCausalLMEagle3"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        eagle_aux_hidden_state_layer_ids=[0],
    ).save_pretrained(directory)
    generator = torch.Generator().manual_seed(23)
    shapes = {
        "fc.weight": (128, 128),
        "midlayer.self_attn.q_proj.weight": (128, 256),
        "midlayer.self_attn.k_proj.weight": (128, 256),
        "midlayer.self_attn.v_proj.weight": (128, 256),
        "midlayer.self_attn.o_proj.weight": (128, 128),
        "midlayer.mlp.gate_proj.weight": (256, 128),
        "midlayer.mlp.up_proj.weight": (256, 128),
        "midlayer.mlp.down_proj.weight": (128, 256),
    }
    weights = {
        name: torch.randn(shape, generator=generator) * 0.02
        for name, shape in shapes.items()
    }
    for name in (
        "norm",
        "midlayer.input_layernorm",
        "midlayer.hidden_norm",
        "midlayer.post_attention_layernorm",
    ):
        weights[f"{name}.weight"] = torch.ones(128)
    save_file(weights, str(directory / "model.safetensors"))


def _source_plugin(directory):
    directory.mkdir()
    (directory / "source_lifetime_fixture.py").write_text("""
import json
import os
import shutil
from importlib.metadata import entry_points
from pathlib import Path
from tokenspeed.runtime.plugins.registry import register_post_model_init_hook

def initialized(context):
    with Path(os.environ["SOURCE_TEST_MARKER"]).open("a") as marker:
        marker.write(json.dumps({
            "model": context.model_config.model_path,
            "draft": context.draft_model_config is not None,
        }) + "\\n")
    if os.environ["SOURCE_TEST_RETIRE"] == "1":
        shutil.rmtree(context.model_config.model_path)

def register():
    register_post_model_init_hook("source_lifetime_fixture", initialized)
""")
    metadata = directory / "source_lifetime_fixture-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: source-lifetime-fixture\nVersion: 1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[tokenspeed.plugins]\nsource-lifetime-fixture = source_lifetime_fixture:register\n"
    )


@pytest.mark.parametrize("with_draft", [False, True])
def test_post_model_init_source_lifetime(tmp_path, with_draft):
    original = tmp_path / "original"
    _checkpoint(original)
    if with_draft:
        _draft_checkpoint(original / "draft")
    fixture = tmp_path / "fixture"
    _source_plugin(fixture)
    bad = tmp_path / "bad"
    bad.mkdir()
    tensors = load_file(original / "model.safetensors")
    tensors["model.embed_tokens.weight"] = torch.zeros(32, 128)
    save_file(tensors, str(bad / "model.safetensors"))
    outputs = []
    for retire in (False, True):
        source = tmp_path / f"source-{int(retire)}"
        shutil.copytree(original, source)
        marker = tmp_path / f"initialized-{int(retire)}.jsonl"
        args = [
            "--model",
            str(source),
            "--dist-init-addr",
            f"127.0.0.1:{get_free_port()}",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "128",
            "--max-num-seqs",
            "4",
            "--max-total-tokens",
            "256",
            "--chunked-prefill-size",
            "64",
            "--gpu-memory-utilization",
            "0.2",
            "--sampling-backend",
            "greedy",
            "--enable-output-logprobs",
            "--disable-kvstore",
            "--retraction-snapshot-ratio",
            "0.5",
        ]
        if with_draft:
            args += [
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                str(source / "draft"),
                "--speculative-num-steps",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--enforce-eager",
            ]
        else:
            args += ["--cudagraph-capture-sizes", "1", "2"]
        driver = WireDriver(
            args,
            num_engines=1,
            routing="round_robin",
            startup_timeout_s=180,
            env={
                "PYTHONPATH": f"{fixture}:{os.environ.get('PYTHONPATH', '')}",
                "TOKENSPEED_DISABLE_PLUGINS": ",".join(
                    ep.name for ep in entry_points(group="tokenspeed.plugins")
                ),
                "SOURCE_TEST_MARKER": str(marker),
                "SOURCE_TEST_RETIRE": str(int(retire)),
            },
        )
        try:
            assert source.exists() == (not retire)
            calls = [json.loads(line) for line in marker.read_text().splitlines()]
            assert calls == [{"model": str(source), "draft": with_draft}]
            request = generate_request(
                [3, 4, 5],
                {"temperature": 0.0, "max_new_tokens": 4, "ignore_eos": True},
                return_logprob=True,
            )
            [result] = driver.generate([request])
            outputs.append((result.output_ids, result.output_token_logprobs))
            [paused] = driver.control(PauseSchedulerReqInput(mode="wait"))
            assert paused.success, paused.message
            [rejected] = driver.control(
                UpdateWeightFromDiskReqInput(model_path=str(bad))
            )
            assert not rejected.success and "has shape" in rejected.message
            [loaded] = driver.control(
                UpdateWeightFromDiskReqInput(model_path=str(original))
            )
            assert loaded.success, loaded.message
            [resumed] = driver.control(ResumeSchedulerReqInput())
            assert resumed.success, resumed.message
            [after] = driver.generate(
                [
                    generate_request(
                        [3, 4, 5],
                        {"temperature": 0.0, "max_new_tokens": 4, "ignore_eos": True},
                        return_logprob=True,
                    )
                ]
            )
            assert (after.output_ids, after.output_token_logprobs) == outputs[-1]
            assert len(marker.read_text().splitlines()) == 1
        finally:
            driver.shutdown()
    assert outputs[0] == outputs[1]
