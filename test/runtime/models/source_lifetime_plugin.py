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

"""A plugin whose cache metadata must outlive its checkpoint source."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from tokenspeed.runtime.configs.model_config import configure_mha_attention
from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.layers.attention.configs.base import resolve_dtype
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.models.llama import LlamaForCausalLM
from tokenspeed.runtime.plugins.registry import (
    register_model,
    register_post_model_init_hook,
)


def _metadata():
    return Path(os.environ["SOURCE_TEST_METADATA"])


def initialized(context):
    source = Path(context.model_config.model_path)
    metadata = json.loads((source / "config.json").read_text())
    metadata["has_draft"] = context.draft_model_config is not None
    # Exclusive creation also rejects repeated hooks during cache rebuilds.
    with _metadata().open("x") as output:
        json.dump(metadata, output)
    if os.environ["SOURCE_TEST_MODE"] == "fail":
        raise RuntimeError("source lifetime hook refused startup")
    if os.environ["SOURCE_TEST_MODE"] == "retire":
        shutil.rmtree(source)


def create_attention_config(server_args, model_config, is_draft):
    # This dependency must be ready before the first cache memory profile.
    metadata = json.loads(_metadata().read_text())
    return replace(
        MHAConfig.generate(server_args, model_config, is_draft),
        kv_cache_dtype=resolve_dtype(metadata["source_cache_dtype"]),
    )


class SourceLifetimeLlama(LlamaForCausalLM):
    @classmethod
    def model_profile(cls, hf_config):
        return ModelProfile(
            configure_attention=configure_mha_attention,
            create_attention_config=create_attention_config,
            cache_family="mha",
            linear_attention=None,
            default_attention_backend=None,
            request_token_history=False,
            is_multimodal=False,
            tokenizer_kwargs={},
            attention_instances_per_layer=1,
            numerics_envelopes=frozenset({"auto"}),
        )


def register():
    register_model(
        SourceLifetimeLlama, architectures=("LlamaForCausalLM",), override=True
    )
    register_post_model_init_hook("source_lifetime_fixture", initialized)
