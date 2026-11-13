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

"""Create deterministic Qwen3-MoE test weights on the CPU."""

import argparse
import json
from pathlib import Path


def make_fixture(output: Path):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig

    heads = 4
    head_dim = 128
    hidden = heads * head_dim
    dense_width = 2 * hidden
    expert_width = 256
    experts = 8
    vocab_size = 256
    output.mkdir()
    config = Qwen3MoeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=dense_width,
        num_hidden_layers=4,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=head_dim,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        torch_dtype="bfloat16",
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=experts,
        num_experts_per_tok=2,
        moe_intermediate_size=expert_width,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    config.save_pretrained(output)
    (output / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}) + "\n"
    )
    vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocabulary.update({f"t{i}": i for i in range(4, vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    ).save_pretrained(output)
    generator = torch.Generator(device="cpu").manual_seed(17)

    def weight(rows, cols):
        return (
            torch.randn(rows, cols, generator=generator).mul_(0.02).to(torch.bfloat16)
        )

    def norm(width):
        return torch.ones(width, dtype=torch.bfloat16)

    weights = {
        "model.embed_tokens.weight": weight(vocab_size, hidden),
        "model.norm.weight": norm(hidden),
        "lm_head.weight": weight(vocab_size, hidden),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = norm(hidden)
        weights[f"{prefix}.post_attention_layernorm.weight"] = norm(hidden)
        for name in ("q_norm", "k_norm"):
            weights[f"{prefix}.self_attn.{name}.weight"] = norm(head_dim)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{prefix}.self_attn.{name}.weight"] = weight(hidden, hidden)
        weights[f"{prefix}.mlp.gate.weight"] = weight(experts, hidden)
        for expert in range(experts):
            mlp = f"{prefix}.mlp.experts.{expert}"
            weights[f"{mlp}.gate_proj.weight"] = weight(expert_width, hidden)
            weights[f"{mlp}.up_proj.weight"] = weight(expert_width, hidden)
            weights[f"{mlp}.down_proj.weight"] = weight(hidden, expert_width)
    save_file(weights, output / "model.safetensors", metadata={"format": "pt"})
    print(
        f"{len(weights)} tensors, {sum(t.numel() for t in weights.values())} elements"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    make_fixture(arguments.output)
