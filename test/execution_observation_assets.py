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

"""Offline checkpoint generation and real observation-test configuration checks."""

import argparse
import json
from pathlib import Path


def resolve_qualification(argv: list[str]) -> dict:
    """Run the real launch resolver and target/draft numerical profile gate.

    This checks configuration support, not numerical or execution qualification.
    It neither aliases policy names nor changes the arguments later served.
    """
    from tokenspeed.runtime.configs.numerics import (
        BITWISE_ENVELOPES,
        require_verified_numerics,
    )
    from tokenspeed.runtime.engine.rank_setup import load_model_config
    from tokenspeed.runtime.utils.hf_transformers_utils import resolve_architecture
    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser(allow_abbrev=False)
    ServerArgs.add_cli_args(parser)
    parser.set_defaults(numerics=None)
    parsed = parser.parse_args(argv)
    if parsed.numerics is None:
        raise ValueError("qualification must explicitly name its numerical envelope")
    target_path = parsed.model or parsed.model_path
    for label, value in (
        ("target", target_path),
        ("tokenizer", parsed.tokenizer or target_path),
        ("draft", parsed.speculative_draft_model_path),
    ):
        if (label != "draft" or value is not None) and (
            value is None or not Path(value).is_dir()
        ):
            raise ValueError(
                f"observation qualification requires a local {label} directory"
            )
    resolved = ServerArgs.from_cli_args(parsed)
    resolved.mapping.rank = 0
    target = load_model_config(resolved, resolved.model, draft_of=None)
    configs = [target]
    if resolved.speculative_draft_model_path is not None:
        configs.append(
            load_model_config(
                resolved, resolved.speculative_draft_model_path, draft_of=target
            )
        )
    architectures = []
    for config in configs:
        architecture = resolve_architecture(config.hf_config)
        require_verified_numerics(
            resolved.numerics,
            model_profile=config.model_profile,
            architecture=architecture,
            quantization=config.quantization,
            vocab_size=config.vocab_size,
        )
        architectures.append(architecture)
    return {
        "numerics": resolved.numerics,
        "bitwise_contract": resolved.numerics in BITWISE_ENVELOPES,
        "architectures": architectures,
        "logprob_order": resolved.logprob_order,
        "sampling_stream": resolved.sampling_stream,
    }


def create_tiny_llama(directory: Path) -> list[str]:
    """Save deterministic CPU weights and a tokenizer for W1/DUMMY W8 routing.

    This synthetic dense target has no draft checkpoint or bitwise profile. Its
    auto-mode runs do not substitute for a qualified deployment's numerical gate.
    """
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    directory.mkdir(parents=True, exist_ok=False)
    config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=256,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(731)
        model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
        model.save_pretrained(directory, safe_serialization=True)
    vocab = {name: i for i, name in enumerate(("<pad>", "<s>", "</s>", "<unk>"))}
    vocab.update({f"token{i}": i for i in range(4, config.vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=config.max_position_embeddings,
    ).save_pretrained(directory)
    return [
        "--model",
        str(directory.resolve()),
        "--numerics",
        "auto",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "mha",
        "--sampling-backend",
        "greedy",
        "--seed",
        "731",
        "--grammar-backend",
        "none",
        "--gpu-memory-utilization",
        "0.5",
        "--max-total-tokens",
        "4096",
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-tiny-llama", type=Path)
    mode.add_argument("--check-args", type=Path, help="JSON file of full engine argv")
    options = parser.parse_args()
    if options.create_tiny_llama is not None:
        print(json.dumps(create_tiny_llama(options.create_tiny_llama)))
    else:
        print(
            json.dumps(
                resolve_qualification(json.loads(options.check_args.read_text()))
            )
        )
