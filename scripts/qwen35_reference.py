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

"""Compare captured inference against unmodified Transformers CPU FP32 text model."""

import argparse
import json
from pathlib import Path

import torch
from qwen35_tp_compare import compare_logits
from safetensors import safe_open
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM
from transformers.initialization import no_init_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--dtype", choices=["float16", "float32"], required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    config = Qwen3_5Config.from_pretrained(
        args.model, local_files_only=True
    ).text_config
    config._attn_implementation = "eager"
    with no_init_weights(), torch.device("cpu"):
        reference = Qwen3_5ForCausalLM(config)
    expected_names = set(reference.state_dict())
    weights = {}
    for path in sorted(args.model.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                mapped = name.replace("model.language_model.", "model.", 1)
                if mapped in expected_names:
                    weights[mapped] = checkpoint.get_tensor(name).to(
                        getattr(torch, args.dtype)
                    )
    if config.tie_word_embeddings:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    reference.load_state_dict(weights, strict=True, assign=True)
    del weights
    reference.eval()
    layer_values = {}

    def observe_layer(index):
        def hook(module, inputs, output):
            layer_values[str(index)] = (
                output.detach().float().reshape(-1, config.hidden_size)
            )

        return hook

    for index, layer in enumerate(reference.model.layers):
        layer.register_forward_hook(observe_layer(index))
    records = []
    with torch.inference_mode():
        for directory in sorted(args.captures.glob("length-*")):
            cache = None
            paths = sorted(directory.glob("[0-9][0-9][0-9][0-9].pt"))
            assert args.steps > 0 and len(paths) >= args.steps
            for path in paths[: args.steps]:
                sample = torch.load(path, weights_only=True)
                output = reference(
                    sample["input_ids"].reshape(1, -1),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
                record = compare_logits(sample["logits"], output.logits[:, -1])
                layer_path = path.with_name("layers-" + path.name)
                if layer_path.exists():
                    actual_layers = torch.load(layer_path, weights_only=True)
                    record["layer_errors"] = [
                        dict(
                            layer=index,
                            rms=(actual_layers[str(index)] - layer_values[str(index)])
                            .square()
                            .mean()
                            .sqrt()
                            .item(),
                            reference_rms=layer_values[str(index)]
                            .square()
                            .mean()
                            .sqrt()
                            .item(),
                        )
                        for index in range(config.num_hidden_layers)
                    ]
                record.update(request=directory.name, step=int(path.stem))
                records.append(record)
                print(json.dumps(record), flush=True)
    assert records, "No captures"
    result = dict(
        reference=f"Transformers 5.12.0 CPU {args.dtype} eager, original checkpoint",
        thresholds=dict(max_abs=0.75, rms=0.10, cosine=0.999),
        compared_steps=len(records),
        all_passed=all(r["passed"] for r in records),
        records=records,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["all_passed"], "Reference logits differ beyond predeclared gates"


if __name__ == "__main__":
    main()
