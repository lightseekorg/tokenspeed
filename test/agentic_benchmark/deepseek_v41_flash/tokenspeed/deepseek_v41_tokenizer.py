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

"""Checkpoint text encoder used to construct SWE-smith inputs."""

import importlib.util
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_encoder(path: Path):
    """Load the checkpoint encoder once per worker without pickling its functions."""
    spec = importlib.util.spec_from_file_location("deepseek_v41_dataset_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V4.1 encoding from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


class DatasetTokenizer:
    """The three tokenizer operations used by the pinned dataset builder."""

    def __init__(self, tokenizer, encoding_path: Path):
        self.tokenizer = tokenizer
        self.encoding_path = encoding_path

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, tools=None
    ):
        if tools is not None or not tokenize or not add_generation_prompt:
            raise ValueError(
                "Dataset construction requires a tokenized text chat prompt without tools"
            )
        # Match the V4.1 chat endpoint's default thinking mode.
        prompt = load_encoder(self.encoding_path)(messages, thinking_mode="thinking")
        return self.tokenizer.encode(prompt, add_special_tokens=False)


def load_tokenizer(model: str) -> DatasetTokenizer:
    """Load only tokenizer assets and the V4.1 checkpoint's standalone encoder."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = Path(model)
    if not path.is_dir():
        path = Path(
            snapshot_download(
                repo_id=model,
                allow_patterns=[
                    "tokenizer*",
                    "special_tokens_map.json",
                    "encoding/*.py",
                ],
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, clean_up_tokenization_spaces=False
    )
    # Keep the .py snapshot symlink; the Hub blob target has no file extension.
    return DatasetTokenizer(tokenizer, (path / "encoding/encoding.py").absolute())
