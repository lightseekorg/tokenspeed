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

"""Optional ATen-compatible CUDA scoring library, built before serving."""

import functools
import hashlib
import json
from pathlib import Path

import torch


def _library() -> Path:
    return (
        Path(__file__).parent / "objs" / "full_vocab_scoring" / "full_vocab_scoring.so"
    )


def _identity() -> dict:
    directory = Path(__file__).parent / "csrc"
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sources": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("full_vocab_scoring.cu", "full_vocab_scoring_topk.cuh")
        },
    }


def available() -> bool:
    """Select only the validated Torch version and matching prebuilt sources."""
    if torch.__version__.split("+")[0] != "2.9.0" or not _library().is_file():
        return False
    identity = _library().with_suffix(".json")
    if not identity.is_file():
        return False
    return json.loads(identity.read_text()) == _identity()


@functools.cache
def _load():
    if not available():
        raise RuntimeError(
            "Build the optional Torch 2.9.0 full-vocabulary library before serving"
        )
    torch.ops.load_library(str(_library()))
    return torch.ops.tokenspeed_full_vocab


def topk(logits, active, k):
    return _load().topk(logits, active, k)


def maximum(logits, values):
    return _load().maximum(logits, values)


def tail(logits, ids, log_z, row_maximum, active):
    return _load().tail(logits, ids, log_z, row_maximum, active)


def build() -> None:
    """Build the optional library for the installed Torch ABI, outside forward."""
    if torch.__version__.split("+")[0] != "2.9.0":
        raise RuntimeError("This exact selection implementation requires Torch 2.9.0")
    from torch.utils.cpp_extension import load

    directory = _library().parent
    directory.mkdir(parents=True, exist_ok=True)
    load(
        name="full_vocab_scoring",
        sources=[str(Path(__file__).parent / "csrc" / "full_vocab_scoring.cu")],
        build_directory=str(directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
            "-DCUB_WRAPPED_NAMESPACE=at_cuda_detail",
        ],
        is_python_module=False,
        verbose=True,
    )
    _library().with_suffix(".json").write_text(json.dumps(_identity(), indent=2))


if __name__ == "__main__":
    build()
