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

"""On-disk cache for LoRA Triton autotune picks.

Triton's ``@triton.autotune`` caches the best config per ``key`` tuple in
``Autotuner.cache``, but only for the current process — every fresh Python
process re-runs the sweep on the first call to each unique shape.  This
module persists that cache as JSON next to the kernels so the picks
survive process restarts and ship in the repo.

Layout: ``configs/<gpu_label>/<kernel_name>.json``.  When a kernel runs
for the first time on a shape that has no saved entry, Triton falls back
to the candidate-config sweep (slow) and the result can be saved by a
follow-up call to :func:`save_kernel_cache`.

Config JSON format::

    {
      "(N, K, 'torch.bfloat16')": {
        "kwargs": {"BLOCK_S": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        "num_warps": 4,
        "num_stages": 3,
        "num_ctas": 1,
        "maxnreg": null
      },
      ...
    }
"""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from tokenspeed_kernel._triton import triton

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"


def _gpu_label() -> str:
    """Compact identifier for the active GPU — partitions config files."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    # Strip vendor prefix and whitespace: "NVIDIA H100 80GB HBM3" → "H100_80GB_HBM3".
    name = name.replace("NVIDIA ", "").strip()
    return name.replace(" ", "_")


def _config_path(kernel_name: str) -> Path:
    return CONFIG_DIR / _gpu_label() / f"{kernel_name}.json"


def _key_to_str(key: tuple) -> str:
    # ``repr(tuple)`` round-trips through ``ast.literal_eval`` provided the
    # tuple only holds primitives and str dtypes — which it does here.
    return repr(tuple(key))


def _str_to_key(s: str) -> tuple:
    return tuple(ast.literal_eval(s))


def _config_to_dict(cfg: triton.Config) -> dict:
    return {
        "kwargs": dict(cfg.kwargs),
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
        "num_ctas": cfg.num_ctas,
        "maxnreg": cfg.maxnreg,
    }


def _dict_to_config(d: dict) -> triton.Config:
    return triton.Config(
        d["kwargs"],
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
        num_ctas=d.get("num_ctas", 1),
        maxnreg=d.get("maxnreg"),
    )


def load_kernel_cache(kernel) -> int:
    """Populate ``kernel.cache`` from the on-disk JSON for the active GPU.

    ``kernel`` is the ``Autotuner`` wrapper produced by
    ``@triton.autotune``.  Returns the number of entries loaded (0 when
    no config file exists for this GPU, which is the normal first-run
    case).
    """
    name = kernel.base_fn.__name__
    path = _config_path(name)
    if not path.exists():
        logger.debug("no autotune cache for %s at %s", name, path)
        return 0
    with open(path) as f:
        raw = json.load(f)
    loaded = 0
    for k, v in raw.items():
        kernel.cache[_str_to_key(k)] = _dict_to_config(v)
        loaded += 1
    logger.info("loaded %d autotune picks for %s from %s", loaded, name, path)
    return loaded


def save_kernel_cache(kernel) -> Path:
    """Dump ``kernel.cache`` to JSON next to the kernel module."""
    name = kernel.base_fn.__name__
    path = _config_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {}
    for key, cfg in kernel.cache.items():
        blob[_key_to_str(key)] = _config_to_dict(cfg)
    with open(path, "w") as f:
        json.dump(blob, f, indent=2, sort_keys=True)
    logger.info("saved %d autotune picks for %s to %s", len(blob), name, path)
    return path
