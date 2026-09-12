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

"""Load the optional metadata extension only with its matching build ABI."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_extension():
    directory = Path(__file__).resolve().parent / "objs/sampling_metadata"
    path = directory / "_sampling_metadata.so"
    try:
        record = json.loads((directory / "build.json").read_text())
        if (
            not isinstance(record, dict)
            or record.get("torch") != torch.__version__
            or record.get("python") != list(sys.version_info[:2])
            or record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            return None
        spec = importlib.util.spec_from_file_location("_sampling_metadata", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (OSError, ImportError, ValueError):
        # CPU/ROCm source installations and incompatible optional binaries
        # retain the caller's original bias-plus-argmax implementation.
        return None


def _unavailable_metadata(
    logits, bias, out, values, indices, current_device, bound_device, global_offset
):
    return None


_module = _load_extension()
if _module is None:
    validate_metadata = _unavailable_metadata
    NativeLaunchPair = None
else:
    _validator = _module.MetadataValidator()
    validate_metadata = _validator.validate_metadata
    NativeLaunchPair = _module.NativeLaunchPair
