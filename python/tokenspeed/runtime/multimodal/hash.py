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

"""Content hashing for multimodal features.

A multimodal feature -- an image/video pixel tensor, a numpy array, or a nested
list of them -- is folded into a single unsigned 64-bit integer. The runtime
uses that integer for within-batch dedup (duplicate features encode once) and
as the seed for a per-item pad value that substitutes the placeholder token ids
so the text-only prefix cache can prefix-match across requests. The hash only
needs to be deterministic and well distributed *within a run*: values are
computed once (in the gateway producer) and travel with the item, never
persisted or compared across builds, so the concrete digest is an
implementation detail.
"""

import hashlib
from typing import Iterable, Union

import numpy as np
import torch

from tokenspeed.runtime.utils import flatten_nested_list

# blake2b emits an 8-byte digest natively, which is exactly our key width.
_KEY_BYTES = 8

ByteChunk = Union[bytes, bytearray, memoryview]


def _fold(chunks: Iterable[ByteChunk]) -> int:
    """Fold an ordered sequence of byte chunks into one unsigned 64-bit key."""
    digest = hashlib.blake2b(digest_size=_KEY_BYTES)
    for chunk in chunks:
        digest.update(chunk)
    return int.from_bytes(digest.digest(), byteorder="big")


def _raw_bytes(buffer: Union[torch.Tensor, np.ndarray]) -> memoryview:
    """Contiguous byte view of a tensor/array; CUDA tensors are pulled to host."""
    if isinstance(buffer, torch.Tensor):
        if buffer.is_cuda:
            buffer = buffer.cpu()
        return memoryview(buffer.detach().contiguous().view(torch.uint8).numpy())
    return memoryview(np.ascontiguousarray(buffer))


def hash_feature(feature) -> int:
    """Deterministic unsigned 64-bit content hash of a multimodal feature.

    Handles a single tensor or numpy array, a (possibly nested) list of those,
    and -- as a fallback -- any bytes-like or ``repr``-able object.
    """
    if isinstance(feature, (torch.Tensor, np.ndarray)):
        return _fold([_raw_bytes(feature)])

    if isinstance(feature, list):
        leaves = flatten_nested_list(feature)
        if leaves and all(isinstance(x, (torch.Tensor, np.ndarray)) for x in leaves):
            return _fold(_raw_bytes(x) for x in leaves)
        # Non-array leaves (e.g. python scalars): hash a stable serialization.
        return _fold([repr(tuple(leaves)).encode()])

    if isinstance(feature, (bytes, bytearray, memoryview)):
        return _fold([feature])

    return _fold([repr(feature).encode()])
