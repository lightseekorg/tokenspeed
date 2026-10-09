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
"""An item's encoding leaves the GPU once its last encoder token is prefilled."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.multimodal.embedder import (  # noqa: E402
    EncoderSpec,
    MultimodalEmbedder,
)
from tokenspeed.runtime.multimodal.inputs import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)

DIM = 4
PLACEHOLDER = 1000

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _encoders(calls: list[int], deepstack: bool) -> dict[Modality, EncoderSpec]:
    """Row r of item h encodes to [h, r, h, r], then its negation for deepstack."""

    def fn(items: list[MultimodalDataItem]) -> torch.Tensor:
        calls.append(len(items))
        rows = []
        for item in items:
            n = sum(end - start + 1 for start, end in item.offsets)
            r = torch.arange(n, dtype=torch.float32, device="cuda")
            h = torch.full_like(r, float(item.hash))
            main = torch.stack([h, r, h, r], dim=1)
            rows.append(torch.cat([main, -main], dim=1) if deepstack else main)
        return torch.cat(rows)

    return {Modality.IMAGE: EncoderSpec(fn=fn, deepstack=deepstack)}


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        deepstack_visual_indexes=[0],
        separate_deepstack_embeds=lambda emb: (emb[:, :DIM], emb[:, DIM:]),
    )


def _ctx(
    items_per_request: list[list[MultimodalDataItem]],
    prefixes: list[int],
    seqs: list[int],
    max_encoder_tokens: int = 8192,
) -> MultimodalForwardContext:
    return MultimodalForwardContext(
        mm_inputs=[MultimodalInputs(mm_items=items) for items in items_per_request],
        extend_prefix_lens=prefixes,
        extend_seq_lens=seqs,
        max_encoder_tokens=max_encoder_tokens,
    )


def _apply(
    embedder: MultimodalEmbedder,
    items_per_request: list[list[MultimodalDataItem]],
    prefixes: list[int],
    seqs: list[int],
    ids: list[int],
    calls: list[int],
    deepstack: bool = False,
    max_encoder_tokens: int = 8192,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    text = nn.Embedding(8, DIM).cuda()
    out = embedder.apply(
        torch.tensor(ids, device="cuda"),
        text,
        _ctx(items_per_request, prefixes, seqs, max_encoder_tokens),
        _encoders(calls, deepstack),
        _model(),
    )
    torch.cuda.synchronize()
    return out


def _image(h: int, start: int, end: int) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=h,
        offsets=[(start, end)],
        feature=torch.tensor([h]),
    )


def _device_bytes(item: MultimodalDataItem) -> int:
    tensors = [item.encoded, item.encoded_deepstack]
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def test_encoder_calls_pack_items_in_order_under_the_bound() -> None:
    def rows(item: MultimodalDataItem, first: int, last: int) -> torch.Tensor:
        r = torch.arange(first, last + 1, dtype=torch.float32)
        h = torch.full_like(r, float(item.hash))
        return torch.stack([h, r, h, r], dim=1)

    # Prefix hits end inside both images: their 8 encoded rows exceed a bound of 4.
    calls: list[int] = []
    images = [_image(5, 0, 3), _image(6, 0, 3)]
    embeds, _ = _apply(
        MultimodalEmbedder(),
        [[images[0]], [images[1]]],
        [2, 2],
        [2, 2],
        [PLACEHOLDER] * 4,
        calls,
        max_encoder_tokens=4,
    )
    assert calls == [1, 1]
    torch.testing.assert_close(
        embeds.cpu(), torch.cat([rows(images[0], 2, 3), rows(images[1], 2, 3)])
    )

    # The bound, not the forward's own 4 tokens, decides: 8 rows fit one call.
    calls = []
    _apply(
        MultimodalEmbedder(),
        [[_image(7, 0, 3)], [_image(8, 0, 3)]],
        [2, 2],
        [2, 2],
        [PLACEHOLDER] * 4,
        calls,
        max_encoder_tokens=8,
    )
    assert calls == [2]

    # In order: an item larger than the bound runs alone, unequal items share a call.
    calls = []
    images = [_image(10, 0, 1), _image(9, 2, 7), _image(11, 8, 8), _image(12, 9, 11)]
    embeds, _ = _apply(
        MultimodalEmbedder(),
        [images],
        [0],
        [12],
        [PLACEHOLDER] * 12,
        calls,
        max_encoder_tokens=4,
    )
    assert calls == [1, 1, 2]
    torch.testing.assert_close(
        embeds.cpu(),
        torch.cat([rows(image, 0, n - 1) for image, n in zip(images, [2, 6, 1, 3])]),
    )


@pytest.mark.parametrize("deepstack", [False, True])
def test_encoding_moves_to_host_after_its_last_chunk_and_serves_recompute(
    deepstack: bool,
) -> None:
    """Tokens 2..7 are the image; chunks [0, 4) then [4, 10), then a recompute."""
    embedder, calls = MultimodalEmbedder(), []
    item = _image(5, 2, 7)
    ids = [1, 1] + [PLACEHOLDER] * 6 + [1, 1]

    first, _ = _apply(embedder, [[item]], [0], [4], ids[:4], calls, deepstack)
    assert item.encoded.is_cuda
    torch.testing.assert_close(first[2:4, 1].cpu(), torch.tensor([0.0, 1.0]))

    second, _ = _apply(embedder, [[item]], [4], [6], ids[4:], calls, deepstack)
    assert item.encoded.device.type == "cpu" and item.encoded.is_pinned()
    assert (item.encoded_deepstack is not None) == deepstack
    if deepstack:
        assert item.encoded_deepstack.device.type == "cpu"
        assert item.encoded_deepstack.is_pinned()
    torch.testing.assert_close(second[0:4, 1].cpu(), torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert calls == [1]

    redo, kwargs = _apply(embedder, [[item]], [0], [10], ids, calls, deepstack)
    rows = torch.arange(6, dtype=torch.float32)
    torch.testing.assert_close(redo[2:8, 1].cpu(), rows)
    torch.testing.assert_close(redo[2:8, 0].cpu(), torch.full((6,), 5.0))
    if deepstack:
        deep = kwargs["input_deepstack_embeds"]
        torch.testing.assert_close(deep[2:8, 1].cpu(), -rows)
    assert calls == [1]


def test_an_alias_still_in_prefill_keeps_its_device_encoding() -> None:
    """Two requests share one image: one finishes it now, the other does not."""
    embedder, calls = MultimodalEmbedder(), []
    done, pending = _image(9, 0, 3), _image(9, 0, 3)

    _apply(
        embedder,
        [[done], [pending]],
        [0, 0],
        [4, 2],
        [PLACEHOLDER] * 6,
        calls,
    )
    assert calls == [1]
    assert done.encoded.device.type == "cpu"
    assert pending.encoded.is_cuda
    torch.testing.assert_close(done.encoded, pending.encoded.cpu())


def test_aliases_finishing_together_share_one_host_copy() -> None:
    embedder, calls = MultimodalEmbedder(), []
    first, second = _image(9, 0, 3), _image(9, 0, 3)

    _apply(embedder, [[first], [second]], [0, 0], [4, 4], [PLACEHOLDER] * 8, calls)
    assert calls == [1]
    assert first.encoded.device.type == "cpu"
    assert second.encoded is first.encoded


def test_prefix_hit_past_the_image_still_offloads_a_published_encoding() -> None:
    """EPD publishes the encoding before scheduling; a prefix hit covers the image."""
    embedder, calls = MultimodalEmbedder(), []
    item = _image(3, 0, 3)
    item.encoded = torch.full((4, DIM), 3.0, device="cuda")
    ids = [PLACEHOLDER] * 4 + [1, 1]

    embeds, _ = _apply(embedder, [[item]], [4], [2], ids[4:], calls)
    assert embeds is None
    assert item.encoded.device.type == "cpu" and item.encoded.is_pinned()

    redo, _ = _apply(embedder, [[item]], [0], [6], ids, calls)
    torch.testing.assert_close(redo[0:4].cpu(), torch.full((4, DIM), 3.0))
    assert calls == []


@pytest.mark.parametrize("deepstack", [False, True])
def test_jointly_encoded_items_free_their_device_memory_once_all_are_prefilled(
    deepstack: bool,
) -> None:
    """Both images encode in one call and share its output until both are prefilled."""
    embedder, calls = MultimodalEmbedder(), []
    short, long = _image(1, 0, 255), _image(2, 0, 511)
    chunk = [PLACEHOLDER] * 128

    _apply(embedder, [[short], [long]], [0, 0], [128, 128], chunk * 2, calls, deepstack)
    assert calls == [2]
    both_on_device = torch.cuda.memory_allocated()
    encoded_bytes = _device_bytes(short) + _device_bytes(long)

    _apply(
        embedder, [[short], [long]], [128, 128], [128, 128], chunk * 2, calls, deepstack
    )
    assert short.encoded.device.type == "cpu"
    assert long.encoded.is_cuda

    _apply(embedder, [[long]], [256], [256], chunk * 2, calls, deepstack)
    assert long.encoded.device.type == "cpu"
    assert both_on_device - torch.cuda.memory_allocated() >= encoded_bytes
    assert calls == [2]


def test_encoding_published_on_another_stream_outlives_this_streams_reads() -> None:
    """EPD allocates on its own stream; the offload must not recycle the block early."""
    embedder, calls = MultimodalEmbedder(), []
    item = _image(7, 0, 3)
    publish, forward = torch.cuda.Stream(), torch.cuda.Stream()
    for stream in (publish, forward):
        with torch.cuda.stream(stream):
            # Cache a segment per stream: the timed section must not call the driver.
            torch.empty(1024, device="cuda")
    torch.empty((4, DIM), pin_memory=True)
    with torch.cuda.stream(publish):
        item.encoded = torch.full((4, DIM), 7.0, device="cuda")
    text = nn.Embedding(8, DIM).cuda()
    ids = torch.tensor([PLACEHOLDER] * 4, device="cuda")
    forward.wait_stream(publish)

    with torch.cuda.stream(forward):
        # Holds the scatter and the host copy back while ``publish`` runs ahead.
        torch.cuda._sleep(200_000_000)
        embeds, _ = embedder.apply(
            ids, text, _ctx([[item]], [0], [4]), _encoders(calls, False), _model()
        )
    assert item.encoded.device.type == "cpu"
    with torch.cuda.stream(publish):
        # Without a stream record this reuses the block the offload just freed.
        _clobber = torch.full((4, DIM), -1.0, device="cuda")
    torch.cuda.synchronize()

    torch.testing.assert_close(embeds[:, 0].cpu(), torch.full((4,), 7.0))
    torch.testing.assert_close(item.encoded[:, 0], torch.full((4,), 7.0))
    assert calls == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
