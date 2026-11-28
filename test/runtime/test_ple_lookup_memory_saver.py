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

"""GPU release/resume checks for the PLE lookup's FP8 table scale tensor."""

import subprocess
import sys
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch

register_cuda_ci(est_time=30, suite="runtime-1gpu")


def _ordinary_bytes() -> int:
    """Bytes held by ordinary Torch CUDA allocator segments on this device."""
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == torch.cuda.current_device()
        and tuple(segment["segment_pool_id"]) == (0, 0)
    )


def _check_fp8_lookup_scale_survives_release(enable: bool) -> None:
    """Release and resume a device FP8 PLE table and its scale tensor, with
    scale updates between the cycles (``enable=False`` runs the same steps
    without a memory saver)."""
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.ple_lookup import PLELookup
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    saver = TorchMemorySaverAdapter.create(enable=enable, device="cuda")
    vocab, heads, dim, tokens = 64, 2, 16, 5
    generator = torch.Generator().manual_seed(19)
    # The model is constructed in the weight region; the scale tensor is
    # allocated with the table.
    with saver.region(tag="weights", enable_cpu_backup=True), torch.device("cuda"):
        lookup = PLELookup(
            Mapping(rank=0, world_size=1),
            vocab_size=vocab,
            ngram_heads=heads,
            head_dim=dim,
            storage_dtype=torch.float8_e4m3fn,
            output_dtype=torch.bfloat16,
            offload=False,
            table_layout="global_sharded",
            enable_hot_cache=False,
            prefix="",
        )
    fp8_rows = (torch.randn(vocab, dim, generator=generator) * 4).to(
        torch.float8_e4m3fn
    )
    lookup.load_shard(fp8_rows, 0, 1, prescanned=False)
    lookup.load_scale(torch.tensor(0.125))
    with saver.named_region("executor_inputs"):
        ids = torch.randint(0, vocab, (tokens, heads), generator=generator).cuda()
    layout = lookup.make_layout(None, tokens)
    table = lookup.ngram_embedding.weight
    scale = lookup.device_table_scale
    pointers = (table.data_ptr(), scale.data_ptr())

    def lookup_rows() -> torch.Tensor:
        return lookup.finish(lookup.start(ids, layout)).cpu()

    def expected(table_scale: float) -> torch.Tensor:
        rows = table.detach().float().cpu()[ids.cpu()]
        return (rows * table_scale).bfloat16().flatten(1)

    online_source = (torch.randn(vocab, dim, generator=generator) * 3).bfloat16()
    online = (online_source.float().abs().max() / 448).bfloat16().item()
    # Each cycle changes the scale first: a checkpoint scale, then online
    # re-quantization, then another checkpoint scale.
    updates = (
        (lambda: lookup.load_scale(torch.tensor(0.0625)), 0.0625),
        (lambda: lookup.load_shard(online_source, 0, 1, prescanned=False), online),
        (lambda: lookup.load_scale(torch.tensor(0.25)), 0.25),
    )
    for update, table_scale in updates:
        update()
        reference = expected(table_scale)
        torch.testing.assert_close(lookup_rows(), reference, rtol=0, atol=0)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        saver.pause()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if enable:
            ordinary_bytes = _ordinary_bytes()
            assert ordinary_bytes == 0, ordinary_bytes
        saver.resume()
        assert (table.data_ptr(), scale.data_ptr()) == pointers
        assert lookup.device_table_scale is scale
        assert scale.item() == table_scale
        torch.testing.assert_close(lookup_rows(), reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fp8_lookup_scale_steps_without_memory_saver() -> None:
    """The release check's steps hold on any GPU, without the saver."""
    _check_fp8_lookup_scale_survives_release(False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
def test_fp8_lookup_scale_survives_release() -> None:
    memory_saver = pytest.importorskip("torch_memory_saver")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_ple_lookup_memory_saver import "
                "_check_fp8_lookup_scale_survives_release; "
                "_check_fp8_lookup_scale_survives_release(True)",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=600,
        )
