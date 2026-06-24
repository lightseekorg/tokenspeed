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

"""NPU graph wrapper and task-update support for Ascend NPU inference.

``NpuGraphWrapper`` extends ``CudaGraphWrapper`` by overriding four hook
methods for Ascend NPU-specific graph capture and replay:

1. ``_create_graph()`` → ``torch.npu.NPUGraph()`` instead of CUDAGraph
2. ``_graph_capture_kwargs()`` → omit ``stream`` (NPU uses its own)
3. ``_before_capture()`` → reset NPU task-update state
4. ``_before_graph_replay()`` → refresh FIA task groups with live seq_lens

``NpuGraphTaskParams`` stores the per-batch-size handles and FIA parameters
so that the wrapper can perform the update before replay, fixing the
frozen-dispatch problem where ``npu_fused_infer_attention_score`` bakes
``actual_seq_lengths_kv`` into the captured graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class NpuGraphTask:
    handle: object
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attn_output: torch.Tensor
    softmax_lse: torch.Tensor
    fia_kwargs: dict


class NpuGraphTaskParams:
    """Per-batch-size store for NPU graph task-update metadata.

    Each batch size captured in the NPU graph may contain multiple FIA
    calls (one per attention layer).  This object records the handle and
    parameters for each so they can be updated before replay.
    """

    def __init__(self) -> None:
        # { padded_bs: [list of per-layer task entries] }
        self.tasks: dict[int, list[NpuGraphTask]] = {}

    def record_decode_task(
        self,
        *,
        handle,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        softmax_lse: torch.Tensor,
        fia_kwargs: dict,
    ) -> None:
        """Record a single FIA task from the current capture."""
        num_tokens = query.shape[0]
        fia_kwargs = dict(fia_kwargs)
        self.tasks.setdefault(num_tokens, []).append(
            NpuGraphTask(
                handle=handle,
                query=query,
                key=key,
                value=value,
                attn_output=attn_output,
                softmax_lse=softmax_lse,
                fia_kwargs=fia_kwargs,
            )
        )

    def update_tasks(
        self,
        padded_bs: int,
        *,
        update_stream,
        seq_lens: torch.Tensor,
        seq_lens_list: list[int] | None = None,
    ) -> None:
        """Update all FIA task groups with current ``seq_lens`` before replay."""
        num_tokens = padded_bs
        tasks = self.tasks.get(num_tokens, [])
        if not tasks:
            raise RuntimeError(
                f"NPU graph task-update: no tasks found for num_tokens={num_tokens} "
                f"(available keys: {list(self.tasks.keys())}), seq_lens will be stale!"
            )

        actual_seq_lengths_kv = (
            [int(x) for x in seq_lens_list]
            if seq_lens_list is not None
            else seq_lens.tolist()
        )
        if len(actual_seq_lengths_kv) == 0:
            raise RuntimeError(
                f"NPU graph task-update: empty seq_lens for num_tokens={num_tokens}"
            )
        graph_task_update_begin = torch.npu.graph_task_update_begin
        graph_task_update_end = torch.npu.graph_task_update_end
        fia_out = torch_npu.npu_fused_infer_attention_score.out

        with torch.npu.stream(update_stream):
            for task in tasks:
                fia_kwargs = task.fia_kwargs
                fia_kwargs["actual_seq_lengths_kv"] = actual_seq_lengths_kv

                graph_task_update_begin(update_stream, task.handle)
                fia_out(
                    query=task.query,
                    key=task.key,
                    value=task.value,
                    out=[task.attn_output, task.softmax_lse],
                    **fia_kwargs,
                )
                graph_task_update_end(update_stream)


# Global singleton – one per process.
_npu_graph_task_params: NpuGraphTaskParams | None = None


def get_npu_graph_task_params() -> NpuGraphTaskParams:
    """Return (creating if needed) the global ``NpuGraphTaskParams``."""
    global _npu_graph_task_params
    if _npu_graph_task_params is None:
        _npu_graph_task_params = NpuGraphTaskParams()
    return _npu_graph_task_params


def reset_npu_graph_task_params() -> None:
    """Reset the global ``NpuGraphTaskParams`` (call between captures)."""
    global _npu_graph_task_params
    _npu_graph_task_params = None


class NpuGraphWrapper(CudaGraphWrapper):
    """Ascend NPU graph wrapper that extends CudaGraphWrapper.

    Overrides four hook methods for NPU-specific graph capture and replay
    while inheriting all the common machinery (metadata init, padding,
    DeepEP adapter, grammar support, etc.).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.npu_update_stream = torch.npu.Stream()
        # Override: NPU graph capture does NOT use a separate CUDA-style
        # stream.  torch.npu internally creates its own side stream.
        self.stream = None

    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _graph_capture_kwargs(self, pool):
        # NPU: omit stream — torch.npu.graph uses its own internal side
        # stream (matching vllm-ascend's working pattern).
        return dict(pool=pool)

    def _before_capture(self):
        reset_npu_graph_task_params()

    def requires_cpu_seq_lens_for_graph_replay(self) -> bool:
        return True

    def _before_graph_replay(self, padded_bs: int, seq_lens: torch.Tensor):
        self._before_graph_replay_with_seq_lens(padded_bs, seq_lens, None)

    def _before_graph_replay_with_seq_lens(
        self,
        padded_bs: int,
        seq_lens: torch.Tensor,
        seq_lens_list: list[int] | None = None,
    ):
        task_params = get_npu_graph_task_params()
        num_tokens = padded_bs * self.max_tokens_per_req
        task_params.update_tasks(
            num_tokens,
            update_stream=self.npu_update_stream,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens_list,
        )
        self.npu_update_stream.synchronize()
