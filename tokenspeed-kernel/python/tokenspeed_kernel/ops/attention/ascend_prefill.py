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

"""Ascend's packed Prefill layout; runtime's boundary to the NPU adapter."""

from dataclasses import dataclass

import torch


@dataclass
class PrefillWorkspace:
    layer_id: int
    batch_key: object
    source_kv: torch.Tensor
    source_index: torch.Tensor | None
    kv: torch.Tensor
    index: torch.Tensor | None
    history_rows: int
    send_rows: int
    current_rows: int
    ranks: tuple[int, ...]
    rank: int
    group: torch.distributed.ProcessGroup | None
    write_slots: torch.Tensor | None
    peak_buffer_bytes: int

    @property
    def workspace_bytes(self):
        return sum(
            x.numel() * x.element_size() for x in (self.kv, self.index) if x is not None
        )


def ascend_prefill_kernels():
    # This fixed packed layout belongs to the Ascend backend, not a portable op.
    from tokenspeed_kernel_npu.ops import prefill

    return prefill


DSA_PREFILL_SCHEMAS = {
    "mlp_lightning_indexer": "custom::mlp_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, Tensor? cur_seq_lengths_query=None, Tensor? cur_seq_lengths_key=None, Tensor? block_table=None, Tensor? init_tensor=None, Tensor? local_tensor=None, str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, int kv_block_len=1, int q_block_len=1, int init_num=0, int local_num=0, int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, bool return_value=False) -> (Tensor, Tensor)",
    "npu_sparse_mla_attention_prefill": "custom::npu_sparse_mla_attention_prefill(Tensor query, Tensor gathered_kv_pages, Tensor logical_sparse_rows, Tensor scan_count, Tensor page_table, Tensor query_seq, float scale_value, bool head_major_output=False) -> Tensor",
}


def require_dsa_prefill_ops(custom_ops):
    schemas = {}
    for name, expected in DSA_PREFILL_SCHEMAS.items():
        try:
            op = getattr(custom_ops, name).default
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                f"flash_ops is missing Prefill operator {name}"
            ) from error
        actual = str(op._schema)
        if actual != str(torch._C.parse_schema(expected)):
            raise RuntimeError(
                f"Incompatible Prefill operator {name}: {actual}; expected {expected}"
            )
        for key in ("PrivateUse1", "Meta"):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key):
                raise RuntimeError(
                    f"Prefill operator {name} is missing its {key} implementation"
                )
        schemas[name] = actual
    return schemas
