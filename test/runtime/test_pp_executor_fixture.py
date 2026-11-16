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


"""Small real executor owners; only the absent PP peer uses test transport."""

from dataclasses import dataclass
from test.runtime.cache_pool_test_utils import make_arena, make_mha_memory_plan

import torch
from torch import nn
from transformers import LlamaConfig

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.pp_stage import PPStageState
from tokenspeed.runtime.execution.model_executor import (
    ModelExecutor,
    ModelExecutorConfig,
)
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    CudaGraphSupport,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import ServerArgs


class DenseStage(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4, device=device)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.arange(128, device=device).reshape(32, 4)
            )

    def pp_stage_state_spec(self, num_tokens, device):
        return [("hidden_states", (num_tokens, 4), torch.float32)]


class DenseModel(nn.Module):
    def __init__(self, config, *, device, last_stage):
        super().__init__()
        self.model = DenseStage(device)
        self.lm_head = nn.Linear(4, 32, bias=False, device=device)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(128, device=device).reshape(32, 4))
        self.logits_processor = (
            LogitsProcessor(
                config,
                tp_rank=0,
                tp_size=1,
                tp_group=None,
                dp_lm_head_tp=False,
                attn_tp_size=1,
            )
            if last_stage
            else None
        )

    def forward(
        self, ctx, input_ids, positions, *, pp_inbound=None, multimodal_context=None
    ):
        if multimodal_context is not None:
            raise NotImplementedError("Dense test model is text-only")
        hidden = (
            self.model.embed_tokens(input_ids.to(torch.int64))
            if pp_inbound is None
            else pp_inbound.hidden_states
        )
        if self.logits_processor is None:
            return PPStageState(hidden_states=hidden)
        logits = self.lm_head(hidden)
        return LogitsProcessorOutput(
            next_token_logits=logits, next_token_ids=logits.argmax(-1).to(torch.int32)
        )


@dataclass
class DenseRunner:
    model: DenseModel
    model_config: ModelConfig
    mapping: Mapping

    def forward(self, ctx, input_ids, positions, **kwargs):
        return self.model(ctx, input_ids, positions, **kwargs)


class DenseBackend(AttentionBackend):
    # The local dense test model has no attention kernel or graph work.
    cuda_graph_support = CudaGraphSupport(decode_graph=False, prefill_graph=False)

    def __init__(self):
        self._init_pool_binding()

    def init_cuda_graph_state(self, max_bs, **kwargs):
        pass


class PipelineTransport:
    def __init__(self):
        self.sent = []
        self.received = []

    def send(self, tensor, rank, group):
        self.sent.append((tensor.clone(), rank, group))

    def recv(self, shape, dtype, device, rank, group):
        self.received.append((rank, group))
        return torch.ones(shape, dtype=dtype, device=device)


def build_pp_executor(tmp_path, monkeypatch, *, device, pp_size, pp_rank):
    # Supply tiny local weights/config, not a replacement for executor.forward.
    hf = LlamaConfig(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
    )
    hf.architectures = ["LlamaForCausalLM"]
    hf.save_pretrained(tmp_path)
    args = ServerArgs(
        model=str(tmp_path),
        device=device,
        dtype="float32",
        trust_remote_code=False,
        max_model_len=16,
        world_size=pp_size,
        pipeline_parallel_size=pp_size,
        disaggregation_mode="prefill",
        max_num_seqs=2,
        chunked_prefill_size=4,
        enforce_eager=True,
        disable_prefill_graph=True,
        disable_cuda_graph_padding=True,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=2,
        input_logprob_chunk_tokens=4,
        grammar_backend="none",
    )
    for key in ("logprob_order", "numerics"):
        monkeypatch.setitem(global_server_args_dict, key, vars(args)[key])
    args.mapping.rank = pp_rank
    facts = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
    )
    model = DenseModel(hf, device=device, last_stage=pp_rank == pp_size - 1)
    plan = make_mha_memory_plan(
        size=16,
        prefix_granularity=4,
        layer_num=1,
        kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
    )
    arena = make_arena(plan, device)
    pool = MHATokenToKVPool(arena, torch.float32, 1, 4, 1, 0)
    sampler = GreedySamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order=args.logprob_order,
            max_bs=2,
            device=device,
            enable_tp_sync=False,
        )
    )
    config = ModelExecutorConfig.from_server_args(
        args,
        facts,
        model,
        max_req_pool_size=3,
        gpu_id=0,
        global_rank=pp_rank,
        prefix_granularity=4,
        overlap_schedule_depth=0,
    )
    return ModelExecutor(
        config, DenseRunner(model, facts, args.mapping), DenseBackend(), pool, sampler
    )
