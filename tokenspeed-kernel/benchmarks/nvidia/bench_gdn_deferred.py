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

"""Continuous GDN steps using the checkpoint's real layer count and geometry.

Run on an idle GPU with --execution graph for serving-like launch overhead.
The state is updated in place at every step; sequence timing preserves cache
behavior between layers. Reported bandwidth counts recurrent-state bytes only.
"""

import argparse
import json
from pathlib import Path

import torch
from tokenspeed_kernel.ops.attention.gdn import gdn_decode_mtp
from tokenspeed_kernel.ops.kvcache.triton import copy_state_rows

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.gdn import GdnAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


def timed(fn, *, execution, repetitions):
    for _ in range(3):
        fn()
    operation = fn
    graph = None
    if execution == "graph":
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        operation = graph.replay
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repetitions):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


@torch.inference_mode()
def run(args):
    config = json.loads(args.model_config.read_text())
    config = config.get("text_config", config)
    layers = sum(kind == "linear_attention" for kind in config["layer_types"])
    heads, vheads = config["linear_num_key_heads"], config["linear_num_value_heads"]
    key_dim, value_dim = config["linear_key_head_dim"], config["linear_value_head_dim"]
    batch, tokens = args.batch, 4
    width = 2 * heads * key_dim + vheads * value_dim
    spec = MHAConfig(
        num_attention_heads=heads, num_kv_heads=heads, head_dim=key_dim, attn_tp_size=1
    )
    linear = LinearAttnConfig(
        num_k_heads=heads,
        num_v_heads=vheads,
        head_k_dim=key_dim,
        head_v_dim=value_dim,
        conv_kernel_size=4,
        layer_ids=tuple(range(layers)),
        tp_size=1,
        replay_ssm=True,
        draft_tree=False,
    )
    attn_config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=4096,
        max_bs=batch,
        max_resident_requests=batch,
        speculative_num_draft_tokens=tokens,
        components=(spec, linear),
    )
    fields = tuple(
        field
        for layer in range(layers)
        for field in (
            CacheFieldSpec(
                f"layer.{layer}.conv", f"conv.{layer}", (width, 3), "bfloat16"
            ),
            CacheFieldSpec(
                f"layer.{layer}.ssm",
                f"ssm.{layer}",
                (vheads, value_dim, key_dim),
                "bfloat16",
            ),
        )
    )
    group = CacheGroupSpec(group_id="gdn", attn_kind="state", replayable=False)
    arena = CacheArena(
        pack(((group, fields),), prefix_granularity=64).bind(batch + 2),
        "cuda",
        cache_group_specs=(group,),
    )
    pool = HybridMHATokenToKVPool(
        arena=arena,
        dtype=torch.bfloat16,
        head_num=heads,
        head_dim=key_dim,
        layer_num=layers,
        rank=0,
        layer_types=("linear_attention",) * layers,
    )
    torch.manual_seed(17)
    qkv = (
        torch.randn(layers, batch * tokens, width, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    a, b = (
        torch.randn(layers, batch * tokens, vheads, device="cuda", dtype=torch.bfloat16)
        * 0.1
        for _ in range(2)
    )
    conv = torch.randn(layers, width, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    parameters = torch.randn(layers, 2, vheads, device="cuda") * 0.1
    slots = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    tables = {"gdn": torch.stack((slots, slots), 1)}
    accepted = torch.full((batch,), 4, device="cuda", dtype=torch.int32)
    frontier = torch.zeros(batch + 2, device="cuda", dtype=torch.int32)
    stopped = torch.zeros(batch + 2, device="cuda", dtype=torch.bool)
    lengths = torch.full((batch,), 16, device="cuda", dtype=torch.int32)
    state_bytes = layers * batch * vheads * value_dim * key_dim * 2
    result = {
        "device": torch.cuda.get_device_name(),
        "layers": layers,
        "batch": batch,
        "verify_tokens": 4,
        "geometry": [heads, vheads, key_dim, value_dim],
        "execution": args.execution,
        "measurements": {},
    }
    for defer in (False, True):
        backend = GdnAttnBackend(
            attn_config,
            spec,
            writes_state_checkpoints_in_forward=False,
            defer_recurrent_state=defer,
        )
        backend.set_kv_pool(pool)
        backend.init_cuda_graph_state(batch)
        workspace_bytes = backend.preallocate_verify_workspace(batch, 4)

        def refresh():
            backend.refresh_decode_metadata(
                batch,
                batch,
                slots,
                lengths,
                forward_mode=ForwardMode.DECODE,
                for_graph_replay=args.execution == "graph",
                block_tables=tables,
            )

        def step():
            refresh()
            backend.prepare_pending_recurrent_state(slots, frontier, allow_defer=True)
            for layer in range(layers):
                backend.forward_decode(
                    None,
                    None,
                    None,
                    layer=None,
                    out_cache_loc=None,
                    token_to_kv_pool=pool,
                    bs=batch,
                    mixed_qkv=qkv[layer],
                    conv_weights=conv[layer],
                    bias=None,
                    activation="silu",
                    key_dim=heads * key_dim,
                    value_dim=vheads * value_dim,
                    attention_tp_size=1,
                    head_k_dim=key_dim,
                    head_v_dim=value_dim,
                    a=a[layer],
                    b=b[layer],
                    A_log=parameters[layer, 0],
                    dt_bias=parameters[layer, 1],
                    layer_id=layer,
                    seq_len=batch * tokens,
                )
            backend.prepare_recurrent_state_acceptance(
                stopped, frontier, allow_defer=True
            )
            backend.commit_verified_state(accepted, accepted_path=None)
            frontier.index_add_(0, slots.long(), accepted)

        ms = timed(step, execution=args.execution, repetitions=args.repetitions)
        measurements = {
            "step_ms": ms,
            "workspace_mib": workspace_bytes / 2**20,
            "sequence_state_gb_s": (2 if defer else 3) * state_bytes / (ms * 1e6),
        }
        if defer:
            pending = backend._pending_state
            if pending is None:
                raise RuntimeError("checkpoint geometry did not enable deferred WY")
            backend.materialize_recurrent_state(None)
            zero_accept = torch.zeros_like(accepted)
            live_pages = slots.expand(layers, -1).contiguous()

            def metadata():
                backend.prepare_pending_recurrent_state(
                    slots, frontier, allow_defer=True
                )
                pending.prepare_acceptance(stopped, frontier)
                pending.accept(
                    zero_accept, read_indices=live_pages, write_indices=live_pages
                )

            measurements["metadata_ms"] = timed(
                metadata, execution=args.execution, repetitions=args.repetitions
            )
            copy_tables = backend._verify_copy_tables_get()
            source = ((slots - 1) * 5 + 4).repeat(layers)
            destination = live_pages.flatten()

            def conv_commit():
                copy_state_rows(
                    copy_tables["conv_scratch"],
                    copy_tables["conv_comp"],
                    source,
                    destination,
                    row_bytes=copy_tables["conv_bytes"],
                    src_row_strides=copy_tables["conv_scratch_stride"],
                    dst_row_strides=copy_tables["conv_comp_stride"],
                )

            measurements["conv_commit_ms"] = timed(
                conv_commit, execution=args.execution, repetitions=args.repetitions
            )

            query = qkv[:, :, : heads * key_dim].view(layers, batch, 4, heads, key_dim)
            key = qkv[:, :, heads * key_dim : 2 * heads * key_dim].view(
                layers, batch, 4, heads, key_dim
            )
            value = qkv[:, :, 2 * heads * key_dim :].view(
                layers, batch, 4, vheads, value_dim
            )
            previous = torch.cat(
                (
                    key.reshape(layers, batch * 4, -1),
                    value.reshape(layers, batch * 4, -1),
                    a,
                    b,
                ),
                -1,
            )
            pending.payload[0].copy_(previous)
            pending.payload[1].copy_(previous)

            def recurrent_step():
                refresh()
                backend.prepare_pending_recurrent_state(
                    slots, frontier, allow_defer=True
                )
                for layer in range(layers):
                    gdn_decode_mtp(
                        query[layer],
                        key[layer],
                        value[layer],
                        A_log=parameters[layer, 0],
                        dt_bias=parameters[layer, 1],
                        a=a[layer].view(batch, 4, vheads),
                        b=b[layer].view(batch, 4, vheads),
                        initial_state=pool.get_component(layer, "recurrent_state"),
                        initial_state_indices=slots,
                        tree_ancestors=None,
                        pending_replay=pending.descriptor(layer, batch),
                    )
                conv_commit()
                pending.prepare_acceptance(stopped, frontier)
                pending.accept(
                    accepted, read_indices=live_pages, write_indices=live_pages
                )
                frontier.index_add_(0, slots.long(), accepted)

            recurrent_ms = timed(
                recurrent_step, execution=args.execution, repetitions=args.repetitions
            )
            measurements["recurrent_step_ms"] = recurrent_ms
            measurements["recurrent_sequence_state_gb_s"] = (
                2 * state_bytes / (recurrent_ms * 1e6)
            )
            backend.materialize_recurrent_state(None)

            def record_pending_replay():
                pending._metadata(
                    "accept",
                    input_accepted=accepted,
                    input_reads=live_pages,
                    input_writes=live_pages,
                    batch=batch,
                    requested=0,
                )

            def dropout_flush(selected_slots):
                record_pending_replay()
                pending.materialize(selected_slots)

            measurements["dropout_flush_ms"] = timed(
                lambda: dropout_flush(None),
                execution=args.execution,
                repetitions=args.repetitions,
            )

            measurements["single_dropout_flush_ms"] = timed(
                lambda: dropout_flush(slots[:1]),
                execution=args.execution,
                repetitions=args.repetitions,
            )
            pending.materialize(None)
            # Measure a layer without forcing a cold cache between sequence layers.
            refresh()
            backend.prepare_pending_recurrent_state(slots, frontier, allow_defer=True)
            pending.rows.copy_(pending.records[slots.long(), 1].to(torch.int32))
            q = qkv[0, :, : heads * key_dim].view(batch, 4, heads, key_dim)
            k = qkv[0, :, heads * key_dim : 2 * heads * key_dim].view(
                batch, 4, heads, key_dim
            )
            v = qkv[0, :, 2 * heads * key_dim :].view(batch, 4, vheads, value_dim)
            pending.rows.copy_(
                (
                    pending.bank[0].long() * batch + torch.arange(batch, device="cuda")
                ).to(torch.int32)
            )
            record_pending_replay()

            def one_layer():
                gdn_decode_mtp(
                    q,
                    k,
                    v,
                    A_log=parameters[0, 0],
                    dt_bias=parameters[0, 1],
                    a=a[0].view(batch, 4, vheads),
                    b=b[0].view(batch, 4, vheads),
                    initial_state=pool.get_component(0, "recurrent_state"),
                    initial_state_indices=slots,
                    tree_ancestors=None,
                    pending_replay=pending.descriptor(0, batch),
                )

            single = timed(
                one_layer, execution=args.execution, repetitions=args.repetitions
            )
            measurements["one_layer_kernel_ms"] = single
            measurements["one_layer_state_gb_s"] = (
                2 * state_bytes / layers / (single * 1e6)
            )
            backend.materialize_recurrent_state(None)
        result["measurements"]["deferred" if defer else "immediate"] = measurements
        print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
