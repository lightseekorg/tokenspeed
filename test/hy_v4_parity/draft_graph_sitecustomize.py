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

"""Sitecustomize that toggles and counts Eagle's draft step-0 prefill graph.

``draft_graph_check.py`` copies this file onto ``PYTHONPATH`` as
``sitecustomize.py`` so it reaches the engine's worker processes, which a patch
in the parent cannot. With ``HY_V4_DRAFT_GRAPH=off`` the drafter declares no
step-0 prefill graph, so the executor neither sizes nor captures one and step 0
runs eagerly while the target keeps its graphs; ``on`` leaves capture alone.
With ``HY_V4_DRAFT_GRAPH_STATS`` set, each drafter worker records the actual
step-0 path, an input fingerprint, proposal ids and margins in its trace. It
also saves complete input, logit and written latent/index-K snapshots for
strict comparison. Decode steps replay inside their graph and never reach
Python. Records are written as they happen because shutdown does not run
``atexit``. This debug mode synchronizes the GPU and writes large artifacts;
leave it unset when timing or running a production server.
"""

import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_next_sitecustomize() -> None:
    """Run the next ``sitecustomize`` on ``sys.path``; Python imports only one."""
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate) as handle:
                source = handle.read()
            namespace = {"__name__": "sitecustomize", "__file__": candidate}
            exec(compile(source, candidate, "exec"), namespace)
            return


_chain_next_sitecustomize()


def _install() -> None:
    mode = os.environ.get("HY_V4_DRAFT_GRAPH")
    if mode not in ("on", "off"):
        return
    from tokenspeed.runtime.execution.drafter import eagle

    if mode == "off":
        eagle.Eagle.captures_prefill_graph = property(lambda self: False)

    stats = os.environ.get("HY_V4_DRAFT_GRAPH_STATS")
    print(f"[draft_graph] mode={mode} stats={stats}", file=sys.stderr)
    if not stats:
        return
    import torch

    from tokenspeed.runtime.execution import forward_step
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.kv_cache.dsa import split_index_k_rows

    stats_path = f"{stats}.{os.getpid()}"
    trace_path = f"{stats_path}.trace"
    counts = {"replays": 0, "eager_extends": 0}
    first_step_evidence = [None]
    evidence_number = [0]

    def record(key: str) -> None:
        counts[key] += 1
        with open(stats_path, "w") as handle:
            json.dump(counts, handle)

    # The decode graph captures the whole step, drafter included: its warmups
    # run on dummy inputs and its replays never reach Python. Skipping the
    # graph phase leaves exactly the served eager steps (extend and mixed).
    forward_first_step = eagle.Eagle._forward_first_step

    def cpu_copy(tensor):
        return tensor.detach().contiguous().cpu().clone()

    def input_key(tensors):
        digest = hashlib.sha256()
        for name, tensor in sorted(tensors.items()):
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def counted_first_step(
        self, ctx, input_ids, positions, captured_hidden_states, **kwargs
    ):
        if forward_step.get_is_cuda_graph_phase():
            return forward_first_step(
                self, ctx, input_ids, positions, captured_hidden_states, **kwargs
            )
        graph = self.prefill_graph
        replayed = graph is not None and graph.can_run(ctx)
        if replayed:
            record("replays")
        elif ctx.num_extends > 0:
            record("eager_extends")
        # Snapshot the live inputs before the forward can mutate metadata.
        # This checker is deliberately one-GPU HY V4 with a native BF16 KV pool.
        inputs = {
            "input_ids": cpu_copy(input_ids),
            "positions": cpu_copy(positions),
            "hidden": cpu_copy(captured_hidden_states),
            "gather_ids": cpu_copy(ctx.gather_ids),
            "frontier": cpu_copy(ctx.draft_narrowing.frontier),
        }
        attention = self.draft_model_runner.model.model.decoder.self_attn
        layer = attention.attn_mqa
        spans = []
        if ctx.num_extends:
            spans.append(ctx.attn_backend.write_locations(layer, ForwardMode.EXTEND))
        if ctx.bs > ctx.num_extends:
            spans.append(ctx.attn_backend.write_locations(layer, ForwardMode.DECODE))
        if not spans:
            raise RuntimeError("draft graph oracle received no KV write span")
        slots = torch.cat(spans).long().clone()
        slots = slots[slots > 0]  # slot zero is the arena's padding sink
        if slots.numel() == 0:
            raise RuntimeError("draft graph oracle received no live KV write slots")
        result = forward_first_step(
            self, ctx, input_ids, positions, captured_hidden_states, **kwargs
        )
        pool = ctx.token_to_kv_pool
        cache = pool.get_key_buffer(layer.layer_id)
        if not isinstance(cache, torch.Tensor) or cache.dtype != torch.bfloat16:
            raise RuntimeError("draft graph oracle requires a native BF16 KV cache")
        width = attention.kv_lora_rank + attention.qk_rope_head_dim
        latent = cache.reshape(-1, width).index_select(0, slots)
        index_format = self.draft_model_runner.model_config.index_k_format
        packed = pool.gather_index_k_rows(
            layer.layer_id, slots, index_k_format=index_format
        )
        keys, scales = split_index_k_rows(
            packed, index_head_dim=pool.index_head_dim, index_k_format=index_format
        )
        if scales is None:
            index_k = keys.float()
        else:
            index_k = keys.contiguous().view(torch.float8_e4m3fn).float()
            index_k = (
                index_k.reshape(slots.numel(), scales.shape[1], -1) * scales[:, :, None]
            ).reshape(slots.numel(), -1)
        evidence_number[0] += 1
        artifact = f"{stats_path}.step0.{evidence_number[0]}.pt"
        torch.save(
            {
                "inputs": inputs,
                "outputs": {
                    "logits": cpu_copy(result.next_token_logits.float()),
                    "latent_kv": cpu_copy(latent.float()),
                    "index_k": cpu_copy(index_k),
                },
            },
            artifact,
        )
        first_step_evidence[0] = {
            "path": "graph" if replayed else "eager",
            "input_key": input_key(inputs),
            "artifact": artifact,
        }
        return result

    eagle.Eagle._forward_first_step = counted_first_step

    # The draft's logits processor picks each step's ids (eagerly, after any
    # replay); the top-2 logit gap per row tells a near-tie flip from a wrong
    # forward. A mid-chunk batch stops after step 0, so the later steps are
    # placeholder ids with no gap.
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor

    argmax = LogitsProcessor._argmax
    margins = []
    drafting = [False]

    def recorded_argmax(self, logits):
        if drafting[0] and not forward_step.get_is_cuda_graph_phase():
            top = logits.float().topk(2, dim=-1).values
            margins.append((top[:, 0] - top[:, 1]).tolist())
        return argmax(self, logits)

    LogitsProcessor._argmax = recorded_argmax
    draft = eagle.Eagle.draft

    def traced_draft(self, draft_input):
        first_step_evidence[0] = None
        margins.clear()
        drafting[0] = True
        try:
            next_tokens = draft(self, draft_input)
        finally:
            drafting[0] = False
        if forward_step.get_is_cuda_graph_phase():
            return next_tokens
        bs, steps = next_tokens.shape[0], next_tokens.shape[1] - 1
        gaps = margins + [[None] * bs] * (steps - len(margins))
        entry = {
            "bs": int(bs),
            "num_extends": int(draft_input.num_extends),
            "accept": draft_input.accept_lengths.tolist(),
            "first_step": first_step_evidence[0],
            "draft": next_tokens[:, 1:].tolist(),
            "margin": [list(row) for row in zip(*gaps)],
        }
        with open(trace_path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
        return next_tokens

    eagle.Eagle.draft = traced_draft


try:
    _install()
except Exception as error:  # pragma: no cover - a silent miss would fake a pass
    print(f"[draft_graph] install failed: {error}", file=sys.stderr)
    raise
