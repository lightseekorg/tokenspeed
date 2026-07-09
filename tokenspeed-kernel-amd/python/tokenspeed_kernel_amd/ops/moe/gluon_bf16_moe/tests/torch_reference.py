"""Pure-torch fp32 reference + reference router for the bf16 MoE tests."""

from __future__ import annotations

import torch


def routing_softmax_topk(router_logits: torch.Tensor, topk: int):
    """Reference router: softmax over experts then top-k (renormalised).
    Returns (topk_ids int32, topk_weights float32) with rows summing to 1."""
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


def torch_moe_ref(
    hidden_states: torch.Tensor,  # (T, D)     bf16/float
    w1: torch.Tensor,  # (E, 2*I, D) bf16/float  gate [0:I], up [I:2I]
    w2: torch.Tensor,  # (E, D, I)   bf16/float
    topk_ids: torch.Tensor,  # (T, topk) int
    topk_weights: torch.Tensor,  # (T, topk) float
) -> torch.Tensor:
    """Golden MoE FFN in fp32; returns ``(T, D)`` float32.

    y[t] = sum_s w[t,s] * ( silu(h@w1_e[:I].T) * (h@w1_e[I:].T) ) @ w2_e.T
    """
    T, D = hidden_states.shape
    topk = topk_ids.shape[1]
    two_I = w1.shape[1]
    I = two_I // 2

    hf = hidden_states.float()
    w1f = w1.float()
    w2f = w2.float()
    ids = topk_ids.cpu()
    wts = topk_weights.float().cpu()

    out = torch.zeros(T, D, dtype=torch.float32, device=hidden_states.device)
    for t in range(T):
        acc = torch.zeros(D, dtype=torch.float32, device=hidden_states.device)
        a = hf[t]
        for s in range(topk):
            e = int(ids[t, s])
            g = a @ w1f[e].T  # (2*I,)
            inter = torch.nn.functional.silu(g[:I]) * g[I:]  # (I,)
            down = inter @ w2f[e].T  # (D,)
            acc += float(wts[t, s]) * down
        out[t] = acc
    return out
