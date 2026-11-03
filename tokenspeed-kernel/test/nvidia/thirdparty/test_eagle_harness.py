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

"""EAGLE-Harness draft-tree kernels against PyTorch references."""

import numpy as np
import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.eagle_harness import (
    eagle_harness_draft_depth_update,
    eagle_harness_publish_draft_q,
    eagle_harness_residual_sample,
    eagle_harness_sample_children,
    eagle_harness_stable_topk_desc,
    eagle_harness_tree_reject_sample,
    eagle_harness_work_deltas,
    eagle_harness_write_verify_tree,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="eagle_harness CUDA kernels are NVIDIA-only",
)

_DEVICE = "cuda"


def _count(value: int) -> torch.Tensor:
    """A device int32 ``[1]`` count, as the drafter's round state holds them."""
    return torch.tensor([value], dtype=torch.int32, device=_DEVICE)


def _distinct_scores(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """Scores in (0, 1) whose values and products almost surely never tie."""
    return (torch.rand((rows, cols), generator=generator) * 0.98 + 0.01).to(_DEVICE)


@pytest.mark.parametrize("n,k", [(5, 5), (1024, 64)])
def test_stable_topk_desc(n: int, k: int) -> None:
    generator = torch.Generator().manual_seed(n)
    # Ties keep the lower index first; a padded row and columns past n are never read or written.
    cols = min(n + 3, 1024)
    scores = (torch.randint(0, 8, (4, cols), generator=generator).float() / 8).to(
        _DEVICE
    )
    values = torch.full((4, k + 2), -1.0, device=_DEVICE)
    indices = torch.full((4, k + 2), -1, dtype=torch.int64, device=_DEVICE)
    eagle_harness_stable_topk_desc(
        scores, _count(3), _count(n), _count(k), values, indices
    )
    expected = torch.sort(scores[:3, :n], dim=-1, descending=True, stable=True)
    torch.testing.assert_close(values[:3, :k], expected.values[:, :k], rtol=0, atol=0)
    torch.testing.assert_close(indices[:3, :k], expected.indices[:, :k], rtol=0, atol=0)
    assert (values[3] == -1).all() and (values[:, k:] == -1).all()


@pytest.mark.parametrize("node_count,verify_num", [(1, 16), (14, 16)])
def test_work_deltas(node_count: int, verify_num: int) -> None:
    bs, topk = 3, 4
    generator = torch.Generator().manual_seed(node_count)
    # One padded row past the live ones, as in a captured graph.
    work = torch.ones((bs + 1, verify_num + topk), device=_DEVICE)
    work[:bs] = 0.0
    work[:bs, :node_count] = (
        _distinct_scores(bs, node_count, generator).sort(dim=-1, descending=True).values
    )
    work[:bs, node_count : node_count + topk] = (
        _distinct_scores(bs, topk, generator).sort(dim=-1, descending=True).values
    )
    deltas = torch.empty(topk + 1, device=_DEVICE)
    live_rows = torch.tensor([bs], dtype=torch.int32, device=_DEVICE)
    counts = _count(node_count), _count(verify_num)
    eagle_harness_work_deltas(work, deltas, live_rows, counts[0], counts[1])
    unpadded = torch.empty(topk + 1, device=_DEVICE)
    eagle_harness_work_deltas(work[:bs], unpadded, live_rows, counts[0], counts[1])
    # A zero node count queues no deltas.
    untouched = torch.full((topk + 1,), -1.0, device=_DEVICE)
    eagle_harness_work_deltas(work, untouched, live_rows, _count(0), counts[1])

    assert torch.equal(deltas, unpadded) and (untouched == -1).all()
    expected = []
    for j in range(topk + 1):
        merged = work[:bs, : node_count + j].sort(dim=-1, descending=True).values
        expected.append(merged[:, : min(node_count + j, verify_num)].sum(-1).mean())
    torch.testing.assert_close(deltas, torch.stack(expected), rtol=1e-6, atol=1e-6)


def _work_deltas_reference(
    work: np.ndarray, bs: int, node_count: int, verify_num: int, topk: int, rows: int
) -> np.ndarray:
    """The work deltas in the kernel's float32 order: each row's merge summed
    in turn, then a halving tree over 256 threads, or 512 above 256 rows,
    whose padded rows hold zeros."""
    threads = 256 if rows <= 256 else 512
    out = np.zeros(topk + 1, dtype=np.float32)
    for j in range(topk + 1):
        sums = np.zeros(threads, dtype=np.float32)
        for row in range(bs):
            head, tail = work[row, :node_count], work[row, node_count:]
            a = b = 0
            total = np.float32(0.0)
            for _ in range(min(node_count + j, verify_num)):
                va = head[a] if a < node_count else np.float32(0.0)
                vb = tail[b] if b < j else np.float32(0.0)
                if va >= vb:
                    total, a = np.float32(total + va), a + 1
                else:
                    total, b = np.float32(total + vb), b + 1
            sums[row] = total
        half = threads // 2
        while half:
            sums[:half] = sums[:half] + sums[half : 2 * half]
            half //= 2
        out[j] = np.float32(sums[0] / np.float32(bs))
    return out


@pytest.mark.parametrize(
    "bs,rows",
    [(3, 4), (200, 256), (256, 256), (250, 272), (257, 264), (300, 512), (512, 512)],
)
def test_work_deltas_sum_order_to_512_rows(bs: int, rows: int) -> None:
    """Bit for bit the kernel's summation order up to 512 rows (padded rows
    past the live ones, as in a captured graph); up to 256 live rows the
    256- and 512-thread halving trees give the same bits."""
    node_count, verify_num, topk = 7, 16, 4
    generator = torch.Generator().manual_seed(bs)
    work = torch.zeros((rows, verify_num + topk))
    work[:bs, :node_count] = _distinct_scores(bs, node_count, generator).cpu()
    work[:bs, :node_count] = work[:bs, :node_count].sort(-1, descending=True).values
    tail = _distinct_scores(bs, topk, generator).cpu().sort(-1, descending=True).values
    work[:bs, node_count : node_count + topk] = tail
    work[bs:] = 1.0
    deltas = torch.empty(topk + 1, device=_DEVICE)
    eagle_harness_work_deltas(
        work.to(_DEVICE), deltas, _count(bs), _count(node_count), _count(verify_num)
    )
    expected = _work_deltas_reference(
        work.numpy(), bs, node_count, verify_num, topk, rows
    )
    assert np.array_equal(deltas.cpu().numpy(), expected)
    if bs <= 256:
        other = _work_deltas_reference(
            work.numpy(), bs, node_count, verify_num, topk, 512 if rows <= 256 else 256
        )
        assert np.array_equal(other, expected)


def _bitonic_sort_64(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernels' descending sort of each row's ``n <= 64`` scores: a
    bitonic network over 64 positions that skips every pair reaching past
    ``n`` and never swaps ties, so its order (not always fully sorted when
    ``n`` is not a power of two) is a fixed function of the inputs. Returns
    the values and their original positions."""
    rows, n = scores.shape
    values = scores.clone()
    order = torch.arange(n, device=scores.device).repeat(rows, 1)
    size = 2
    while size <= 64:
        stride = size // 2
        while stride:
            lower = [p for p in range(64) if not p & stride and p + stride < n]
            if lower:
                i = torch.tensor(lower, device=scores.device)
                j = i + stride
                vi, vj, oi, oj = values[:, i], values[:, j], order[:, i], order[:, j]
                swap = torch.where((i & size) == 0, vi < vj, vi > vj)
                values[:, i] = torch.where(swap, vj, vi)
                values[:, j] = torch.where(swap, vi, vj)
                order[:, i] = torch.where(swap, oj, oi)
                order[:, j] = torch.where(swap, oi, oj)
            stride //= 2
        size *= 2
    return values, order


def _lane_mask_words(mask: torch.Tensor, lane_mask: torch.Tensor) -> None:
    """Bool ``[rows, K, window]`` lane masks as int64 words: window slot ``j``
    at bit ``j % 64`` of word ``j // 64``, later words zero."""
    window = mask.shape[-1]
    bits = 1 << (torch.arange(window, device=mask.device) % 64)
    for word in range(lane_mask.shape[-1]):
        lo, hi = 64 * word, min(64 * word + 64, window)
        if lo >= hi:
            lane_mask[..., word].zero_()
            continue
        torch.sum(mask[..., lo:hi] * bits[lo:hi], dim=-1, out=lane_mask[..., word])


def _unfused_depth(
    state: dict[str, torch.Tensor],
    masks: list[torch.Tensor],
    child_p: torch.Tensor,
    child_tokens: torch.Tensor,
    expanded: torch.Tensor,
    child_hidden: torch.Tensor,
    depth: int,
    cumu_count: int,
    verify_num: int,
    num_slots: int,
) -> None:
    """The sequence the fused update replaces (child scoring and sort, the
    next frontier, the pool append, the sort and work lists, the next depth's
    bool ancestor masks ``masks[depth + 1]`` and their lane-mask words, the
    lane increments) on ``state``'s first rows."""
    rows, width, k = child_p.shape
    n = width * k
    row = torch.arange(rows, device=_DEVICE)[:, None]
    path = (child_p * state["scores"][:rows, :width, None]).reshape(rows, n)
    tokens = child_tokens.reshape(rows, n)
    layer, order = _bitonic_sort_64(path)
    best = order[:, :k]
    parent = best // k
    state["frontier_index"][depth % 2][:rows] = cumu_count + best
    state["scores"][:rows] = layer[:, :k]
    state["tokens"][:rows] = tokens.gather(1, best)
    state["hidden"][:rows] = child_hidden[row, parent]
    # Device powf: CUDA pow with a tensor exponent.
    decay = torch.pow(
        torch.tensor(0.9999, device=_DEVICE), torch.tensor(depth + 1.0, device=_DEVICE)
    )
    new = slice(cumu_count, cumu_count + n)
    state["pool_tokens"][:rows, new] = tokens
    state["pool_scores"][:rows, new] = path * decay
    state["pool_prev"][:rows, new] = expanded[:, :width].repeat_interleave(k, dim=1)
    # Work list, then the sort list's stable rank merge with the children.
    head, kept = min(verify_num, cumu_count), min(verify_num, cumu_count + n)
    merged = torch.cat([state["sort"][:rows, :head], layer], dim=1)
    pos = torch.arange(merged.shape[1], device=_DEVICE)
    above = (merged[:, None, :] > merged[:, :, None]) | (
        (merged[:, None, :] == merged[:, :, None]) & (pos[None, :] < pos[:, None])
    )
    rank = above.sum(-1)
    state["work"][:rows, :head] = state["sort"][:rows, :head]
    state["work"][:rows, head : head + k] = layer[:, :k]
    keep = rank < kept
    state["sort"][row.expand_as(rank)[keep], rank[keep]] = merged[keep]
    # The next depth's ancestors: the parents' rows, then each lane's own slot.
    if (depth + 1) * k <= num_slots:
        window = depth * k
        masks[depth + 1][:rows, :, :window] = masks[depth][row, parent, :window]
        _lane_mask_words(masks[depth + 1][:rows], state["lane_mask"][:rows])
    state["positions"][:rows] += 1
    state["slot_starts"][:rows] += k


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, name: str):
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name


# A 64-step chain, 128 lane slots (two mask words), 64 children per row.
@pytest.mark.parametrize(
    "rows,max_rows,k,steps",
    [(3, 5, 1, 64), (3, 5, 4, 33), (3, 5, 8, 16), (8, 8, 2, 64)],
)
def test_draft_depth_update_matches_the_unfused_sequence(
    rows: int, max_rows: int, k: int, steps: int
) -> None:
    """A round of depths at random widths: from the same forward outputs the
    fused update leaves every buffer bit-identical to the unfused sequence,
    ties included (few distinct probabilities and path scores); rows past
    ``rows`` (graph padding of the buffers) stay untouched."""
    gen = torch.Generator(device=_DEVICE).manual_seed(1000 * rows + 10 * k + steps)
    hidden_size = 1032  # 129 16-byte vectors per row
    nodes, verify = max(1024, k * k * steps), max(64, k * (steps + 1) + 1)
    num_slots = (steps - 1) * k
    words = (num_slots + 63) // 64

    def ints(shape, high=1000):
        return torch.randint(-1, high, shape, generator=gen, device=_DEVICE)

    def ties(shape):
        return torch.randint(1, 5, shape, generator=gen, device=_DEVICE) / 4

    state = {
        "pool_tokens": ints((max_rows, nodes)),
        "pool_scores": torch.rand((max_rows, nodes), generator=gen, device=_DEVICE),
        "pool_prev": ints((max_rows, nodes)),
        "work": torch.rand((max_rows, verify + k), generator=gen, device=_DEVICE),
        "sort": ties((max_rows, verify)).sort(dim=-1, descending=True).values,
        "tokens": ints((max_rows, k)),
        "scores": ties((max_rows, k)),
        "hidden": torch.randn(
            (max_rows, k, hidden_size), generator=gen, device=_DEVICE
        ).to(torch.bfloat16),
        "frontier_index": ints((2, max_rows, k)),
        "positions": ints((max_rows, k)),
        "slot_starts": ints((max_rows,)),
        "lane_mask": ints((max_rows, k, words), 1 << 62),
    }
    # Depth 1's lane masks: lane r sees its own slot.
    state["lane_mask"][:rows] = 0
    state["lane_mask"][:rows, :, :1] = (1 << torch.arange(k, device=_DEVICE))[:, None]
    fused = {name: t.clone() for name, t in state.items()}
    # The unfused sequence's bool masks per depth: stale ancestors, own slots.
    masks = [torch.empty(0)]
    for s in range(1, steps):
        mask = torch.rand((max_rows, k, s * k), generator=gen, device=_DEVICE) < 0.5
        mask[:, :, (s - 1) * k :] = torch.eye(k, dtype=torch.bool, device=_DEVICE)
        masks.append(mask)
    root_children = torch.arange(1, k + 1, device=_DEVICE).repeat(rows, 1)

    def fused_update(
        child_p, child_tokens, expanded, child_hidden, width, depth, cumu_count
    ):
        eagle_harness_draft_depth_update(
            child_p, child_tokens, expanded, child_hidden, fused["tokens"][:rows],
            fused["scores"][:rows], fused["hidden"][:rows],
            fused["frontier_index"][depth % 2][:rows], fused["pool_tokens"][:rows],
            fused["pool_scores"][:rows], fused["pool_prev"][:rows], fused["work"][:rows],
            fused["sort"][:rows], fused["positions"][:rows],
            fused["slot_starts"][:rows], fused["lane_mask"][:rows], _count(width),
            depth, _count(cumu_count), _count(verify_num), num_slots,
        )  # fmt: skip

    cumu_count = 1 + k
    verify_num = int(torch.randint(1, 65, (1,), generator=gen, device=_DEVICE))
    for depth in range(1, steps):
        width = int(torch.randint(1, k + 1, (1,), generator=gen, device=_DEVICE))
        # The lane forward's outputs: K children per lane, the first width expanded.
        child_p = ties((rows * k, k)).view(rows, k, k)
        child_tokens = ints((rows * k, k)).view(rows, k, k)
        child_hidden = torch.randn(
            (rows * k, hidden_size), generator=gen, device=_DEVICE
        ).to(torch.bfloat16)
        expanded = [
            root_children if depth == 1 else s["frontier_index"][(depth + 1) % 2][:rows]
            for s in (state, fused)
        ]
        _unfused_depth(
            state, masks, child_p[:, :width], child_tokens[:, :width], expanded[0],
            child_hidden.view(rows, k, -1), depth, cumu_count, verify_num, num_slots,
        )  # fmt: skip
        fused_update(
            child_p, child_tokens, expanded[1], child_hidden.view(rows, k, -1), width,
            depth, cumu_count,
        )  # fmt: skip
        cumu_count += width * k
        for name, expected in state.items():
            _assert_bitwise_equal(fused[name], expected, f"{name} at depth {depth}")


def test_sample_children_reads_strided_rows() -> None:
    """Each request's first R of K lane rows, read in place (and the head's
    row-strided candidate ids), draw exactly what packed copies draw."""
    reqs, k, expanded, nc, vocab = 3, 4, 2, 64, 300
    gen = torch.Generator().manual_seed(7)
    logits = (torch.randn((reqs * k, nc), generator=gen) * 2).to(_DEVICE)
    cand = torch.randint(0, vocab + 20, (reqs * k, nc + 3), generator=gen).to(_DEVICE)
    cand = cand[:, :nc]
    tokens = torch.randint(0, vocab, (reqs * k, k), generator=gen).to(_DEVICE)
    scores = torch.rand((reqs * k, k), generator=gen).to(_DEVICE)
    # A sampling request, a greedy one, and one cut to top-k / top-p.
    temps = torch.tensor([1.0, 0.0, 0.7], device=_DEVICE)
    top_ks = torch.tensor([1 << 30, 1 << 30, 5], dtype=torch.int32, device=_DEVICE)
    top_ps = torch.tensor([1.0, 1.0, 0.8], device=_DEVICE)

    def lanes(rows: torch.Tensor) -> torch.Tensor:
        return rows.view(reqs, k, -1)[:, :expanded]

    def draw(*rows: torch.Tensor, width: torch.Tensor | None) -> list[torch.Tensor]:
        tables = [
            torch.zeros((reqs, 1 + k, nc), device=_DEVICE),
            torch.zeros((reqs, 1 + k, nc), dtype=torch.int32, device=_DEVICE),
            torch.full((reqs, 64), -1, dtype=torch.int32, device=_DEVICE),
            torch.zeros((reqs, 64), dtype=torch.int32, device=_DEVICE),
        ]
        eagle_harness_sample_children(
            *rows, temps, top_ks, top_ps, *tables, width, 1, _count(5), vocab, 11,
            torch.tensor([3], device=_DEVICE), 1,
        )  # fmt: skip
        return tables

    def full(rows: torch.Tensor) -> torch.Tensor:
        return rows.view(reqs, k, -1)

    # All K lanes with the expanded width on the device draw what the expanded lanes alone draw.
    packed = [lanes(rows).contiguous() for rows in (logits, cand, tokens, scores)]
    packed_tables = draw(*packed, width=None)
    in_place = tokens.clone(), scores.clone()
    tables = draw(
        full(logits), full(cand), *map(full, in_place), width=_count(expanded)
    )
    assert (packed_tables[2][0] >= 0).any() and (packed_tables[2][1] == -1).all()
    for actual, expected in zip(
        [*map(lanes, in_place), *tables], packed[2:] + packed_tables
    ):
        assert torch.equal(actual, expected)
    # The lanes past the expanded ones keep their children.
    for actual, before in zip(in_place, (tokens, scores)):
        assert torch.equal(
            actual.view(reqs, k, k)[:, expanded:], before.view(reqs, k, k)[:, expanded:]
        )


@pytest.mark.parametrize("verify_num", [1, 16])
def test_write_verify_tree(verify_num: int) -> None:
    """Ranked nodes' tokens and parent slots, the rest the root (pool node 0)
    with parent -1, in each request's row of the next round's maps."""
    bs, nodes, slots = 3, 64, 16
    generator = torch.Generator().manual_seed(verify_num)
    prev = torch.full((bs, nodes), -1, dtype=torch.int64)
    for node in range(1, nodes):
        prev[:, node] = torch.randint(0, node, (bs,), generator=generator)
    tokens = torch.randint(0, 1000, (bs, nodes), generator=generator)
    # Ancestor-first rankings (each parent ranked first); a padded row past the live ones is never written.
    ranked = torch.zeros((bs + 1, 16), dtype=torch.int64)
    for b in range(bs):
        chosen = [0]
        while len(chosen) < 16:
            node = int(torch.randint(1, nodes, (1,), generator=generator))
            if node not in chosen and int(prev[b, node]) in chosen:
                chosen.append(node)
        ranked[b] = torch.tensor(chosen)
    rows = torch.tensor([5, 2, 9, 11], dtype=torch.int64)
    prev, tokens = (torch.cat([t, t[:1]]) for t in (prev, tokens))
    input_map = torch.zeros((12, slots), dtype=torch.int32, device=_DEVICE)
    parent_map = torch.zeros((12, slots), dtype=torch.int32, device=_DEVICE)
    eagle_harness_write_verify_tree(
        ranked.to(_DEVICE), tokens.to(_DEVICE), prev.to(_DEVICE), rows.to(_DEVICE),
        input_map, parent_map, _count(bs), _count(verify_num),
    )  # fmt: skip
    assert (input_map[11] == 0).all() and (parent_map[11] == 0).all()

    for b in range(bs):
        want_tokens = [int(tokens[b, 0])] * slots
        want_parents = [-1] * slots
        for i in range(verify_num):
            node = int(ranked[b, i])
            want_tokens[i] = int(tokens[b, node])
            parent = int(prev[b, node])
            want_parents[i] = ranked[b, :i].tolist().index(parent) if i else -1
        assert input_map[rows[b]].tolist() == want_tokens
        assert parent_map[rows[b]].tolist() == want_parents


def _sampled_round(target_rows, draft_logits, cand, topk, generator):
    """One round per request at temperature 1: sample the root's K children
    (and each child's K grandchildren) from the draft, accept by rejection,
    draw the bonus; returns each request's emitted tokens (-1 past them).

    target_rows: ``[1 + K + K * K, V]`` the target's distribution at every node
    (root, children, grandchildren in pool order); draft_logits / cand:
    ``[1 + K, NC]`` the draft's logits over candidate ids at the root and at
    each child."""
    # The children kernel keys at most 1024 rows per launch: 256 requests x 3 children.
    reqs, nc = 256, cand.shape[1]
    k, vocab = topk, target_rows.shape[1]
    max_nodes, slots = 64, 1 + k
    nodes = 1 + k + k * k
    dev = "cuda"
    temps = torch.ones(reqs, device=dev)
    top_ks = _no_top_k(reqs)
    top_ps = torch.ones(reqs, device=dev)
    q_store = torch.zeros((reqs, slots, nc), device=dev)
    qid_store = torch.zeros((reqs, slots, nc), dtype=torch.int32, device=dev)
    node_slot = torch.full((reqs, max_nodes), -1, dtype=torch.int32, device=dev)
    node_pos = torch.zeros((reqs, max_nodes), dtype=torch.int32, device=dev)
    counter = torch.randint(0, 1 << 30, (1,), generator=generator).to(dev)
    seed = int(torch.randint(0, 1 << 40, (1,), generator=generator))
    # Depth 1: the root's children; depth 2: each child's children.
    root_tokens = cand[0, :k].repeat(reqs, 1).contiguous()
    eagle_harness_sample_children(
        draft_logits[:1].repeat(reqs, 1)[:, None], cand[:1].repeat(reqs, 1)[:, None],
        root_tokens[:, None], torch.zeros((reqs, 1, k), device=dev), temps, top_ks, top_ps,
        q_store, qid_store, node_slot, node_pos, None, 0, _count(1), vocab, seed, counter,
        0,
    )  # fmt: skip
    # A child's draft row is its own (indexed by the child's rank among the root's).
    child_logits = draft_logits[1:].repeat(reqs, 1).contiguous()
    child_cand = cand[1:].repeat(reqs, 1).contiguous()
    grand_tokens = child_cand[:, :k].contiguous()
    eagle_harness_sample_children(
        child_logits.view(reqs, k, -1), child_cand.view(reqs, k, -1),
        grand_tokens.view(reqs, k, k), torch.zeros((reqs, k, k), device=dev),
        temps, top_ks, top_ps, q_store, qid_store, node_slot, node_pos,
        None, 1, _count(1 + k), vocab, seed, counter, 1,
    )  # fmt: skip
    tokens = torch.cat(
        [torch.zeros((reqs, 1), dtype=torch.int64, device=dev), root_tokens,
         grand_tokens.view(reqs, k * k)], dim=1,
    ).to(torch.int32)  # fmt: skip
    parents = torch.tensor(
        [-1] + [0] * k + [1 + c for c in range(k) for _ in range(k)],
        dtype=torch.int32,
        device=dev,
    ).repeat(reqs, 1)
    rows = torch.arange(reqs, device=dev)
    probas = target_rows.repeat(reqs, 1).contiguous()
    argmax = probas.argmax(dim=-1)
    q_tab = torch.zeros_like(q_store)
    qid_tab = torch.zeros_like(qid_store)
    slot_tab = torch.zeros((reqs, nodes), dtype=torch.int32, device=dev)
    pos_tab = torch.zeros_like(slot_tab)
    ranked = torch.arange(nodes, device=dev).repeat(reqs, 1)
    eagle_harness_publish_draft_q(
        ranked, node_slot, node_pos, q_store, qid_store, rows, q_tab, qid_tab,
        slot_tab, pos_tab, _count(reqs), _count(nodes),
    )  # fmt: skip
    # Every drafted node publishes a q (slot, position) naming its own candidate.
    sampled = slot_tab >= 0
    assert sampled[:, 1:].all()
    req = torch.arange(reqs, device=dev)[:, None].expand(-1, nodes)
    named = qid_tab[req, slot_tab.clamp(min=0).long(), pos_tab.long()]
    assert torch.equal(named[sampled], tokens[sampled])
    coins = torch.rand((reqs, nodes), generator=generator).to(dev)
    final_coins = torch.rand(reqs, generator=generator).to(dev)
    path, predicts, accept_length, _ = _verify(
        probas, tokens, parents, rows, (slot_tab, pos_tab, q_tab, qid_tab), coins,
        final_coins, _no_top_k(reqs * nodes), argmax,
    )  # fmt: skip
    emitted = predicts[:, :3].cpu()
    length = accept_length.cpu()
    emitted[torch.arange(3)[None, :] >= length[:, None]] = -1
    return emitted, path.cpu()


# K = 1: a sampled chain.
@pytest.mark.parametrize("topk", [3, 1])
def test_sampled_tree_emits_the_target_distribution(topk: int) -> None:
    """Children drawn from the draft, recursive rejection and the residual bonus
    together emit exactly the target: the first token's law is the root's
    target, the second's (after an accepted child) that child's."""
    vocab, nc = 48, 32
    generator = torch.Generator().manual_seed(0)
    target = torch.softmax(
        torch.randn((1 + topk + topk * topk, vocab), generator=generator) * 1.5, dim=-1
    ).cuda()
    cand = torch.stack(
        [torch.randperm(vocab, generator=generator)[:nc] for _ in range(1 + topk)]
    ).cuda()
    draft_logits = (torch.randn((1 + topk, nc), generator=generator) * 1.5).cuda()
    first = torch.zeros(vocab)
    second = torch.zeros((topk, vocab))
    accepted = 0
    trials = 240
    for _ in range(trials):
        emitted, path = _sampled_round(target, draft_logits, cand, topk, generator)
        first += torch.bincount(emitted[:, 0], minlength=vocab).float()
        # Rows whose first token is an accepted child (node path[1]) and that emitted a second.
        child = path[:, 1].long() - 1
        accepted += int((child >= 0).sum())
        has_second = (child >= 0) & (emitted[:, 1] >= 0)
        for c in range(topk):
            rows = has_second & (child == c)
            second[c] += torch.bincount(emitted[rows, 1], minlength=vocab).float()
    total = trials * 256
    torch.testing.assert_close(first / total, target[0].cpu(), rtol=0, atol=0.01)
    if topk == 1:
        # x ~ q accepted w.p. min(1, p(x) / q(x)): sum(min(p, q)), above sum(p * q) of accepting by p.
        q = torch.softmax(draft_logits[0], dim=-1)
        p = target[0, cand[0]]
        want = float(torch.minimum(p, q).sum())
        assert want > float((p * q).sum()) + 0.05
        assert abs(accepted / total - want) < 0.01
        assert second[0].sum() > 3000
    for c in range(topk):
        if second[c].sum() > 3000:
            torch.testing.assert_close(
                second[c] / second[c].sum(), target[1 + c].cpu(), rtol=0, atol=0.02
            )


def test_deterministic_children_emit_the_target_distribution() -> None:
    """Children without a draft q are accepted by p'(child) over the mass their
    rejected siblings left: with the residual bonus the first token's law is
    the root's target."""
    vocab, bs, trials = 16, 8192, 24
    generator = torch.Generator().manual_seed(1)
    target = torch.softmax(torch.randn(vocab, generator=generator) * 1.5, dim=-1)
    second, best = torch.topk(target, 2).indices.tolist()[::-1]
    tokens = torch.tensor([0, second, best], dtype=torch.int32, device=_DEVICE)
    tokens = tokens.repeat(bs, 1)
    parents = torch.tensor([-1, 0, 0], dtype=torch.int32, device=_DEVICE).repeat(bs, 1)
    q_tables = (
        torch.full((1, 3), -1, dtype=torch.int32, device=_DEVICE),
        torch.zeros((1, 3), dtype=torch.int32, device=_DEVICE),
        torch.zeros((1, 1, 1), device=_DEVICE),
        torch.zeros((1, 1, 1), dtype=torch.int32, device=_DEVICE),
    )
    unread = torch.full((bs * 3,), -1, dtype=torch.int64, device=_DEVICE)
    probas = target.to(_DEVICE).repeat(bs * 3, 1)
    first = torch.zeros(vocab)
    for _ in range(trials):
        _, predicts, _, _ = _verify(
            probas.clone(), tokens, parents, None, q_tables,
            torch.rand((bs, 3), device=_DEVICE), torch.rand(bs, device=_DEVICE),
            _no_top_k(bs * 3), unread,
        )  # fmt: skip
        first += torch.bincount(predicts[:, 0].long().cpu(), minlength=vocab).float()
    torch.testing.assert_close(first / (bs * trials), target, rtol=0, atol=0.005)


def _no_top_k(rows: int) -> torch.Tensor:
    """Per-row top-k of sampled (never greedy) rows."""
    return torch.full((rows,), 1 << 30, dtype=torch.int32, device=_DEVICE)


def _verify(
    probas, tokens, parents, q_rows, q_tables, coins, final_coins, top_ks, argmax
):
    """Rejection-sample each request's tree, then draw its bonus: path,
    predicts (``[bs, N]``), accept length and last row."""
    bs, n = tokens.shape
    path = torch.empty((bs, n), dtype=torch.int32, device=_DEVICE)
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=_DEVICE)
    accept_length = torch.empty(bs, dtype=torch.int32, device=_DEVICE)
    last_row = torch.empty(bs, dtype=torch.int64, device=_DEVICE)
    eagle_harness_tree_reject_sample(
        probas, tokens, parents, q_rows, *q_tables, coins, top_ks, argmax, path,
        predicts, accept_length, last_row,
    )  # fmt: skip
    eagle_harness_residual_sample(
        probas, last_row, final_coins, top_ks, argmax, accept_length, predicts, n
    )
    return path, predicts.view(bs, n), accept_length, last_row


def _verify_sampled(probas, tokens, parents, q_tables, coins, final_coins):
    """``_verify`` of sampled rows, q tables at the requests' own rows; the
    argmax is the kernels' own, read from probas."""
    bs, n = tokens.shape
    unread = torch.full((bs * n,), -1, dtype=torch.int64, device=_DEVICE)
    _, predicts, accept_length, _ = _verify(
        probas, tokens, parents, torch.arange(bs, device=_DEVICE), q_tables, coins,
        final_coins, _no_top_k(bs * n), unread,
    )  # fmt: skip
    return predicts, accept_length


def _sample_root_children(logits, cand, tokens, *, top_k, top_p, vocab, seed):
    """Redraw each request's ``[bs, K]`` root children in place from its
    ``[bs, NC]`` draft logits over ``cand`` at temperature 1; returns the q
    tables (store, ids, node slots, node positions)."""
    bs, nc = logits.shape
    k = tokens.shape[1]
    tables = (
        torch.zeros((bs, 1, nc), device=_DEVICE),
        torch.zeros((bs, 1, nc), dtype=torch.int32, device=_DEVICE),
        torch.full((bs, 8), -1, dtype=torch.int32, device=_DEVICE),
        torch.zeros((bs, 8), dtype=torch.int32, device=_DEVICE),
    )
    eagle_harness_sample_children(
        logits[:, None], cand[:, None], tokens[:, None],
        torch.zeros((bs, 1, k), device=_DEVICE), torch.ones(bs, device=_DEVICE),
        torch.full((bs,), top_k, dtype=torch.int32, device=_DEVICE),
        torch.full((bs,), top_p, device=_DEVICE), *tables, None, 0, _count(1), vocab,
        seed, torch.zeros(1, dtype=torch.int64, device=_DEVICE), 0,
    )  # fmt: skip
    return tables


def test_exhausted_residual_accepts_the_target_argmax() -> None:
    """q one-hot on the target's one-hot token (a logit gap past fp32 exp): the
    drawn child is accepted for every coin, even one just below 1."""
    bs, vocab, nc, k = 64, 300, 16, 3
    logits = torch.full((bs, nc), -200.0, device=_DEVICE)
    logits[:, 5] = 0.0
    cand = torch.arange(nc, device=_DEVICE).repeat(bs, 1).contiguous()
    tokens = cand[:, :k].contiguous()
    q_store, qid_store, node_slot, node_pos = _sample_root_children(
        logits, cand, tokens, top_k=1 << 30, top_p=1.0, vocab=vocab, seed=3
    )
    assert (tokens[:, 0] == 5).all()
    tree = (
        torch.stack([torch.full((bs,), 7, device=_DEVICE), tokens[:, 0]], 1).to(torch.int32),
        torch.tensor([-1, 0], dtype=torch.int32, device=_DEVICE).repeat(bs, 1),
        (
            torch.stack([torch.full((bs,), -1, device=_DEVICE), node_slot[:, 1]], 1).to(torch.int32),
            torch.stack([torch.zeros(bs, device=_DEVICE), node_pos[:, 1]], 1).to(torch.int32),
            q_store, qid_store,
        ),
    )  # fmt: skip
    probas = torch.zeros((bs * 2, vocab), device=_DEVICE)
    probas[:, 5] = 1.0
    coins = torch.full((bs, 2), 1.0 - 2**-24, device=_DEVICE)
    predicts, accept_length = _verify_sampled(
        probas, *tree, coins, torch.rand(bs, device=_DEVICE)
    )
    assert (accept_length == 2).all() and (predicts[:, 0] == 5).all()


@pytest.mark.parametrize("top_k,top_p", [(3, 1.0), (1 << 30, 0.5)])
def test_sampled_children_respect_top_k_and_top_p(top_k: int, top_p: float) -> None:
    """A request draws only from the draft's ``top_k`` best candidates and,
    with top-p, from the smallest prefix (by q) holding that mass."""
    bs, vocab, nc, k = 256, 300, 16, 2
    gen = torch.Generator().manual_seed(4)
    logits = (
        (torch.randn((1, nc), generator=gen) * 2).to(_DEVICE).repeat(bs, 1).contiguous()
    )
    cand = (
        torch.randperm(vocab, generator=gen)[:nc].to(_DEVICE).repeat(bs, 1).contiguous()
    )
    probs = torch.softmax(logits[0], -1)
    order = torch.argsort(probs, descending=True)
    keep = min(top_k, int((torch.cumsum(probs[order], 0) < top_p).sum()) + 1)
    allowed = set(cand[0, order[:keep]].tolist())
    tokens = cand[:, :k].contiguous()
    _sample_root_children(
        logits, cand, tokens, top_k=top_k, top_p=top_p, vocab=vocab, seed=5
    )
    first = set(tokens[:, 0].tolist())
    assert first <= allowed and len(first) > 1


def test_publish_marks_slots_past_the_verify_width_deterministic() -> None:
    bs, nodes, nc = 4, 8, 16
    ranked = torch.arange(nodes, device=_DEVICE).repeat(bs, 1)
    node_slot = torch.zeros((bs, 64), dtype=torch.int32, device=_DEVICE)
    node_pos = torch.ones_like(node_slot)
    q_store = torch.rand((bs, 3, nc), device=_DEVICE)
    qid_store = torch.randint(0, 100, (bs, 3, nc), dtype=torch.int32, device=_DEVICE)
    slot_tab = torch.full((bs + 1, nodes), 7, dtype=torch.int32, device=_DEVICE)
    pos_tab = torch.zeros_like(slot_tab)
    q_tab = torch.zeros((bs + 1, 3, nc), device=_DEVICE)
    qid_tab = torch.zeros((bs + 1, 3, nc), dtype=torch.int32, device=_DEVICE)
    rows = torch.arange(1, bs + 1, device=_DEVICE)
    eagle_harness_publish_draft_q(
        ranked, node_slot, node_pos, q_store, qid_store, rows, q_tab, qid_tab,
        slot_tab, pos_tab, _count(bs), _count(5),
    )  # fmt: skip
    assert (slot_tab[1:, 0] == -1).all()  # the root
    assert (slot_tab[1:, 1:5] == 0).all()
    assert (slot_tab[1:, 5:] == -1).all()  # past the verify width
    torch.testing.assert_close(q_tab[1:], q_store, rtol=0, atol=0)


def test_vanished_residual_takes_the_rows_first_argmax() -> None:
    """A sampled row whose residual vanishes takes torch.argmax of its target
    row as the bonus, although the rejection already lowered that entry: the
    drawn child (p = 0) is rejected, and q equal to p on the other candidates
    leaves no residual. The target ties at tokens 50 and 20 (q position 0 and
    1): the lower token wins."""
    bs, vocab, nc = 64, 300, 16
    ids = torch.arange(nc, dtype=torch.int32, device=_DEVICE)
    ids[:4] = torch.tensor([50, 20, 80, 9], dtype=torch.int32)
    q = torch.zeros(nc, device=_DEVICE)
    q[:4] = torch.tensor([0.375, 0.375, 0.25 - 3.0e-8, 3.0e-8])

    def rows(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=_DEVICE).repeat(bs, 1)

    # A root (token 7) and one child (token 9) drawn from q (slot 0, position 3).
    tokens, parents = rows([7, 9]), rows([-1, 0])
    q_tables = (rows([-1, 0]), rows([0, 3]), q.repeat(bs, 1, 1), ids.repeat(bs, 1, 1))
    probas = torch.zeros((bs * 2, vocab), device=_DEVICE)
    probas[:, [50, 20, 80]] = torch.tensor([0.375, 0.375, 0.25], device=_DEVICE)
    coins = torch.rand((bs, 2), device=_DEVICE)
    # Final coins below 0.99: the bonus row's 3e-8 residual left on token 80 is never drawn.
    predicts, accept_length = _verify_sampled(
        probas, tokens, parents, q_tables, coins, torch.rand(bs, device=_DEVICE) * 0.99
    )
    assert (accept_length == 1).all() and (predicts[:, 0] == 20).all()
    # The root's row: its argmax set to 1 over the vanished residual.
    assert (probas[0::2, 20] == 1.0).all() and (probas[0::2, 50] < 1e-6).all()


@pytest.mark.parametrize("vocab", [151936, 40001])
def test_residual_bonus_inverts_the_cdf_of_a_wide_row(vocab: int) -> None:
    """Wide rows stage each thread's segment in several runs: the bonus is the
    token whose CDF interval holds the coin (aimed at interval middles, clear
    of rounding), over an unnormalized row."""
    bs, n, positive = 16, 4, 64
    gen = torch.Generator().manual_seed(vocab)
    probas = torch.zeros((bs * n, vocab))
    idx = torch.stack(
        [torch.randperm(vocab, generator=gen)[:positive] for _ in range(bs * n)]
    )
    probas.scatter_(1, idx, torch.rand((bs * n, positive), generator=gen) + 0.5)
    last_row = torch.arange(bs) * n + torch.randint(0, n, (bs,), generator=gen)
    expected = idx[last_row, torch.randint(0, positive, (bs,), generator=gen)]
    rows = probas[last_row].double()
    mass = rows.gather(1, expected[:, None]).squeeze(1)
    below = rows.cumsum(-1).gather(1, expected[:, None]).squeeze(1) - mass
    coins = ((below + 0.5 * mass) / rows.sum(-1)).float()
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=_DEVICE)
    eagle_harness_residual_sample(
        probas.to(_DEVICE), last_row.to(_DEVICE), coins.to(_DEVICE),
        _no_top_k(bs * n), torch.full((bs * n,), -1, dtype=torch.int64, device=_DEVICE),
        torch.full((bs,), n, dtype=torch.int32, device=_DEVICE), predicts, n,
    )  # fmt: skip
    assert torch.equal(predicts.view(bs, n)[:, -1].cpu().long(), expected)
