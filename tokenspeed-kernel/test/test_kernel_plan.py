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

"""Real registry planning and numerical execution, without GPU dependencies."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk, plan_dsa_topk
from tokenspeed_kernel.ops.attention.mla import (
    mla_project_value,
    plan_mla_project_value,
)
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import (
    KernelPlan,
    NoKernelFoundError,
    kernel_override,
)
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

pytestmark = pytest.mark.usefixtures("fresh_registry")
SIGNATURE = format_signature(a=dense_tensor_format(torch.float32))


def _register(name, *, priority, invariant, traits):
    # A row-local reduction whose order is independent of the number of rows.
    def reduce_rows(x):
        output = torch.zeros(x.shape[0], dtype=x.dtype)
        for column in range(x.shape[1]):
            output = output + x[:, column]
        return output

    spec = KernelSpec(
        name=name,
        family="test_plan",
        mode="reduce",
        solution=name,
        format_signatures=frozenset({SIGNATURE}),
        priority=priority,
        features=frozenset({"batch_invariant"}) if invariant else frozenset(),
        traits=traits,
    )
    KernelRegistry.get().register(spec, reduce_rows)


def _plan(platform):
    return KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4},
        variable_traits=frozenset({"m"}),
        solution=None,
        declared_traits=frozenset(),
        ignore_layout=False,
        platform=platform,
    )


def test_plan_survives_registry_changes_and_batch_changes(h100_platform):
    _register("fixed", priority=1, invariant=True, traits={})
    plan = _plan(h100_platform)
    _register("later_faster", priority=19, invariant=True, traits={})
    x = torch.tensor([[1e10, -1e10, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    for count in (1, 2, 17):
        inputs = x.repeat(count, 1)
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"m": inputs.shape[0], "k": 4},
            features=None,
            solution=None,
            override=None,
        )
        selected = plan.kernel
        assert selected.name == "fixed"
        assert torch.equal(selected(inputs), torch.tensor([4.0, 10.0]).repeat(count))


def test_plans_compare_by_selected_kernel(h100_platform):
    _register("fixed", priority=1, invariant=True, traits={})
    first = _plan(h100_platform)
    assert _plan(h100_platform) == first
    _register("later_faster", priority=19, invariant=True, traits={})
    assert _plan(h100_platform) != first


@pytest.mark.parametrize(
    "traits", [{name: frozenset({16})} for name in ("m", "m_min", "m_max", "m_align")]
)
def test_plan_rejects_batch_limited_provider(h100_platform, traits):
    _register("limited", priority=19, invariant=True, traits=traits)
    with pytest.raises(NoKernelFoundError):
        _plan(h100_platform)


def test_plan_requires_declared_invariance(h100_platform):
    _register("ordinary", priority=19, invariant=False, traits={})
    with pytest.raises(NoKernelFoundError):
        _plan(h100_platform)


def test_plan_checks_geometry_and_does_not_follow_new_override(h100_platform):
    _register("fixed", priority=1, invariant=True, traits={})
    plan = _plan(h100_platform)
    with pytest.raises(ValueError, match="static trait k changed"):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"m": 1, "k": 8},
            features=None,
            solution=None,
            override=None,
        )
    with kernel_override("test_plan", "reduce", "another"):
        with pytest.raises(ValueError, match="change override"):
            plan.validate(
                "test_plan",
                "reduce",
                SIGNATURE,
                traits={"m": 1, "k": 4},
                features=None,
                solution=None,
                override=None,
            )


@pytest.mark.parametrize("names", [("z", "a"), ("a", "z")])
def test_ties_do_not_depend_on_registration_order(h100_platform, names):
    for name in names:
        _register(name, priority=1, invariant=True, traits={})
    assert _plan(h100_platform).kernel.name == "a"


def test_initial_override_cannot_bypass_capability(h100_platform):
    _register("ordinary", priority=19, invariant=False, traits={})
    _register("fixed", priority=1, invariant=True, traits={})
    with kernel_override("test_plan", "reduce", "ordinary"):
        with pytest.raises(NoKernelFoundError):
            _plan(h100_platform)


def test_plan_rejects_changed_format(h100_platform):
    _register("fixed", priority=1, invariant=True, traits={})
    plan = _plan(h100_platform)
    signature = format_signature(a=dense_tensor_format(torch.bfloat16))
    with pytest.raises(ValueError, match="Input contract changed"):
        plan.validate(
            "test_plan",
            "reduce",
            signature,
            traits={"k": 4, "m": 1},
            features=None,
            solution=None,
            override=None,
        )


def test_plan_checks_every_value_in_a_static_domain(h100_platform):
    _register("both", priority=1, invariant=True, traits={"q_len": frozenset({1, 2})})
    plan = KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4, "q_len": frozenset({1, 2})},
        variable_traits=frozenset({"m"}),
        solution=None,
        declared_traits=frozenset(),
        ignore_layout=False,
        platform=h100_platform,
    )
    for q_len in (1, 2):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": 3, "q_len": q_len},
            features=None,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match="static trait q_len changed"):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": 3, "q_len": 3},
            features=None,
            solution=None,
            override=None,
        )


def test_plan_does_not_infer_a_semantic_promise_from_an_absent_trait(h100_platform):
    _register("silent", priority=19, invariant=True, traits={})
    _register(
        "ordered",
        priority=1,
        invariant=True,
        traits={"slot_order": frozenset({"sorted"})},
    )
    plan = KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4, "slot_order": "sorted"},
        variable_traits=frozenset({"m"}),
        declared_traits=frozenset({"slot_order"}),
        solution=None,
        ignore_layout=False,
        platform=h100_platform,
    )
    assert plan.kernel.name == "ordered"


@pytest.mark.parametrize(
    "dimension,semantic", [("topk", "topk_layout"), ("m", "m_layout")]
)
def test_variable_dimension_does_not_reject_static_layout(
    h100_platform, dimension, semantic
):
    _register(
        "ordered",
        priority=1,
        invariant=True,
        traits={semantic: frozenset({"global_slots"})},
    )
    plan = KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4, semantic: "global_slots"},
        variable_traits=frozenset({"m", dimension}),
        declared_traits=frozenset({semantic}),
        solution=None,
        ignore_layout=False,
        platform=h100_platform,
    )
    for rows in (1, 7):
        inputs = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": rows, dimension: rows, semantic: "global_slots"},
            features=None,
            solution=None,
            override=None,
        )
        assert torch.equal(plan.kernel(inputs), inputs.sum(dim=1))
    with pytest.raises(ValueError, match="static trait"):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": 1, dimension: 1, semantic: "different"},
            features=None,
            solution=None,
            override=None,
        )


def _register_decode_topk(name, *, priority, traits):
    # Scores every causal position in a fixed order and emits the chosen
    # positions in ascending order (slot_order="sorted").
    def decode_topk(
        *, q, weights, seq_lens, block_table, page_size, topk, q_len_per_req, **kwargs
    ):
        index_k_cache = kwargs["index_k_cache"]
        slots = torch.full((q.shape[0], topk), -1, dtype=torch.int32)
        lens = torch.zeros(q.shape[0], dtype=torch.int32)
        for token in range(q.shape[0]):
            request = token // q_len_per_req
            bound = int(seq_lens[request]) - q_len_per_req + 1 + token % q_len_per_req
            positions = torch.arange(bound)
            rows = (
                block_table[request, positions // page_size].long() * page_size
                + positions % page_size
            )
            logits = q[token].float() @ index_k_cache[rows].float().T
            scores = (logits.relu() * weights[token, :, None]).sum(0)
            chosen = torch.topk(scores, min(topk, bound)).indices.sort().values
            slots[token, : len(chosen)] = rows[chosen].to(torch.int32)
            lens[token] = len(chosen)
        return slots, lens

    KernelRegistry.get().register(
        KernelSpec(
            name=name,
            family="attention",
            mode="dsa_decode_topk",
            solution=name,
            format_signatures=frozenset(
                {
                    format_signature(
                        q=dense_tensor_format(torch.bfloat16),
                        weights=dense_tensor_format(torch.float32),
                    )
                }
            ),
            priority=priority,
            features=frozenset({"batch_invariant"}),
            traits=traits,
        ),
        decode_topk,
    )


def _plan_decode_topk(**overrides):
    kwargs = dict(
        mode="dsa_decode_topk",
        index_heads=2,
        head_dim=8,
        topk=3,
        page_size=4,
        q_dtype=torch.bfloat16,
        weights_dtype=torch.float32,
        index_k_format="bf16",
        slot_order="sorted",
        forced_initial_local=False,
        q_lens=frozenset({1, 2}),
        workspace_rows=False,
    )
    return plan_dsa_topk(**{**kwargs, **overrides})


def _decode_topk(plan, rows, *, topk=3):
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(4, 2, 8, generator=generator).to(torch.bfloat16)
    weights = torch.rand(4, 2, generator=generator)
    index_k_cache = torch.randn(16, 8, generator=generator).to(torch.bfloat16)
    # Two requests of two query tokens each, on non-contiguous pages.
    block_table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([7, 6], dtype=torch.int32)
    tokens = torch.cat([torch.arange(2 * r, 2 * r + 2) for r in rows])
    return dsa_decode_topk(
        q[tokens],
        weights[tokens],
        seq_lens[rows],
        block_table[rows],
        page_size=4,
        topk=topk,
        softmax_scale=1.0,
        batch_invariant=True,
        q_len_per_req=2,
        index_k_cache=index_k_cache,
        slot_order="sorted",
        kernel_plan=plan,
    )


def test_dsa_topk_plan_keeps_its_provider_across_batches():
    sorted_trait = {"slot_order": frozenset({"sorted"}), "q_len": frozenset({1, 2})}
    _register_decode_topk("fixed", priority=1, traits=sorted_trait)
    plan = _plan_decode_topk()
    assert plan.kernel.name == "fixed"
    # Execution reuses the saved callable; nothing is left to reselect.
    KernelRegistry.get()._unregister("fixed")
    slots, lens = _decode_topk(plan, [0, 1])
    for request in (0, 1):
        solo_slots, solo_lens = _decode_topk(plan, [request])
        rows = slice(2 * request, 2 * request + 2)
        assert torch.equal(solo_slots, slots[rows])
        assert torch.equal(solo_lens, lens[rows])
    with pytest.raises(ValueError, match="static trait topk changed"):
        _decode_topk(plan, [0, 1], topk=2)


@pytest.mark.parametrize(
    "traits",
    [
        # No slot-order declaration, so the sorted promise is not inferred.
        {"q_len": frozenset({1, 2})},
        {"slot_order": frozenset({"sorted"}), "q_len": frozenset({1})},
        # Bounded in a dimension that varies with the batch.
        {"slot_order": frozenset({"sorted"}), "max_pages": frozenset({2})},
    ],
)
def test_dsa_topk_plan_rejects_a_provider_outside_the_contract(traits):
    _register_decode_topk("limited", priority=19, traits=traits)
    with pytest.raises(NoKernelFoundError):
        _plan_decode_topk()


def test_dsa_topk_plan_checks_the_mode_domain():
    with pytest.raises(ValueError, match="nonempty q_lens"):
        _plan_decode_topk(q_lens=frozenset())
    with pytest.raises(ValueError, match="prefill top-k input"):
        _plan_decode_topk(workspace_rows=True)


def _register_project_value(name, *, priority, invariant):
    # Per-head FP32 accumulation over the latent dimension in a fixed order.
    def project_value(*, attention, weight, gate, out):
        if not invariant:
            raise AssertionError("a planned projection must not run this leaf")
        heads, latent_dim, value_dim = weight.shape
        result = torch.zeros(attention.shape[0], heads, value_dim)
        for k in range(latent_dim):
            result += attention[:, :, k, None].float() * weight[None, :, k].float()
        out.copy_(result.to(attention.dtype).flatten(1))
        return out

    KernelRegistry.get().register(
        KernelSpec(
            name=name,
            family="attention",
            mode="mla_project_value",
            solution=name,
            format_signatures=format_signatures(
                ("attention", "weight", "out"), "dense", {torch.bfloat16}
            ),
            priority=priority,
            features=frozenset({"batch_invariant"}) if invariant else frozenset(),
            traits={"gate_kind": frozenset({"none"})},
        ),
        project_value,
    )


def _plan_project_value():
    return plan_mla_project_value(
        num_heads=2,
        latent_dim=8,
        value_dim=4,
        dtype=torch.bfloat16,
        gate_kind="none",
        batch_invariant=True,
    )


def test_mla_value_projection_plan_keeps_its_provider_across_batches():
    _register_project_value("ordinary", priority=19, invariant=False)
    _register_project_value("fixed", priority=1, invariant=True)
    plan = _plan_project_value()
    assert plan.kernel.name == "fixed"
    KernelRegistry.get()._unregister("fixed")
    # Small integers keep every FP32 sum exact, whatever the reference's order.
    attention = (torch.arange(5 * 2 * 8).reshape(5, 2, 8) % 5 - 2).to(torch.bfloat16)
    weight = (torch.arange(2 * 8 * 4).reshape(2, 8, 4) % 3 - 1).to(torch.bfloat16)
    expected = torch.einsum("bhk,hkv->bhv", attention.float(), weight.float())
    full = mla_project_value(attention, weight, plan=plan)
    torch.testing.assert_close(
        full, expected.to(torch.bfloat16).flatten(1), rtol=0, atol=0
    )
    for rows in (slice(1, 2), slice(2, 5)):
        assert torch.equal(
            mla_project_value(attention[rows], weight, plan=plan), full[rows]
        )
    with pytest.raises(ValueError, match="static trait"):
        mla_project_value(attention[:, :, :4], weight[:, :4], plan=plan)


def test_mla_value_projection_plan_requires_declared_invariance():
    _register_project_value("ordinary", priority=19, invariant=False)
    with pytest.raises(NoKernelFoundError):
        _plan_project_value()
