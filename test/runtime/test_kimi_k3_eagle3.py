"""CPU tests pinning the Kimi-K3 EAGLE3 completed-layer capture contract."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.factory import (  # noqa: E402
    _eagle_aux_layer_ids,
)
from tokenspeed.runtime.models import kimi_k3  # noqa: E402


def _post_layer_attnres_reference(
    prefix_sum: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """K3 capture reference: ``prefix_sum + hidden_states`` after a decoder layer."""
    return prefix_sum + hidden_states


def test_capture_tensor_matches_post_layer_attnres_reference():
    """TokenSpeed's post-layer prefix_sum is the K3 capture tensor."""

    class FakeLayer:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

        def __call__(self, positions, prefix_sum, ctx, out_cache_loc, block_residual):
            return (
                _post_layer_attnres_reference(prefix_sum, self.hidden_states),
                block_residual,
            )

    model = SimpleNamespace(
        embed_tokens=lambda input_ids: input_ids.to(torch.bfloat16).unsqueeze(-1),
        config=SimpleNamespace(num_hidden_layers=4, attn_res_block_size=2),
        # Single-stage pipeline window: the forward walks
        # [pp_start_layer, pp_end_layer) and only the last stage applies
        # the output AttnRes fold.
        pp_start_layer=0,
        pp_end_layer=4,
        mapping=SimpleNamespace(is_last_pp_rank=True),
        layers=[
            FakeLayer(torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)),
            FakeLayer(torch.tensor([[3.0], [4.0]], dtype=torch.bfloat16)),
            FakeLayer(torch.tensor([[5.0], [6.0]], dtype=torch.bfloat16)),
            FakeLayer(torch.tensor([[7.0], [8.0]], dtype=torch.bfloat16)),
        ],
        output_attn_res_proj=None,
        output_attn_res_norm=None,
        norm=None,
        layers_to_capture=[],
        eagle3_layers_to_capture=(2, 3),
    )

    with mock.patch.object(
        kimi_k3, "_apply_attn_res", side_effect=lambda prefix, *args, **kwargs: prefix
    ):
        _, aux = kimi_k3.KimiLinearModel.forward(
            model,
            torch.tensor([0, 1]),
            positions=None,
            ctx=None,
            out_cache_loc=None,
        )

    # IDs are one-based completed-layer IDs.  IDs 2 and 3 select the
    # outputs of FakeLayer 1 and FakeLayer 2, not the following layer boundary.
    torch.testing.assert_close(
        aux[0], torch.tensor([[4.0], [7.0]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        aux[1], torch.tensor([[9.0], [13.0]], dtype=torch.bfloat16)
    )


def test_k3_eagle3_layer_ids_preserve_draft_config_values():
    target = object.__new__(kimi_k3.KimiLinearForCausalLM)
    torch.nn.Module.__init__(target)
    target.model = SimpleNamespace(layers=[object() for _ in range(93)])

    target.set_eagle3_layers_to_capture([2, 46, 90])

    assert target.model.eagle3_layers_to_capture == (2, 46, 90)


def test_k3_eagle3_default_and_invalid_layer_ids():
    target = object.__new__(kimi_k3.KimiLinearForCausalLM)
    torch.nn.Module.__init__(target)
    target.model = SimpleNamespace(layers=[object() for _ in range(93)])

    target.set_eagle3_layers_to_capture()
    assert target.model.eagle3_layers_to_capture == (2, 46, 90)

    with pytest.raises(ValueError, match="completed K3 layer"):
        target.set_eagle3_layers_to_capture([2, 46, 94])


def test_eagle_aux_ids_resolve_from_k3_text_config():
    draft = SimpleNamespace(
        text_config=SimpleNamespace(
            eagle_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=[2, 46, 90])
        )
    )
    assert _eagle_aux_layer_ids(draft) == [2, 46, 90]


def test_eagle_aux_ids_resolve_from_k3_text_config_dict():
    draft = {
        "text_config": {
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [2, 46, 90]}
        }
    }
    assert _eagle_aux_layer_ids(draft) == [2, 46, 90]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
