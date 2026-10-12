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

"""Draft-tree support: each backend node declares its own capability, and
``resolve_tree_support`` composes it over the target and draft backend trees
once at startup (docs/design/tree-speculation.md, "Scope")."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (  # noqa: E402
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (  # noqa: E402
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import (  # noqa: E402
    MHAAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import (  # noqa: E402
    CacheGroupRouter,
)
from tokenspeed.runtime.layers.attention.backends.paged.tokenspeed_mla import (  # noqa: E402
    CuteDSLMLABackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (  # noqa: E402
    TRTLLMMHAAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (  # noqa: E402
    TRTLLMMLABackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import (  # noqa: E402
    KdaAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E402
    MambaAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba2 import (  # noqa: E402
    Mamba2AttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.support import (  # noqa: E402
    resolve_tree_support,
)

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _trtllm(kv_cache_dtype=torch.bfloat16):
    leaf = object.__new__(TRTLLMMHAAttnBackend)
    leaf.kv_cache_dtype = kv_cache_dtype
    leaf.dtype = torch.bfloat16
    return leaf


def _mla(cls, kv_cache_dtype=torch.bfloat16, kvp_size=1):
    leaf = object.__new__(cls)
    leaf.data_type = kv_cache_dtype
    leaf.q_data_type = torch.bfloat16
    leaf.kvp_group = tuple(range(kvp_size))
    return leaf


def _router(*leaves, retention="full_history", entry_stride=1):
    router = object.__new__(CacheGroupRouter)
    router.leaves = {str(gid): leaf for gid, leaf in enumerate(leaves)}
    router._geometry = CacheGroupGeometry(
        row_geometry={gid: (64, entry_stride) for gid in router.leaves},
        retentions={gid: (retention, None) for gid in router.leaves},
    )
    return router


def _mamba(cls=MambaAttnBackend, replay_ssm=False):
    backend = object.__new__(cls)
    backend.replay_ssm = replay_ssm
    return backend


def _hybrid(full, linear):
    hybrid = object.__new__(HybridLinearAttnBackend)
    hybrid.full_attn_backend = full
    hybrid.linear_attn_backend = linear
    return hybrid


@pytest.mark.parametrize("kv_cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_trtllm_router_supports_verify_and_lanes(kv_cache_dtype):
    resolve_tree_support(
        _router(_trtllm(kv_cache_dtype)), _router(_trtllm(kv_cache_dtype))
    )


@pytest.mark.parametrize("cls", [TRTLLMMLABackend, CuteDSLMLABackend])
@pytest.mark.parametrize("kv_cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mla_routers_support_verify_and_lanes(cls, kv_cache_dtype):
    resolve_tree_support(
        _router(_mla(cls, kv_cache_dtype)), _router(_mla(cls, kv_cache_dtype))
    )


def test_full_history_cache_groups_verify_and_draft_trees():
    resolve_tree_support(_router(_trtllm(), _trtllm()), _router(_trtllm(), _trtllm()))


def test_gdn_target_verifies_trees():
    resolve_tree_support(_hybrid(_router(_trtllm()), _mamba()), _router(_trtllm()))


def test_replay_ssm_target_verifies_trees():
    resolve_tree_support(
        _hybrid(_router(_trtllm()), _mamba(replay_ssm=True)), _router(_trtllm())
    )


@pytest.mark.parametrize("replay_ssm", [False, True])
def test_mamba2_target_verifies_trees(replay_ssm):
    resolve_tree_support(
        _hybrid(_router(_trtllm()), _mamba(Mamba2AttnBackend, replay_ssm)),
        _router(_trtllm()),
    )


def test_draft_with_linear_layers_is_refused():
    with pytest.raises(NotImplementedError, match="draft: .*linear-attention"):
        resolve_tree_support(_router(_trtllm()), _hybrid(_router(_trtllm()), _mamba()))


@pytest.mark.parametrize(
    "target, blocker",
    [
        (lambda: _router(MHAAttnBackend.__new__(MHAAttnBackend)), "MHAAttnBackend"),
        (lambda: _router(_trtllm(torch.float8_e5m2)), "kv_cache_dtype"),
        (
            lambda: _router(_trtllm(), _trtllm(), retention="sliding_window"),
            "one row per token; 0, 1 slide",
        ),
        (
            lambda: _router(_trtllm(), entry_stride=4),
            "one row per token; 0 slide or pack",
        ),
        (lambda: _hybrid(_router(_trtllm()), _mamba(KdaAttnBackend)), "KDA"),
        (
            lambda: _router(_mla(TRTLLMMLABackend, torch.float8_e5m2)),
            "latent rows .* kv_cache_dtype",
        ),
        (
            lambda: _router(_mla(CuteDSLMLABackend, kvp_size=2)),
            "KV parallelism",
        ),
    ],
)
def test_unsupported_target_nodes_are_named(target, blocker):
    with pytest.raises(NotImplementedError, match=f"verify: .*{blocker}"):
        resolve_tree_support(target(), _router(_trtllm()))


def test_every_blocker_is_reported_at_once():
    with pytest.raises(NotImplementedError) as err:
        resolve_tree_support(
            _router(_trtllm(torch.float8_e5m2)),
            _hybrid(_router(_trtllm()), _mamba()),
        )
    assert "verify: " in str(err.value) and "draft: " in str(err.value)
