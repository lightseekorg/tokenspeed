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

"""LongCat Pro checkpoint interception within main's weight-update session."""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.longcat.over_embedding import LongCatOverEmbedding
from tokenspeed.runtime.utils.env import global_server_args_dict


class LongcatWeightState:
    """Load separate index-K/weight projections into the packed projection."""

    def __init__(self, quant_config) -> None:
        self.quant_config = quant_config
        self.pending_fp8_wk: dict[str, dict[str, torch.Tensor]] = {}
        self.loaded_shards: dict[str, set[int]] = {}
        self.initialized = False

    def reset_pairing(self) -> None:
        self.pending_fp8_wk.clear()
        self.loaded_shards.clear()

    def intercept(self, owner, weights, loaded):
        params = dict(owner.named_parameters())
        modules = dict(owner.named_modules())
        param_names = {id(param): name for name, param in params.items()}
        embed = owner.model.embed_tokens
        for name, weight in weights:
            if name.startswith(("mtp.", "model.mtp.")) or ".mtp." in name:
                continue
            name = name.removeprefix("language_model.")
            if isinstance(embed, LongCatOverEmbedding) and embed.load_weight(
                name, weight
            ):
                loaded.update(
                    f"model.embed_tokens.{key}" for key, _ in embed.named_parameters()
                )
                continue
            if ".indexer." not in name:
                yield name, weight
                continue
            if ".self_attn.1.indexer." in name:
                raise RuntimeError(
                    f"LongCat CLI reuse branch must not own indexer weight {name}"
                )
            if self._try_load_fused_indexer_projection(
                name=name,
                loaded_weight=weight,
                params_dict=params,
                modules_dict=modules,
            ):
                module_name = name.split(".indexer.", 1)[0] + ".indexer"
                fused_name = f"{module_name}.wk_weights_proj.weight"
                if fused_name in params:
                    loaded.add(fused_name)
                continue
            param = owner.get_param(params, name)
            if param is None:
                raise RuntimeError(f"LongCat indexer parameter is missing: {name}")
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(param_names[id(param)])

    def validate_update(self) -> None:
        if self.pending_fp8_wk:
            raise RuntimeError(
                f"LongCat indexer FP8 weight/scale pair is incomplete: {sorted(self.pending_fp8_wk)}"
            )

    def prepare_derivation(self, owner) -> None:
        dummy = global_server_args_dict.get("load_format") == "dummy"
        modules = dict(owner.named_modules())
        if dummy:
            for name, module in modules.items():
                if name.endswith(".indexer"):
                    module.set_wk_weights_proj_loaded(True)
        else:
            if isinstance(owner.model.embed_tokens, LongCatOverEmbedding):
                owner.model.embed_tokens.validate_loaded_weights()
            if not self.initialized:
                self._validate_fused_indexer_projections(
                    modules_dict=modules,
                )
        self.initialized = True

    def _load_fused_indexer_projection(
        self,
        module_name: str,
        shard_id: int | None,
        weight: torch.Tensor,
        params: dict[str, torch.Tensor],
        modules: dict[str, nn.Module],
    ) -> bool:
        param = params.get(f"{module_name}.wk_weights_proj.weight")
        if param is None:
            return False
        loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is None:
            loader(param, weight)
        else:
            loader(param, weight, shard_id)
        shards = self.loaded_shards.setdefault(module_name, set())
        for shard in (0, 1) if shard_id is None else (shard_id,):
            shards.add(int(shard))
            if shards == {0, 1}:
                module = modules.get(module_name)
                if module is not None:
                    module.set_wk_weights_proj_loaded()
        return True

    def _flush_fused_indexer_fp8_wk(
        self,
        *,
        module_name: str,
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
    ) -> None:
        entry = self.pending_fp8_wk.get(module_name)
        if not entry or "weight" not in entry or "scale" not in entry:
            return
        weight_block_size = getattr(self.quant_config, "weight_block_size", None)
        if weight_block_size is None:
            return
        weight_bf16 = block_dequant(
            entry["weight"],
            entry["scale"],
            list(weight_block_size),
        ).to(torch.bfloat16)
        if self._load_fused_indexer_projection(
            module_name, 0, weight_bf16, params_dict, modules_dict
        ):
            del self.pending_fp8_wk[module_name]

    def _try_load_fused_indexer_projection(
        self,
        *,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
    ) -> bool:
        if name.endswith(".indexer.wk_weights_proj.weight"):
            module_name = name.rsplit(".wk_weights_proj.weight", 1)[0]
            shard_id = None
        elif name.endswith(".indexer.weights_proj.weight"):
            module_name = name.rsplit(".weights_proj.weight", 1)[0]
            shard_id = 1
        elif ".indexer.wk." in name:
            module_name = name.rsplit(".wk.", 1)[0]
            part = None
            if name.endswith(".weight"):
                if loaded_weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                    part = "weight"
            elif "weight_scale_inv" in name:
                part = "scale"
            else:
                return False
            if part is not None:
                self.pending_fp8_wk.setdefault(module_name, {})[part] = loaded_weight
                self._flush_fused_indexer_fp8_wk(
                    module_name=module_name,
                    params_dict=params_dict,
                    modules_dict=modules_dict,
                )
                return True
            shard_id = 0
        else:
            return False

        if not self._load_fused_indexer_projection(
            module_name, shard_id, loaded_weight, params_dict, modules_dict
        ):
            raise RuntimeError(
                f"Packed DSA indexer projection is missing: {module_name}"
            )
        return True

    def _validate_fused_indexer_projections(
        self,
        *,
        modules_dict: dict[str, nn.Module],
    ) -> None:
        expected = {name for name in modules_dict if name.endswith(".indexer")}
        incomplete = {
            name for name in expected if self.loaded_shards.get(name, set()) != {0, 1}
        }
        if self.pending_fp8_wk or incomplete:
            raise RuntimeError(
                "DSA indexer packed projections are incomplete: "
                f"pending_fp8={sorted(self.pending_fp8_wk)}, "
                f"missing={sorted(incomplete)}"
            )
