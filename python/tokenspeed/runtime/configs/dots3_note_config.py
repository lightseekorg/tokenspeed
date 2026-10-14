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

"""Flat checkpoint configuration; cache geometry is validated by the recipe."""

from transformers import PretrainedConfig


class Dots3NoteConfig(PretrainedConfig):
    model_type = "dots3_note"

    def __init__(
        self,
        n_group=1,
        topk_group=1,
        num_nextn_predict_layers=1,
        mtp_layer_offset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_group = n_group
        self.topk_group = topk_group

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_layer_offset = mtp_layer_offset

    def for_mtp(self):
        """Derive the checkpoint's single dense SWA draft without changing the target."""
        if self.num_nextn_predict_layers != 1:
            raise ValueError("dots3-note supports one MTP layer")
        config = self.to_dict()
        config.update(
            mtp_layer_offset=self.num_hidden_layers,
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
        )
        for name in (
            "num_attention_heads",
            "num_key_value_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "rope_theta",
            "attention_gate_type",
        ):
            config[name] = config[f"swa_{name}"]
        return type(self)(**config)
