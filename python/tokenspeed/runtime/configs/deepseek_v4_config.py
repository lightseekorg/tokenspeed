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


from transformers.configuration_utils import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict | None = None,
        vision_n_layers: int = 0,
        vision_dim: int = 1024,
        vision_n_heads: int = 16,
        vision_inter_dim: int = 2816,
        vision_patch_size: int = 14,
        vision_rope_theta: float = 10000.0,
        vision_downsample_ratio: int = 3,
        vision_max_n_token: int = 384,
        vision_min_pixels: int = 147456,
        vision_max_wh_ratio: int = 8,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_scaling or {}
        self.vision_n_layers = vision_n_layers
        self.vision_dim = vision_dim
        self.vision_n_heads = vision_n_heads
        self.vision_inter_dim = vision_inter_dim
        self.vision_patch_size = vision_patch_size
        self.vision_rope_theta = vision_rope_theta
        self.vision_downsample_ratio = vision_downsample_ratio
        self.vision_max_n_token = vision_max_n_token
        self.vision_min_pixels = vision_min_pixels
        self.vision_max_wh_ratio = vision_max_wh_ratio
        super().__init__(rope_scaling=rope_scaling, **kwargs)
