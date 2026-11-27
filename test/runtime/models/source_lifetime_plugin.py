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

"""A hook prepares metadata once; the ordinary cache recipe consumes it."""

import json
import os
import shutil
from pathlib import Path

from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import OrdinaryRecipe
from tokenspeed.runtime.plugins.registry import (
    register_cache_recipe,
    register_post_model_init_hook,
)


def _metadata():
    return Path(os.environ["SOURCE_TEST_METADATA"])


def initialized(context):
    # Exclusive creation rejects a repeated hook during cache rebuilds.
    with _metadata().open("x") as output:
        json.dump(
            {
                "family": context.model_config.attention_arch.name.lower(),
                "has_draft": context.draft_model_config is not None,
            },
            output,
        )
    if os.environ["SOURCE_TEST_MODE"] == "fail":
        raise RuntimeError("source lifetime hook refused startup")
    if os.environ["SOURCE_TEST_MODE"] == "retire":
        shutil.rmtree(context.model_config.model_path)


def cache_recipe(**kwargs):
    metadata = json.loads(_metadata().read_text())
    return OrdinaryRecipe(family=metadata["family"], **kwargs)


def register():
    register_post_model_init_hook("source_lifetime_fixture", initialized)
    register_cache_recipe("mha", cache_recipe, override=True)
