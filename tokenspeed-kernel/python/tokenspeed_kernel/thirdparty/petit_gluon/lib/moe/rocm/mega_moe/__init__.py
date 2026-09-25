"""Keep native mega_moe.py beside mega_moe/ while exposing its host API.

Python resolves this directory first. Load the sibling lazily so its kernel
imports can use the package without a circular import during specialization.
"""

import sys
from functools import cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


@cache
def _api():
    name = __name__ + "._api"
    spec = spec_from_file_location(name, Path(__file__).parent.parent / "mega_moe.py")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def __getattr__(name):
    return getattr(_api(), name)
