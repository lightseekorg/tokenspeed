from __future__ import annotations

import ast
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODEL_EXECUTOR = _ROOT / "python/tokenspeed/runtime/execution/model_executor.py"
_INPUT_BUFFER = _ROOT / "python/tokenspeed/runtime/execution/input_buffer.py"
_EAGLE = _ROOT / "python/tokenspeed/runtime/execution/drafter/eagle.py"
_DFLASH = _ROOT / "python/tokenspeed/runtime/execution/drafter/dflash.py"


def _class_method(path: pathlib.Path, class_name: str, method_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    segment = ast.get_source_segment(source, method)
    assert segment is not None
    return segment


class V4GroupKeyedCacheLocWiringTest(unittest.TestCase):
    def test_model_executor_does_not_select_first_full_history_group(self):
        source = _MODEL_EXECUTOR.read_text()
        init = _class_method(_MODEL_EXECUTOR, "ModelExecutor", "__init__")
        update = _class_method(_MODEL_EXECUTOR, "ModelExecutor", "update_block_table")

        self.assertNotIn("_flat_full_group_id", source)
        self.assertIn("resolve_deepseek_v4_flat_loc_policy", init)
        self.assertIn("legacy_flat_loc_group_id", init)
        self.assertIn("self._uses_group_keyed_cache_locs", update)
        self.assertIn("self.input_buffers.flat_block_table_staging", update)
        self.assertIn("validate_forward_op_schema", update)
        self.assertNotIn('getattr(forward_op, "flat_block_tables"', update)
        self.assertIn("return", update)

    def test_input_and_eagle_emit_only_dummy_scalar_locs_for_v4_flat(self):
        init = _class_method(_INPUT_BUFFER, "InputBuffers", "__init__")
        fill = _class_method(_INPUT_BUFFER, "InputBuffers", "fill_input_buffers")
        eagle = _class_method(_EAGLE, "Eagle", "_run_multi_step_decode")

        self.assertIn("uses_group_keyed_cache_locs", init)
        self.assertIn("self.out_cache_loc_buf[:total_tokens].fill_", fill)
        self.assertIn("self.input_buffers.uses_group_keyed_cache_locs", eagle)
        self.assertIn("cache_locs.fill_", eagle)

    def test_dflash_rejects_group_keyed_page_domains(self):
        init_buffers = _class_method(_DFLASH, "DFlash", "_init_native_buffers")

        self.assertIn("uses_group_keyed_cache_locs", init_buffers)
        self.assertIn("does not support group-keyed flat cache locations", init_buffers)


if __name__ == "__main__":
    unittest.main()
