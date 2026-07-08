from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _routing_configs(relpath: str) -> list[dict[str, ast.AST]]:
    tree = ast.parse((REPO_ROOT / relpath).read_text(), filename=relpath)
    configs: list[dict[str, ast.AST]] = []
    for scope in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        dict_assignments: dict[str, ast.Dict] = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        dict_assignments[target.id] = node.value
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node.func) not in {"MoELayer", "_MoELayer"}:
                continue
            for keyword in node.keywords:
                if keyword.arg != "routing_config":
                    continue
                value = keyword.value
                if isinstance(value, ast.Name):
                    value = dict_assignments.get(value.id)
                if not isinstance(value, ast.Dict):
                    continue
                entries = {
                    key.value: item
                    for key, item in zip(value.keys, value.values)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                configs.append(entries)
    return configs


def _normalize_expr(relpath: str) -> str:
    matches = [
        ast.unparse(config["normalize_topk_weights"])
        for config in _routing_configs(relpath)
        if "normalize_topk_weights" in config
    ]
    assert len(matches) == 1
    return matches[0]


def test_config_driven_moe_models_propagate_topk_normalization_flag() -> None:
    for relpath in (
        "python/tokenspeed/runtime/models/deepseek_v3.py",
        "python/tokenspeed/runtime/models/deepseek_v4.py",
        "python/tokenspeed/runtime/models/longcat_flash.py",
        "python/tokenspeed/runtime/models/qwen3_5_moe.py",
    ):
        assert _normalize_expr(relpath) == "config.norm_topk_prob"


def test_minimax_moe_routing_matches_hardcoded_topk_normalization() -> None:
    assert _normalize_expr("python/tokenspeed/runtime/models/minimax_m2.py") == "True"
