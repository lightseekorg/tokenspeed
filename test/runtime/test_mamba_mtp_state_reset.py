import ast
from pathlib import Path


def _calls_runtime_state_method(node: ast.AST, method_name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not isinstance(func, ast.Attribute) or func.attr != method_name:
            continue
        value = func.value
        if not isinstance(value, ast.Attribute) or value.attr != "runtime_states":
            continue
        if isinstance(value.value, ast.Name) and value.value.id == "self":
            return True
    return False


def _guard_uses_num_extends(test: ast.AST) -> bool:
    for child in ast.walk(test):
        if not isinstance(child, ast.Compare):
            continue
        if not isinstance(child.left, ast.Name) or child.left.id != "num_extends":
            continue
        if not child.ops or not isinstance(child.ops[0], ast.Gt):
            continue
        if not child.comparators:
            continue
        comparator = child.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == 0:
            return True
    return False


def _guard_uses_forward_mode_is_extend(test: ast.AST) -> bool:
    for child in ast.walk(test):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr != "is_extend":
                continue
            value = child.func.value
            if isinstance(value, ast.Name) and value.id == "forward_mode":
                return True
    return False


def test_mamba_state_reset_only_runs_for_real_prefill_extend():
    """TARGET_VERIFY satisfies ForwardMode.is_extend(), so guard with num_extends."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/execution/model_executor.py"
    )
    tree = ast.parse(source_path.read_text())

    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if _calls_runtime_state_method(node, "copy_mamba_states") or (
            _calls_runtime_state_method(node, "zero_mamba_states")
        ):
            guards.append(node.test)

    assert guards, "Expected a guarded Mamba state initialization block."
    for guard in guards:
        assert _guard_uses_num_extends(guard)
        assert not _guard_uses_forward_mode_is_extend(guard)
