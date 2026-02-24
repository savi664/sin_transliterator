# tests/test_model_registry.py
import pytest
from sin_transliterator.model_registry import resolve
from sin_transliterator.exceptions import InvalidModelError


def test_resolve_all_four_combinations():
    """All four model specs must resolve without error."""
    for model in ("transformer", "llm"):
        for code_mix in (True, False):
            spec = resolve(model, code_mix)
            assert spec.repo_id, f"Empty repo_id for model={model}, code_mix={code_mix}"


def test_resolve_code_mix_flag_is_reflected():
    base = resolve("transformer", False)
    mix = resolve("transformer", True)
    assert base.repo_id != mix.repo_id, "Base and code-mix models must have distinct repo IDs"