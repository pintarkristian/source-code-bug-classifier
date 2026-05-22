"""Tests for reusable prediction logic."""

from __future__ import annotations

import pytest

from src.predict import CodeBugPredictor


def test_predictor_returns_expected_response_shape_without_model(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    result = predictor.predict("def add(a, b): return a + b")

    assert result["label"] in {"clean", "buggy"}
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["model_name"].endswith("heuristic-fallback")
    assert isinstance(result["notes"], list)


def test_predictor_adds_notes_for_risky_patterns(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    result = predictor.predict("subprocess.run(cmd, shell=True); eval(user_input)")

    assert result["label"] == "buggy"
    assert any("shell=True" in note for note in result["notes"])
    assert any("eval" in note for note in result["notes"])


def test_predictor_notes_division_without_obvious_zero_check(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    result = predictor.predict("def divide(a, b): return a / b")

    assert any("division-by-zero" in note for note in result["notes"])


def test_predictor_does_not_note_division_when_non_zero_check_exists(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    code = "def divide(a, b):\n    if b != 0:\n        return a / b\n    return None"
    result = predictor.predict(code)

    assert not any("division-by-zero" in note for note in result["notes"])


def test_predictor_still_notes_division_after_zero_equality_check(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    code = "def divide(a, b):\n    if b == 0:\n        log_error('zero')\n    return a / b"
    result = predictor.predict(code)

    assert any("division-by-zero" in note for note in result["notes"])


def test_predictor_rejects_blank_code(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    with pytest.raises(ValueError, match="code must not be blank"):
        predictor.predict("   ")


def test_predictor_ignores_division_patterns_in_comments_and_strings(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    code = """
# docs: divide with a / b
message = "https://example.com/a/b"
print("division a / b in docs")
return safe_value
"""
    result = predictor.predict(code)

    assert not any("division-by-zero" in note for note in result["notes"])


def test_predictor_ignores_division_patterns_in_block_comments(tmp_path) -> None:
    predictor = CodeBugPredictor(model_dir=tmp_path / "missing-model")

    code = """
/* denominator in docs: a / b */
int value = 42;
"""
    result = predictor.predict(code)

    assert not any("division-by-zero" in note for note in result["notes"])
