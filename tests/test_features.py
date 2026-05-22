"""Tests for static code feature extraction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    SUSPICIOUS_KEYWORDS,
    extract_features_dataframe,
    extract_features_from_code,
)

EXPECTED_COLUMNS = [
    "char_count",
    "line_count",
    "avg_line_length",
    "function_call_count",
    "loop_count",
    "conditional_count",
    "try_except_count",
    "comment_line_count",
    "suspicious_keyword_count",
]


def test_empty_code_returns_zero_features() -> None:
    features = extract_features_from_code("")

    assert set(features) == set(EXPECTED_COLUMNS)
    assert features["char_count"] == 0
    assert features["line_count"] == 0
    assert features["avg_line_length"] == 0.0
    assert features["function_call_count"] == 0
    assert features["suspicious_keyword_count"] == 0


def test_counts_basic_size_line_and_comment_features() -> None:
    code = "# comment\ndef add(a, b):\n    return a + b\n"
    features = extract_features_from_code(code)

    assert features["char_count"] == len(code)
    assert features["line_count"] == 3
    assert features["avg_line_length"] > 0
    assert features["comment_line_count"] == 1


def test_counts_function_calls_without_counting_definitions() -> None:
    code = "def run(x):\n    print(eval(x))\n    result = helper(x)\n"
    features = extract_features_from_code(code)

    assert features["function_call_count"] == 3


def test_counts_control_flow_features() -> None:
    code = """
try:
    for item in values:
        if item:
            process(item)
except ValueError:
    while retry:
        retry = False
"""
    features = extract_features_from_code(code)

    assert features["loop_count"] == 2
    assert features["conditional_count"] == 1
    assert features["try_except_count"] == 2


def test_counts_suspicious_keywords_and_shell_true_variants() -> None:
    code = """
import pickle
import subprocess
os.system(command)
subprocess.run(command, shell = True)
value = input()
return eval(value)
"""
    features = extract_features_from_code(code)

    assert "shell=True" in SUSPICIOUS_KEYWORDS
    assert features["suspicious_keyword_count"] >= 5


def test_extract_features_dataframe_uses_default_code_column() -> None:
    df = pd.DataFrame({"code": ["print('hello')", "for i in range(3): print(i)"]})
    features = extract_features_dataframe(df)

    assert list(features.columns) == EXPECTED_COLUMNS
    assert len(features) == 2
    assert features.loc[1, "loop_count"] == 1


def test_extract_features_dataframe_supports_custom_code_column() -> None:
    df = pd.DataFrame({"snippet": ["if ready:\n    run()"]})
    features = extract_features_dataframe(df, code_column="snippet")

    assert features.loc[0, "conditional_count"] == 1
    assert features.loc[0, "function_call_count"] == 1


def test_extract_features_dataframe_raises_for_missing_code_column() -> None:
    df = pd.DataFrame({"text": ["print('hello')"]})

    with pytest.raises(KeyError, match="Missing required code column"):
        extract_features_dataframe(df)


def test_control_flow_and_suspicious_keywords_ignore_comments_and_strings() -> None:
    code = """
# for if while try except eval os.system
message = "for if while eval os.system"
text = 'try except shell=True input'
/* switch case catch finally subprocess */
real = safe_value
"""
    features = extract_features_from_code(code)

    assert features["loop_count"] == 0
    assert features["conditional_count"] == 0
    assert features["try_except_count"] == 0
    assert features["suspicious_keyword_count"] == 0
    assert features["function_call_count"] == 0
