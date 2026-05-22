"""Static feature extraction for source-code snippets.

The functions in this module intentionally use lightweight text-based heuristics.
They are useful for exploratory analysis, simple baselines, and explanatory notes,
but they are not a replacement for a language-aware parser or static analyzer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "eval",
    "exec",
    "pickle",
    "subprocess",
    "os.system",
    "strcpy",
    "sprintf",
    "gets",
    "input",
    "shell=True",
)

_FEATURE_COLUMNS: tuple[str, ...] = (
    "char_count",
    "line_count",
    "avg_line_length",
    "function_call_count",
    "loop_count",
    "conditional_count",
    "try_except_count",
    "comment_line_count",
    "suspicious_keyword_count",
)

# Match common function or method invocation forms such as ``foo(``,
# ``obj.method(``, and ``package.module.fn(``.
_FUNCTION_CALL_PATTERN = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(")

# Control-flow keywords can appear before parentheses in many languages, but they
# should not be counted as function calls.
_NON_CALL_NAMES: frozenset[str] = frozenset(
    {
        "if",
        "elif",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "catch",
        "try",
        "except",
        "finally",
        "with",
        "class",
        "def",
        "return",
    }
)

_SUSPICIOUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\beval\b", re.IGNORECASE),
    re.compile(r"\bexec\b", re.IGNORECASE),
    re.compile(r"\bpickle\b", re.IGNORECASE),
    re.compile(r"\bsubprocess\b", re.IGNORECASE),
    re.compile(r"\bos\.system\b", re.IGNORECASE),
    re.compile(r"\bstrcpy\b", re.IGNORECASE),
    re.compile(r"\bsprintf\b", re.IGNORECASE),
    re.compile(r"\bgets\b", re.IGNORECASE),
    re.compile(r"\binput\b", re.IGNORECASE),
    re.compile(r"\bshell\s*=\s*true\b", re.IGNORECASE),
)


def _safe_code(code: Any) -> str:
    """Return ``code`` as a string, treating missing values as an empty string."""
    if code is None or pd.isna(code):
        return ""
    return str(code)


def _line_values(code: str) -> list[str]:
    """Split code into lines while returning no lines for an empty snippet."""
    return code.splitlines() if code else []


def _strip_strings_and_comments(code: str) -> str:
    """Return code with common comments and string literals removed."""
    text = _safe_code(code)
    text = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', " ", text)
    text = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", " ", text)
    text = re.sub(r"/\*[\s\S]*?\*/", " ", text)
    text = re.sub(r"//.*", " ", text)
    text = re.sub(r"#.*", " ", text)
    return text


def _is_definition_call(match: re.Match[str], code: str) -> bool:
    """Return True when a call-like token is part of a function definition."""
    prefix = code[max(0, match.start() - 12) : match.start()].lower()
    return bool(re.search(r"\b(def|function)\s+$", prefix))


def _count_function_calls(code: str) -> int:
    """Count call-like tokens without counting definitions or control blocks."""
    count = 0
    for match in _FUNCTION_CALL_PATTERN.finditer(code):
        token = match.group(0).strip()[:-1].strip()
        function_name = token.split(".")[-1].lower()
        if function_name in _NON_CALL_NAMES or _is_definition_call(match, code):
            continue
        count += 1
    return count


def count_suspicious_keywords(code: str) -> int:
    """Count occurrences of risky keywords and API patterns in executable code."""
    searchable_text = _strip_strings_and_comments(code)
    return sum(len(pattern.findall(searchable_text)) for pattern in _SUSPICIOUS_PATTERNS)


def extract_features_from_code(code: str) -> dict[str, int | float]:
    """Extract static features from one source-code snippet.

    Parameters
    ----------
    code:
        Source-code text. Empty strings are valid and return zero-valued
        features.

    Returns
    -------
    dict[str, int | float]
        A dictionary containing character, line, control-flow, comment, function
        call, and suspicious-keyword counts.
    """
    text = _safe_code(code)
    lines = _line_values(text)
    line_count = len(lines)
    total_line_length = sum(len(line) for line in lines)

    stripped_lines = [line.strip() for line in lines]
    comment_line_count = sum(
        1
        for line in stripped_lines
        if line.startswith(("#", "//", "/*", "*", "--"))
    )

    searchable_text = _strip_strings_and_comments(text)

    features: Mapping[str, int | float] = {
        "char_count": len(text),
        "line_count": line_count,
        "avg_line_length": total_line_length / line_count if line_count else 0.0,
        "function_call_count": _count_function_calls(searchable_text),
        "loop_count": len(
            re.findall(r"\b(for|while)\b", searchable_text, flags=re.IGNORECASE)
        ),
        "conditional_count": len(
            re.findall(
                r"\b(if|elif|else|switch|case)\b",
                searchable_text,
                flags=re.IGNORECASE,
            )
        ),
        "try_except_count": len(
            re.findall(
                r"\b(try|except|catch|finally)\b",
                searchable_text,
                flags=re.IGNORECASE,
            )
        ),
        "comment_line_count": comment_line_count,
        "suspicious_keyword_count": count_suspicious_keywords(searchable_text),
    }
    return dict(features)


def extract_features_dataframe(
    df: pd.DataFrame,
    code_column: str = "code",
) -> pd.DataFrame:
    """Extract static code features for every row in a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame containing a source-code column.
    code_column:
        Name of the column that contains code snippets. Defaults to ``"code"``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per input row and the feature columns:
        ``char_count``, ``line_count``, ``avg_line_length``,
        ``function_call_count``, ``loop_count``, ``conditional_count``,
        ``try_except_count``, ``comment_line_count``, and
        ``suspicious_keyword_count``.

    Raises
    ------
    KeyError
        If ``code_column`` is not present in ``df``.
    """
    if code_column not in df.columns:
        raise KeyError(f"Missing required code column: {code_column}")

    features = df[code_column].apply(extract_features_from_code)
    return pd.DataFrame(features.tolist(), index=df.index, columns=_FEATURE_COLUMNS)
