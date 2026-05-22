"""Tests for preprocessing utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_preprocessing import (
    MAX_CODE_LENGTH,
    clean_dataset,
    normalize_columns,
    preprocess_dataframe,
    save_splits,
    split_dataset,
)


def test_normalize_columns_renames_inputs() -> None:
    raw = pd.DataFrame({"snippet": ["print('ok')"], "target": [0]})

    normalized = normalize_columns(raw, code_column="snippet", label_column="target")

    assert list(normalized.columns) == ["code", "label"]


def test_normalize_columns_infers_codexglue_schema() -> None:
    raw = pd.DataFrame({"func": ["int main() { return 0; }"], "target": [0]})

    normalized = normalize_columns(raw)

    assert list(normalized.columns) == ["code", "label"]
    assert normalized.loc[0, "code"] == "int main() { return 0; }"


def test_normalize_columns_rejects_missing_explicit_code_column() -> None:
    raw = pd.DataFrame({"func": ["int main() { return 0; }"], "target": [0]})

    with pytest.raises(KeyError, match="missing_code"):
        normalize_columns(raw, code_column="missing_code", label_column="target")


def test_normalize_columns_rejects_missing_explicit_label_column() -> None:
    raw = pd.DataFrame({"func": ["int main() { return 0; }"], "target": [0]})

    with pytest.raises(KeyError, match="missing_label"):
        normalize_columns(raw, code_column="func", label_column="missing_label")


def test_clean_dataset_removes_missing_values() -> None:
    raw = pd.DataFrame(
        {
            "code": ["def valid_function(): return 1", None, "def also_valid(): return 2"],
            "label": [0, 1, None],
        }
    )

    cleaned = clean_dataset(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "code"] == "def valid_function(): return 1"


def test_clean_dataset_removes_duplicate_code_snippets_with_matching_labels() -> None:
    raw = pd.DataFrame(
        {
            "code": [
                "def duplicate(): return 1",
                "def duplicate(): return 1",
                "def unique_function(): return 2",
            ],
            "label": [0, 0, 1],
        }
    )

    cleaned = clean_dataset(raw)

    assert cleaned["code"].tolist() == [
        "def duplicate(): return 1",
        "def unique_function(): return 2",
    ]
    assert cleaned["label"].tolist() == [0, 1]


def test_clean_dataset_rejects_duplicate_code_snippets_with_conflicting_labels() -> None:
    raw = pd.DataFrame(
        {
            "code": [
                "def duplicate(): return 1",
                "def duplicate(): return 1",
                "def unique_function(): return 2",
            ],
            "label": [0, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="Conflicting labels found"):
        clean_dataset(raw)


def test_clean_dataset_removes_short_snippets() -> None:
    raw = pd.DataFrame(
        {
            "code": ["x = 1", "def long_enough(): return 1"],
            "label": [0, 1],
        }
    )

    cleaned = clean_dataset(raw)

    assert cleaned["code"].tolist() == ["def long_enough(): return 1"]


def test_clean_dataset_rejects_max_length_smaller_than_min_length() -> None:
    raw = pd.DataFrame(
        {
            "code": ["def long_enough(): return 1"],
            "label": [0],
        }
    )

    with pytest.raises(ValueError, match="max_code_length must be greater than or equal"):
        clean_dataset(raw, min_code_length=20, max_code_length=10)


def test_clean_dataset_strips_truncates_and_converts_labels() -> None:
    long_code = "  " + "a" * (MAX_CODE_LENGTH + 25) + "  "
    raw = pd.DataFrame(
        {
            "code": [long_code, "def risky(): return eval(user)"],
            "label": ["clean", "buggy"],
        }
    )

    cleaned = clean_dataset(raw)

    assert len(cleaned.loc[0, "code"]) == MAX_CODE_LENGTH
    assert cleaned["label"].tolist() == [0, 1]


def test_split_dataset_uses_stratification() -> None:
    dataframe = pd.DataFrame(
        {
            "code": [f"def function_{i}(): return {i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50,
        }
    )

    train, valid, test = split_dataset(dataframe)

    assert len(train) == 80
    assert len(valid) == 10
    assert len(test) == 10
    assert train["label"].value_counts().to_dict() == {0: 40, 1: 40}
    assert valid["label"].value_counts().to_dict() == {0: 5, 1: 5}
    assert test["label"].value_counts().to_dict() == {0: 5, 1: 5}


def test_split_dataset_rejects_too_few_rows_for_all_splits() -> None:
    dataframe = pd.DataFrame(
        {
            "code": ["def first(): return 1", "def second(): return 2"],
            "label": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="fewer than 3 rows"):
        split_dataset(dataframe)


def test_split_dataset_keeps_tiny_single_class_splits_non_empty() -> None:
    dataframe = pd.DataFrame(
        {
            "code": [
                "def first(): return 1",
                "def second(): return 2",
                "def third(): return 3",
            ],
            "label": [0, 0, 0],
        }
    )

    train, valid, test = split_dataset(dataframe)

    assert len(train) == 1
    assert len(valid) == 1
    assert len(test) == 1


def test_save_splits_creates_output_files(tmp_path) -> None:
    train = pd.DataFrame({"code": ["def train_fn(): return 1"], "label": [0]})
    valid = pd.DataFrame({"code": ["def valid_fn(): return 1"], "label": [1]})
    test = pd.DataFrame({"code": ["def test_fn(): return 1"], "label": [0]})

    paths = save_splits(train, valid, test, tmp_path)

    assert paths["train"].name == "train.csv"
    assert paths["valid"].name == "valid.csv"
    assert paths["test"].name == "test.csv"
    assert all(path.exists() for path in paths.values())


def test_preprocess_dataframe_saves_train_valid_test_files(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "snippet": [f"def function_{i}(): return {i}" for i in range(100)],
            "target": [0] * 50 + [1] * 50,
        }
    )

    paths = preprocess_dataframe(
        raw,
        output_dir=tmp_path,
        code_column="snippet",
        label_column="target",
    )

    assert set(paths) == {"train", "valid", "test"}
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "valid.csv").exists()
    assert (tmp_path / "test.csv").exists()

    train = pd.read_csv(tmp_path / "train.csv")
    valid = pd.read_csv(tmp_path / "valid.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    assert len(train) == 80
    assert len(valid) == 10
    assert len(test) == 10
