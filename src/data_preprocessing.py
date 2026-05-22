"""Data preprocessing for the AI code bug classifier.

This module loads code-classification datasets from local CSV/JSONL files or
from Hugging Face Datasets, normalizes them to ``code`` and ``label`` columns,
cleans the records, creates stratified train/validation/test splits, and saves
those splits as CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from src.config import get_settings
from src.utils import ensure_dir

MIN_CODE_LENGTH = 10
MAX_CODE_LENGTH = 4000
DEFAULT_VALID_SIZE = 0.10
DEFAULT_TEST_SIZE = 0.10
DEFAULT_RANDOM_STATE = 42

CODE_COLUMN_CANDIDATES = (
    "code",
    "func",
    "function",
    "snippet",
    "source_code",
    "source",
    "text",
    "content",
)
LABEL_COLUMN_CANDIDATES = (
    "label",
    "target",
    "buggy",
    "is_buggy",
    "defect",
    "has_bug",
    "risk",
)
POSITIVE_LABELS = {"1", "true", "buggy", "risky", "risk", "defect", "defective", "vulnerable"}
NEGATIVE_LABELS = {"0", "false", "clean", "safe", "non_buggy", "non-buggy", "benign"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset with Pandas."""
    return pd.read_csv(path)


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Load a JSON Lines dataset with Pandas."""
    return pd.read_json(path, lines=True)


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate column present in ``columns``."""
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def infer_code_and_label_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """Infer likely code and label columns from common dataset schemas.

    The CodeXGLUE defect-detection dataset commonly uses ``func`` for source
    code and ``target`` for the binary label, while local datasets often use
    ``code`` and ``label``.
    """
    code_column = _first_existing_column(dataframe.columns, CODE_COLUMN_CANDIDATES)
    label_column = _first_existing_column(dataframe.columns, LABEL_COLUMN_CANDIDATES)

    if code_column is None or label_column is None:
        raise KeyError(
            "Could not infer code/label columns. "
            f"Available columns: {list(dataframe.columns)}. "
            "Pass --code-column and --label-column explicitly."
        )
    return code_column, label_column


def normalize_columns(
    dataframe: pd.DataFrame,
    code_column: str | None = None,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Normalize arbitrary input columns to exactly ``code`` and ``label``.

    Passing ``None`` for a column enables inference using common names such as
    ``func``/``target`` for CodeXGLUE. Explicit column names must exist so typos
    are not silently replaced by inferred alternatives.
    """
    if code_column is None or label_column is None:
        inferred_code, inferred_label = infer_code_and_label_columns(dataframe)
        code_column = inferred_code if code_column is None else code_column
        label_column = inferred_label if label_column is None else label_column

    missing = [
        column
        for column in (code_column, label_column)
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    normalized = dataframe[[code_column, label_column]].rename(
        columns={code_column: "code", label_column: "label"}
    )
    return normalized


def convert_label_to_int(value: object) -> int | None:
    """Convert common binary-label representations to integer ``0`` or ``1``."""
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) == value and int(value) in {0, 1}:
            return int(value)
        return None

    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0

    try:
        numeric = float(normalized)
    except ValueError:
        return None
    if int(numeric) == numeric and int(numeric) in {0, 1}:
        return int(numeric)
    return None


def clean_dataset(
    dataframe: pd.DataFrame,
    min_code_length: int = MIN_CODE_LENGTH,
    max_code_length: int = MAX_CODE_LENGTH,
) -> pd.DataFrame:
    """Clean a normalized dataset containing ``code`` and ``label`` columns."""
    required = {"code", "label"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if min_code_length < 0:
        raise ValueError("min_code_length must be non-negative")
    if max_code_length <= 0:
        raise ValueError("max_code_length must be positive")
    if max_code_length < min_code_length:
        raise ValueError("max_code_length must be greater than or equal to min_code_length")

    cleaned = dataframe.copy()
    cleaned = cleaned.dropna(subset=["code", "label"])
    cleaned["code"] = cleaned["code"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].map(convert_label_to_int)
    cleaned = cleaned.dropna(subset=["code", "label"])
    cleaned["label"] = cleaned["label"].astype(int)
    cleaned = cleaned[cleaned["code"].str.len() >= min_code_length]
    cleaned["code"] = cleaned["code"].str.slice(0, max_code_length)

    label_counts_by_code = cleaned.groupby("code")["label"].nunique()
    conflicting_codes = label_counts_by_code[label_counts_by_code > 1]
    if not conflicting_codes.empty:
        examples = ", ".join(repr(code) for code in conflicting_codes.index[:3])
        raise ValueError(
            "Conflicting labels found for duplicate code snippets: "
            f"{examples}. Resolve duplicate labels before preprocessing."
        )

    cleaned = cleaned.drop_duplicates(subset=["code", "label"], keep="first")
    cleaned = cleaned[["code", "label"]]
    return cleaned.reset_index(drop=True)


def load_hugging_face_dataset(
    dataset_name: str,
    config_name: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Load a Hugging Face dataset and return it as a Pandas dataframe.

    For a DatasetDict, all available splits are concatenated before cleaning and
    re-splitting so the final output uses this project's 80/10/10 split policy.
    """
    try:
        from datasets import DatasetDict, concatenate_datasets, load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Hugging Face dataset loading requires the 'datasets' package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    if isinstance(dataset, DatasetDict):
        if not dataset:
            raise ValueError(f"Hugging Face dataset {dataset_name!r} contains no splits")
        dataset = concatenate_datasets(list(dataset.values()))

    return dataset.to_pandas()


def load_input_dataset(
    input_path: str | Path | None = None,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    hf_split: str | None = None,
) -> pd.DataFrame:
    """Load a dataset from a local file or Hugging Face Datasets."""
    if bool(input_path) == bool(hf_dataset):
        raise ValueError("Provide exactly one of --input or --hf-dataset")

    if hf_dataset:
        return load_hugging_face_dataset(
            dataset_name=hf_dataset,
            config_name=hf_config,
            split=hf_split,
        )

    path = Path(input_path)  # type: ignore[arg-type]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix in {".jsonl", ".json"}:
        return load_jsonl(path)
    raise ValueError(f"Unsupported input file extension: {suffix}. Use CSV or JSONL.")


def _split_counts(group_size: int, valid_size: float, test_size: float) -> tuple[int, int, int]:
    """Return train/valid/test counts for one class group."""
    test_count = int(round(group_size * test_size))
    valid_count = int(round(group_size * valid_size))

    if group_size >= 3:
        test_count = max(1, test_count)
        valid_count = max(1, valid_count)

    if test_count + valid_count >= group_size:
        overflow = test_count + valid_count - group_size + 1
        if valid_count >= test_count:
            valid_count = max(0, valid_count - overflow)
        else:
            test_count = max(0, test_count - overflow)

    train_count = group_size - valid_count - test_count
    return train_count, valid_count, test_count


def _simple_random_split(
    dataframe: pd.DataFrame,
    valid_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback split for tiny or single-class datasets."""
    shuffled = dataframe.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    total = len(shuffled)
    test_count = int(round(total * test_size))
    valid_count = int(round(total * valid_size))
    if total >= 3:
        test_count = max(1, test_count)
        valid_count = max(1, valid_count)
    train_count = max(0, total - valid_count - test_count)
    train = shuffled.iloc[:train_count]
    valid = shuffled.iloc[train_count : train_count + valid_count]
    test = shuffled.iloc[train_count + valid_count :]
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def split_dataset(
    dataframe: pd.DataFrame,
    train_size: float = 0.80,
    valid_size: float = DEFAULT_VALID_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test splits with label stratification.

    The split is implemented with Pandas to keep the module lightweight at import
    time. Each label group is shuffled independently and split according to the
    requested proportions, preserving class balance where the dataset is large
    enough. Tiny or single-class datasets fall back to a deterministic random
    split.
    """
    if dataframe.empty:
        raise ValueError("Cannot split an empty dataframe")
    if len(dataframe) < 3:
        raise ValueError(
            "Cannot create train, validation, and test splits with fewer than 3 rows"
        )
    if any(size <= 0 for size in (train_size, valid_size, test_size)):
        raise ValueError("train_size, valid_size, and test_size must all be positive")
    if abs((train_size + valid_size + test_size) - 1.0) > 1e-6:
        raise ValueError("train_size + valid_size + test_size must equal 1.0")
    if "label" not in dataframe.columns:
        raise KeyError("Missing required column: label")

    if dataframe["label"].nunique() < 2:
        return _simple_random_split(dataframe, valid_size, test_size, random_state)

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for offset, (_, group) in enumerate(dataframe.groupby("label", sort=True)):
        if len(group) < 3:
            return _simple_random_split(dataframe, valid_size, test_size, random_state)

        train_count, valid_count, test_count = _split_counts(
            group_size=len(group),
            valid_size=valid_size,
            test_size=test_size,
        )
        shuffled = group.sample(frac=1.0, random_state=random_state + offset).reset_index(drop=True)
        train_parts.append(shuffled.iloc[:train_count])
        valid_parts.append(shuffled.iloc[train_count : train_count + valid_count])
        test_end = train_count + valid_count + test_count
        test_parts.append(shuffled.iloc[train_count + valid_count : test_end])

    train = (
        pd.concat(train_parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    valid = (
        pd.concat(valid_parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    test = (
        pd.concat(test_parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return train, valid, test


def save_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save train, validation, and test CSV files."""
    directory = ensure_dir(output_dir)
    paths = {
        "train": directory / "train.csv",
        "valid": directory / "valid.csv",
        "test": directory / "test.csv",
    }
    train.to_csv(paths["train"], index=False)
    valid.to_csv(paths["valid"], index=False)
    test.to_csv(paths["test"], index=False)
    return paths


def preprocess_dataframe(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    code_column: str | None = None,
    label_column: str | None = None,
    min_code_length: int = MIN_CODE_LENGTH,
    max_code_length: int = MAX_CODE_LENGTH,
    train_size: float = 0.80,
    valid_size: float = DEFAULT_VALID_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Path]:
    """Normalize, clean, split, and save a dataframe."""
    normalized = normalize_columns(
        dataframe,
        code_column=code_column,
        label_column=label_column,
    )
    cleaned = clean_dataset(
        normalized,
        min_code_length=min_code_length,
        max_code_length=max_code_length,
    )
    train, valid, test = split_dataset(
        cleaned,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )
    return save_splits(train, valid, test, output_dir)


def preprocess_dataset(
    output_dir: str | Path,
    input_path: str | Path | None = None,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    hf_split: str | None = None,
    code_column: str | None = None,
    label_column: str | None = None,
    min_code_length: int = MIN_CODE_LENGTH,
    max_code_length: int = MAX_CODE_LENGTH,
    train_size: float = 0.80,
    valid_size: float = DEFAULT_VALID_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Path]:
    """Load, normalize, clean, split, and save a dataset."""
    raw = load_input_dataset(
        input_path=input_path,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        hf_split=hf_split,
    )
    return preprocess_dataframe(
        raw,
        output_dir=output_dir,
        code_column=code_column,
        label_column=label_column,
        min_code_length=min_code_length,
        max_code_length=max_code_length,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )


def preprocess_file(
    input_path: str | Path,
    output_dir: str | Path,
    code_column: str | None = None,
    label_column: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_code_length: int = MAX_CODE_LENGTH,
) -> dict[str, Path]:
    """Compatibility wrapper for local CSV/JSONL preprocessing."""
    return preprocess_dataset(
        input_path=input_path,
        output_dir=output_dir,
        code_column=code_column,
        label_column=label_column,
        random_state=random_state,
        max_code_length=max_code_length,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Preprocess code classification data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local CSV or JSONL input file.")
    source.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face dataset name, for example google/code_x_glue_cc_defect_detection.",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--hf-split",
        default=None,
        help="Optional Hugging Face split to load before creating project splits.",
    )
    parser.add_argument("--output-dir", type=Path, default=get_settings().processed_data_dir)
    parser.add_argument(
        "--code-column",
        default=None,
        help="Optional explicit code column name; omitted means infer.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional explicit label column name; omitted means infer.",
    )
    parser.add_argument("--min-code-length", type=int, default=MIN_CODE_LENGTH)
    parser.add_argument("--max-code-length", type=int, default=MAX_CODE_LENGTH)
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--valid-size", type=float, default=DEFAULT_VALID_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def main() -> None:
    """Run preprocessing from the command line."""
    args = build_arg_parser().parse_args()
    paths = preprocess_dataset(
        input_path=args.input,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        output_dir=args.output_dir,
        code_column=args.code_column,
        label_column=args.label_column,
        min_code_length=args.min_code_length,
        max_code_length=args.max_code_length,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    for split, path in paths.items():
        print(f"Saved {split}: {path}")


if __name__ == "__main__":
    main()
