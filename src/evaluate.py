"""Evaluate trained code bug classifiers on a held-out test split.

This module supports both project model families:

* TensorFlow/Keras baseline models saved as ``.keras`` files.
* PyTorch/Hugging Face Transformer models saved with ``save_pretrained``.

The script loads ``test.csv``, generates probabilities and hard labels, computes
standard binary-classification metrics, and saves both JSON metrics and a
confusion-matrix figure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


MetricDict = dict[str, Any]


def configure_logging() -> None:
    """Configure console logging for evaluation progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained code bug classifier.")
    parser.add_argument("--test", type=Path, required=True, help="Path to processed test.csv.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=(
            "Path to the trained model. Use models/tensorflow_baseline.keras for "
            "TensorFlow or models/codebert-bug-classifier for Transformers."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["tensorflow", "transformer"],
        required=True,
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/test_metrics.json"),
        help="Path for the metrics JSON file.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/confusion_matrix.png"),
        help="Path for the confusion-matrix PNG figure.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for Transformer inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum Transformer token length.",
    )
    return parser


def load_test_data(path: Path) -> pd.DataFrame:
    """Load and validate the held-out test split.

    Parameters
    ----------
    path:
        Path to a CSV file containing ``code`` and ``label`` columns.

    Returns
    -------
    pandas.DataFrame
        A cleaned dataframe with only ``code`` and ``label`` columns.
    """
    LOGGER.info("Loading test data from %s", path)
    dataframe = pd.read_csv(path)
    required_columns = {"code", "label"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.loc[:, ["code", "label"]].dropna(subset=["code", "label"]).copy()
    dataframe["code"] = dataframe["code"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)

    invalid_labels = sorted(set(dataframe["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{path} contains non-binary labels: {invalid_labels}")
    if dataframe.empty:
        raise ValueError(f"{path} contains no usable rows.")

    LOGGER.info("Loaded %d usable test rows", len(dataframe))
    return dataframe


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Return numerically stable softmax probabilities for class logits."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def probabilities_to_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert positive-class probabilities to binary predicted labels."""
    return (probabilities >= threshold).astype(int)


def normalize_tensorflow_predictions(raw_predictions: np.ndarray) -> np.ndarray:
    """Normalize TensorFlow model outputs to positive-class probabilities."""
    predictions = np.asarray(raw_predictions)

    if predictions.ndim == 1:
        positive_probabilities = predictions
    elif predictions.ndim == 2 and predictions.shape[1] == 1:
        positive_probabilities = predictions[:, 0]
    elif predictions.ndim == 2 and predictions.shape[1] == 2:
        row_sums = predictions.sum(axis=1)
        if np.all((predictions >= 0) & (predictions <= 1)) and np.allclose(row_sums, 1, atol=1e-3):
            positive_probabilities = predictions[:, 1]
        else:
            positive_probabilities = stable_softmax(predictions)[:, 1]
    else:
        raise ValueError(f"Unsupported TensorFlow prediction shape: {predictions.shape}")

    return np.clip(positive_probabilities.astype(float), 0.0, 1.0)


def predict_tensorflow(model_path: Path, code_snippets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Generate probabilities and labels from a TensorFlow/Keras model."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for evaluating TensorFlow models. "
            "Install project dependencies first with: pip install -r requirements.txt"
        ) from exc

    LOGGER.info("Loading TensorFlow model from %s", model_path)
    model = tf.keras.models.load_model(model_path)

    LOGGER.info("Generating TensorFlow predictions for %d snippets", len(code_snippets))
    raw_predictions = model.predict(np.asarray(code_snippets, dtype=object), verbose=0)
    probabilities = normalize_tensorflow_predictions(raw_predictions)
    labels = probabilities_to_labels(probabilities)
    return probabilities, labels


def normalize_transformer_logits(raw_logits: Any) -> np.ndarray:
    """Normalize Transformer logits to positive-class probabilities."""
    logits = np.asarray(raw_logits)

    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    if logits.ndim != 2:
        raise ValueError(f"Unsupported Transformer logits shape: {logits.shape}")

    if logits.shape[1] == 1:
        positive_probabilities = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    elif logits.shape[1] == 2:
        positive_probabilities = stable_softmax(logits)[:, 1]
    else:
        raise ValueError(
            "Transformer model must output either one logit or two class logits; "
            f"received shape {logits.shape}"
        )

    return np.clip(positive_probabilities.astype(float), 0.0, 1.0)


def predict_transformer(
    model_dir: Path,
    code_snippets: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate probabilities and labels from a Hugging Face Transformer model."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Hugging Face Transformers are required for Transformer evaluation. "
            "Install project dependencies first with: pip install -r requirements.txt"
        ) from exc

    LOGGER.info("Loading Transformer tokenizer and model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device for Transformer evaluation: %s", device)
    model.to(device)
    model.eval()

    all_probabilities: list[np.ndarray] = []
    for start_index in range(0, len(code_snippets), batch_size):
        batch = code_snippets[start_index : start_index + batch_size]
        encoded_batch = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_batch = {key: value.to(device) for key, value in encoded_batch.items()}

        with torch.no_grad():
            outputs = model(**encoded_batch)
            probabilities = normalize_transformer_logits(
                outputs.logits.detach().cpu().numpy()
            )

        all_probabilities.append(probabilities)

    positive_probabilities = np.concatenate(all_probabilities).astype(float)
    predicted_labels = probabilities_to_labels(positive_probabilities)
    return positive_probabilities, predicted_labels


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
) -> MetricDict:
    """Compute binary classification metrics and confusion matrix values."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels_for_matrix = [0, 1]
    matrix = confusion_matrix(y_true, predicted_labels, labels=labels_for_matrix)

    metrics: MetricDict = {
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": labels_for_matrix,
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        LOGGER.warning("ROC-AUC and PR-AUC require both classes in y_true; storing null values.")
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def plot_confusion_matrix(matrix: list[list[int]], output_path: Path) -> Path:
    """Save a matplotlib confusion-matrix plot to disk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    matrix_array = np.asarray(matrix, dtype=int)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix_array)
    figure.colorbar(image, ax=axis)

    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["clean (0)", "buggy/risky (1)"])
    axis.set_yticklabels(["clean (0)", "buggy/risky (1)"])

    for row_index in range(matrix_array.shape[0]):
        for column_index in range(matrix_array.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix_array[row_index, column_index]),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def evaluate_model(args: argparse.Namespace) -> MetricDict:
    """Evaluate the requested model type and return metrics."""
    dataframe = load_test_data(args.test)
    code_snippets = dataframe["code"].tolist()
    y_true = dataframe["label"].to_numpy(dtype=int)

    if args.model_type == "tensorflow":
        probabilities, predicted_labels = predict_tensorflow(args.model_dir, code_snippets)
    elif args.model_type == "transformer":
        probabilities, predicted_labels = predict_transformer(
            args.model_dir,
            code_snippets,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:  # pragma: no cover - argparse prevents this path.
        raise ValueError(f"Unsupported model type: {args.model_type}")

    metrics = compute_binary_metrics(y_true, probabilities, predicted_labels)
    metrics.update(
        {
            "model_type": args.model_type,
            "model_dir": str(args.model_dir),
            "test_path": str(args.test),
            "num_examples": int(len(y_true)),
        }
    )
    return metrics


def main() -> None:
    """Run model evaluation from the command line."""
    configure_logging()
    args = build_arg_parser().parse_args()

    LOGGER.info("Starting evaluation for model_type=%s", args.model_type)
    metrics = evaluate_model(args)

    metrics_path = write_json(metrics, args.output)
    figure_path = plot_confusion_matrix(metrics["confusion_matrix"], args.figure_output)

    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved confusion matrix figure to %s", figure_path)


if __name__ == "__main__":
    main()
