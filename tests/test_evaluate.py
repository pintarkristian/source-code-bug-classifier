"""Tests for evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import normalize_transformer_logits


def test_normalize_transformer_logits_supports_single_logit_output() -> None:
    logits = np.array([[0.0], [2.0], [-2.0]], dtype=float)

    probabilities = normalize_transformer_logits(logits)

    assert probabilities.shape == (3,)
    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] > 0.8
    assert probabilities[2] < 0.2


def test_normalize_transformer_logits_supports_two_class_logits() -> None:
    logits = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=float)

    probabilities = normalize_transformer_logits(logits)

    assert probabilities.shape == (2,)
    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] > 0.8


def test_normalize_transformer_logits_rejects_unsupported_shapes() -> None:
    with pytest.raises(ValueError, match="Unsupported Transformer logits shape"):
        normalize_transformer_logits(np.array(1.0))

    with pytest.raises(ValueError, match="must output either one logit or two class logits"):
        normalize_transformer_logits(np.array([[0.1, 0.2, 0.3]], dtype=float))
