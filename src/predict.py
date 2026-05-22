"""Prediction utilities for code bug/risk classification.

The primary inference path uses a fine-tuned Hugging Face Transformer model. When a
trained model directory is not available yet, the predictor falls back to a small
heuristic scorer so the FastAPI service can still start during early development.
Heuristic notes are explanatory only and do not prove that code is safe or unsafe.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, TypedDict

from src.config import get_settings

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_SUBDIR = Path("models") / "codebert-bug-classifier"
SUSPICIOUS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\beval\s*\(", re.IGNORECASE), "Use of eval detected"),
    (re.compile(r"\bexec\s*\(", re.IGNORECASE), "Use of exec detected"),
    (
        re.compile(r"\bshell\s*=\s*True\b", re.IGNORECASE),
        "shell=True subprocess risk detected",
    ),
    (re.compile(r"\bos\.system\s*\(", re.IGNORECASE), "os.system usage detected"),
    (re.compile(r"\bpickle\b", re.IGNORECASE), "pickle usage detected"),
    (re.compile(r"\bstrcpy\s*\(", re.IGNORECASE), "Potential unsafe C function strcpy detected"),
    (re.compile(r"\bsprintf\s*\(", re.IGNORECASE), "Potential unsafe C function sprintf detected"),
    (re.compile(r"\bgets\s*\(", re.IGNORECASE), "Potential unsafe C function gets detected"),
)
DIVISION_BY_VARIABLE_PATTERN = re.compile(r"(?<!/)/(?![/*=])\s*([A-Za-z_][A-Za-z0-9_]*)")


class PredictionResult(TypedDict):
    """Structured prediction output returned by :class:`CodeBugPredictor`."""

    label: str
    risk_score: float
    model_name: str
    notes: list[str]


class CodeBugPredictor:
    """Predict whether a code snippet appears clean or potentially buggy/risky.

    Parameters
    ----------
    model_dir:
        Directory containing a fine-tuned Hugging Face Transformer model and
        tokenizer. Defaults to ``models/codebert-bug-classifier`` relative to the
        project root.
    model_path:
        Backward-compatible alias for ``model_dir``. Prefer ``model_dir`` in new code.
    max_length:
        Maximum token length used during Transformer inference.
    threshold:
        Probability threshold for assigning the ``"buggy"`` label.
    model_name:
        Human-readable model name included in API responses.
    allow_heuristic_fallback:
        If ``True``, use a lightweight heuristic score when no trained model is
        available. This is useful before training has been completed.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        max_length: int = 256,
        threshold: float = 0.5,
        model_name: str | None = None,
        allow_heuristic_fallback: bool = True,
    ) -> None:
        settings = get_settings()
        default_model_dir = settings.project_root / DEFAULT_MODEL_SUBDIR
        resolved_model_dir = model_dir if model_dir is not None else model_path
        self.model_dir = (
            Path(resolved_model_dir)
            if resolved_model_dir is not None
            else default_model_dir
        )
        self.max_length = max_length
        self.threshold = threshold
        self.model_name = model_name or "codebert"
        self.allow_heuristic_fallback = allow_heuristic_fallback

        self.device: Any | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.load_error: str | None = None
        self.is_loaded = False

        self._load_transformer_if_available()

    def predict(self, code: str) -> PredictionResult:
        """Classify one code snippet and return label, score, model name, and notes.

        The returned ``risk_score`` is the model probability for the risky/buggy
        class when a trained Transformer model is loaded. If no model is loaded,
        it is a heuristic fallback score. In both cases, heuristic notes are only
        explanatory signals and should not be interpreted as proof of safety.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        stripped_code = code.strip()
        if not stripped_code:
            raise ValueError("code must not be blank")

        notes = self._build_heuristic_notes(stripped_code)

        if self.is_loaded:
            risk_score = self._predict_risk_with_transformer(stripped_code)
            label = "buggy" if risk_score >= self.threshold else "clean"
            return {
                "label": label,
                "risk_score": round(float(risk_score), 6),
                "model_name": self.model_name,
                "notes": notes,
            }

        if not self.allow_heuristic_fallback:
            raise RuntimeError(
                f"Transformer model is not loaded from {self.model_dir}. "
                "Train the model first or enable heuristic fallback."
            )

        risk_score = self._heuristic_risk_score(notes)
        label = "buggy" if risk_score >= self.threshold else "clean"
        fallback_notes = [
            "Trained Transformer model not loaded; using heuristic fallback only.",
            *notes,
        ]
        return {
            "label": label,
            "risk_score": round(float(risk_score), 6),
            "model_name": f"{self.model_name}-heuristic-fallback",
            "notes": fallback_notes,
        }

    def _load_transformer_if_available(self) -> None:
        """Load Transformer model/tokenizer when a trained model directory exists."""
        if not self._looks_like_transformer_model_dir(self.model_dir):
            LOGGER.info("Transformer model artifacts not found at %s", self.model_dir)
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            LOGGER.info("Loaded Transformer model from %s on %s", self.model_dir, self.device)
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps/artifacts
            self.load_error = str(exc)
            self.is_loaded = False
            self.model = None
            self.tokenizer = None
            LOGGER.warning("Could not load Transformer model from %s: %s", self.model_dir, exc)

    @staticmethod
    def _looks_like_transformer_model_dir(model_dir: Path) -> bool:
        """Return ``True`` if ``model_dir`` contains expected HF model files."""
        if not model_dir.exists() or not model_dir.is_dir():
            return False

        has_config = (model_dir / "config.json").exists()
        has_model_weights = any(
            (model_dir / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin", "tf_model.h5")
        )
        has_tokenizer = any(
            (model_dir / filename).exists()
            for filename in ("tokenizer.json", "vocab.json", "vocab.txt", "merges.txt")
        )
        return has_config and has_model_weights and has_tokenizer

    def _predict_risk_with_transformer(self, code: str) -> float:
        """Return the Transformer probability for the buggy/risky class."""
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("Transformer model is not loaded")

        import torch

        encoded = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        if logits.shape[-1] == 1:
            probability = torch.sigmoid(logits).squeeze().detach().cpu().item()
            return float(probability)

        probabilities = torch.softmax(logits, dim=-1).squeeze().detach().cpu()
        return float(probabilities[1].item())

    @classmethod
    def _build_heuristic_notes(cls, code: str) -> list[str]:
        """Create explanatory notes for obvious risky code patterns."""
        notes: list[str] = []

        for pattern, note in SUSPICIOUS_PATTERNS:
            if pattern.search(code) and note not in notes:
                notes.append(note)

        denominator = cls._find_division_without_zero_check(code)
        if denominator:
            notes.append(
                "Potential division-by-zero risk: "
                f"denominator '{denominator}' is used without an obvious zero check."
            )

        return notes

    @staticmethod
    def _strip_strings_and_comments(code: str) -> str:
        """Remove common string literals and comments before regex heuristics."""
        text = code
        text = re.sub(r"""\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''""", ' ', text)
        text = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", ' ', text)
        text = re.sub(r'/\*[\s\S]*?\*/', ' ', text)
        text = re.sub(r'//.*', ' ', text)
        text = re.sub(r'#.*', ' ', text)
        return text

    @classmethod
    def _find_division_without_zero_check(cls, code: str) -> str | None:
        """Return a denominator variable that lacks an obvious zero check, if any."""
        searchable_code = cls._strip_strings_and_comments(code)
        for match in DIVISION_BY_VARIABLE_PATTERN.finditer(searchable_code):
            denominator = match.group(1)
            if not cls._has_obvious_zero_check(searchable_code, denominator):
                return denominator
        return None

    @staticmethod
    def _has_obvious_zero_check(code: str, variable_name: str) -> bool:
        """Detect simple zero-check patterns for a denominator variable."""
        escaped = re.escape(variable_name)
        zero_check_patterns = (
            rf"\b{escaped}\s*(?:!=|>|>=)\s*0\b",
            rf"\b0\s*(?:!=|<|<=)\s*{escaped}\b",
            rf"\bassert\s+{escaped}\s*!=\s*0\b",
            rf"\bassert\s+0\s*!=\s*{escaped}\b",
        )
        return any(re.search(pattern, code) for pattern in zero_check_patterns)

    @staticmethod
    def _heuristic_risk_score(notes: list[str]) -> float:
        """Convert heuristic notes into a bounded fallback risk score."""
        if not notes:
            return 0.15
        return min(0.95, 0.35 + 0.12 * len(notes))
