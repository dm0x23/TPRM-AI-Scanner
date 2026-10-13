"""TEST DOUBLES for the two NLP models. NOT the assessment method. Never used by the app.

They exist so the pipeline's plumbing (evidence -> NLP -> aggregation -> risk ->
Excel) and code parity with the notebook can be checked where the real models
(facebook/bart-large-mnli, Qwen/Qwen2.5-0.5B-Instruct) cannot be downloaded. They are
deterministic, so the notebook and the integrated code receive identical numbers and
any difference in output must come from code. Their labels are NOT meaningful AI
findings.
"""
from __future__ import annotations

import hashlib

import numpy as np

from services.optiv_pipeline.nlp_engine import LABEL_ORDER


class StubZeroShotClassifier:
    """Deterministic pseudo-probabilities (keyword-tilted hash). Same interface as ZeroShotClassifier."""

    is_test_double = True

    def predict_proba(self, texts):
        out = np.zeros((len(texts), len(LABEL_ORDER)))
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode()).digest()
            base = np.array([1 + h[k] / 255 for k in range(len(LABEL_ORDER))], dtype=np.float64)
            low = t.lower()
            if any(w in low for w in ("uses machine learning", "large language model", "in production",
                                      "generally available", "anomaly detection")):
                base[0] += 4
            if any(w in low for w in ("we believe", "webinar", "roadmap", "plans to")):
                base[1] += 3
            if "rule-based" in low or "deterministic" in low:
                base[3] += 3
            out[i] = base / base.sum()
        return out


class StubCapabilitySummarizer:
    is_test_double = True

    def summarize(self, items):
        return [f"{v} appears to provide: {t[:150]}" for v, t in items]


def stub_model_factory():
    return StubZeroShotClassifier(), StubCapabilitySummarizer()
