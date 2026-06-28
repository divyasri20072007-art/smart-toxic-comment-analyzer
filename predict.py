"""
predict.py
----------
Inference module for ToxiShield AI.

Loads the saved model + TF-IDF vectoriser and exposes:
  • predict_single(text)  – classify one comment
  • predict_batch(df)     – classify a DataFrame of comments
"""

import os

import joblib
import numpy as np
import pandas as pd

from preprocessing import preprocess_text

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/model.pkl"
TFIDF_PATH = "models/tfidf.pkl"
META_PATH  = "models/model_meta.pkl"

# ── Class mapping ────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Non-Toxic",
    1: "Toxic",
    2: "Severe Toxic",
    3: "Obscene",
    4: "Threat",
    5: "Insult",
    6: "Identity Hate",
}

RISK_LEVELS = {
    "Non-Toxic":     ("Low",      "✅"),
    "Toxic":         ("High",     "🚨"),
    "Severe Toxic":  ("Critical", "💀"),
    "Obscene":       ("High",     "🔞"),
    "Threat":        ("Critical", "⚠️"),
    "Insult":        ("Medium",   "😠"),
    "Identity Hate": ("Critical", "🚫"),
}

SUGGESTED_ACTIONS = {
    "Non-Toxic":     "Allow – no action required.",
    "Toxic":         "Flag for human review.",
    "Severe Toxic":  "Remove immediately and warn user.",
    "Obscene":       "Hide behind content warning.",
    "Threat":        "Remove immediately and escalate to safety team.",
    "Insult":        "Issue a community-guidelines warning.",
    "Identity Hate": "Remove immediately and report to trust & safety.",
}


# ── Lazy loaders ─────────────────────────────────────────────────────────────
_model = None
_tfidf = None
_meta  = None


def _load_artefacts():
    global _model, _tfidf, _meta
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Run train_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
        _tfidf = joblib.load(TFIDF_PATH)
        _meta  = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}


# ── Single-text prediction ────────────────────────────────────────────────────
def predict_single(text: str) -> dict:
    """
    Classify a single comment.

    Returns
    -------
    dict with keys:
        label, confidence, probabilities, risk_level, risk_emoji,
        suggested_action, cleaned_text
    """
    _load_artefacts()

    cleaned = preprocess_text(text)
    vec     = _tfidf.transform([cleaned])

    label_idx  = int(_model.predict(vec)[0])
    label_name = CLASS_NAMES.get(label_idx, "Unknown")

    # Probability distribution (may not exist for all estimators)
    if hasattr(_model, "predict_proba"):
        probs_raw = _model.predict_proba(vec)[0]
    else:
        probs_raw = np.zeros(len(CLASS_NAMES))
        probs_raw[label_idx] = 1.0

    probabilities = {
        CLASS_NAMES[i]: float(p)
        for i, p in enumerate(probs_raw)
        if i in CLASS_NAMES
    }
    confidence = float(probs_raw[label_idx]) * 100

    risk_level, risk_emoji = RISK_LEVELS.get(label_name, ("Unknown", "❓"))
    action = SUGGESTED_ACTIONS.get(label_name, "Review manually.")

    return {
        "label":            label_name,
        "confidence":       round(confidence, 2),
        "probabilities":    probabilities,
        "risk_level":       risk_level,
        "risk_emoji":       risk_emoji,
        "suggested_action": action,
        "cleaned_text":     cleaned,
    }


# ── Batch prediction ──────────────────────────────────────────────────────────
def predict_batch(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """
    Classify every row in *df*.

    Parameters
    ----------
    df       : DataFrame containing a text column.
    text_col : Name of the column with raw comment text.

    Returns
    -------
    DataFrame with added columns:
        cleaned_text, predicted_label, confidence, risk_level, suggested_action
    """
    _load_artefacts()

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df["cleaned_text"] = df[text_col].apply(preprocess_text)

    X       = _tfidf.transform(df["cleaned_text"])
    indices = _model.predict(X).astype(int)

    df["predicted_label"] = [CLASS_NAMES.get(i, "Unknown") for i in indices]

    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(X)
        df["confidence"] = [
            round(float(probs[row, idx]) * 100, 2)
            for row, idx in enumerate(indices)
        ]
    else:
        df["confidence"] = 100.0

    df["risk_level"]       = df["predicted_label"].map(
        lambda x: RISK_LEVELS.get(x, ("Unknown", ""))[0]
    )
    df["suggested_action"] = df["predicted_label"].map(
        lambda x: SUGGESTED_ACTIONS.get(x, "Review manually.")
    )

    return df
