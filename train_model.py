"""
train_model.py
--------------
Trains Logistic Regression and Multinomial Naïve Bayes classifiers
on the Jigsaw Toxic Comment dataset, evaluates both, selects the
better model, and persists artefacts to disk.

Usage
-----
    python train_model.py --data path/to/train.csv
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from preprocessing import preprocess_dataframe

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
LABEL_COLS   = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_PATH   = "models/model.pkl"
TFIDF_PATH   = "models/tfidf.pkl"
META_PATH    = "models/model_meta.pkl"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MAX_FEATURES = 50_000
NGRAM_RANGE  = (1, 2)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the Jigsaw CSV and validate expected columns."""
    df = pd.read_csv(csv_path)
    required = {"comment_text"} | set(LABEL_COLS)
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    print(f"[INFO] Loaded {len(df):,} rows from '{csv_path}'.")
    return df


# ── Label engineering ─────────────────────────────────────────────────────────
def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Collapse the six binary label columns into a single integer class:
      0 – Non-Toxic
      1 – Toxic (catch-all for any flagged label)

    Priority ordering for multi-label rows:
      severe_toxic > threat > identity_hate > obscene > insult > toxic
    """
    def row_label(r):
        if r["severe_toxic"]:  return 2
        if r["threat"]:        return 4
        if r["identity_hate"]: return 6
        if r["obscene"]:       return 3
        if r["insult"]:        return 5
        if r["toxic"]:         return 1
        return 0

    return df.apply(row_label, axis=1)


CLASS_NAMES = {
    0: "Non-Toxic",
    1: "Toxic",
    2: "Severe Toxic",
    3: "Obscene",
    4: "Threat",
    5: "Insult",
    6: "Identity Hate",
}


# ── TF-IDF vectoriser ────────────────────────────────────────────────────────
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=2,
    )


# ── Model factories ──────────────────────────────────────────────────────────
def logistic_regression_model() -> LogisticRegression:
    return LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def naive_bayes_model() -> MultinomialNB:
    return MultinomialNB(alpha=0.1)


# ── Evaluation helper ────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Return a dict of evaluation metrics for the fitted model."""
    y_pred = model.predict(X_test)

    metrics = {
        "name":           model_name,
        "accuracy":       accuracy_score(y_test, y_pred),
        "precision":      precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":         recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":             f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
            zero_division=0,
        ),
    }
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(metrics["classification_report"])
    return metrics


def cross_validate(model, X_train, y_train, model_name: str, cv: int = 5) -> float:
    """Run stratified k-fold CV and return mean F1."""
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1)
    print(f"[CV] {model_name} – Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean())


# ── Main training routine ────────────────────────────────────────────────────
def train(csv_path: str):
    os.makedirs("models", exist_ok=True)

    # 1. Load & preprocess
    df = load_data(csv_path)
    print("[INFO] Preprocessing text …")
    df = preprocess_dataframe(df)

    # 2. Target
    y = build_target(df)
    X_raw = df["cleaned_text"]

    # 3. Train / test split
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4. TF-IDF fit on training set only
    print("[INFO] Fitting TF-IDF vectoriser …")
    tfidf   = build_vectorizer()
    X_train = tfidf.fit_transform(X_tr_raw)
    X_test  = tfidf.transform(X_te_raw)

    # 5. Train both models
    print("[INFO] Training Logistic Regression …")
    lr = logistic_regression_model()
    lr.fit(X_train, y_train)

    print("[INFO] Training Multinomial Naïve Bayes …")
    nb = naive_bayes_model()
    nb.fit(X_train, y_train)

    # 6. Evaluate
    lr_metrics = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    nb_metrics = evaluate_model(nb, X_test, y_test, "Multinomial Naïve Bayes")

    # 7. Cross-validation
    lr_cv = cross_validate(lr, X_train, y_train, "Logistic Regression")
    nb_cv = cross_validate(nb, X_train, y_train, "Multinomial Naïve Bayes")

    lr_metrics["cv_f1"] = lr_cv
    nb_metrics["cv_f1"] = nb_cv

    # 8. Model selection
    best_model  = lr if lr_metrics["f1"] >= nb_metrics["f1"] else nb
    best_name   = "Logistic Regression" if best_model is lr else "Multinomial Naïve Bayes"
    print(f"\n[INFO] ✅ Best model: {best_name}")

    # 9. Persist artefacts
    joblib.dump(best_model,              MODEL_PATH)
    joblib.dump(tfidf,                   TFIDF_PATH)
    joblib.dump({
        "lr":          lr_metrics,
        "nb":          nb_metrics,
        "best":        best_name,
        "class_names": CLASS_NAMES,
    }, META_PATH)

    print(f"[INFO] Saved model   → {MODEL_PATH}")
    print(f"[INFO] Saved tfidf   → {TFIDF_PATH}")
    print(f"[INFO] Saved meta    → {META_PATH}")
    return best_model, tfidf, {"lr": lr_metrics, "nb": nb_metrics}


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ToxiShield AI models.")
    parser.add_argument(
        "--data",
        default="data/train.csv",
        help="Path to the Jigsaw train.csv file (default: data/train.csv)",
    )
    args = parser.parse_args()
    train(args.data)
