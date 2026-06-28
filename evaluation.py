"""
evaluation.py
-------------
Loads saved evaluation metadata and renders charts / tables
used by the Streamlit Model Performance page.
"""

import os

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

META_PATH = "models/model_meta.pkl"

CLASS_NAMES = [
    "Non-Toxic", "Toxic", "Severe Toxic",
    "Obscene", "Threat", "Insult", "Identity Hate",
]

_PALETTE = {
    "Logistic Regression":      "#4ECDC4",
    "Multinomial Naïve Bayes":  "#FF6B6B",
}

plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#C9D1D9",
    "grid.color":       "#21262D",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})


def load_meta() -> dict:
    """Return saved model metadata dict or empty dict if not yet trained."""
    if os.path.exists(META_PATH):
        return joblib.load(META_PATH)
    return {}


# ── Metric comparison table ───────────────────────────────────────────────────
def metrics_dataframe(meta: dict) -> pd.DataFrame:
    """Build a tidy DataFrame comparing LR vs NB on key metrics."""
    rows = []
    for key, label in [("lr", "Logistic Regression"), ("nb", "Multinomial Naïve Bayes")]:
        m = meta.get(key, {})
        rows.append({
            "Model":     label,
            "Accuracy":  round(m.get("accuracy", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall":    round(m.get("recall", 0), 4),
            "F1 Score":  round(m.get("f1", 0), 4),
            "CV F1":     round(m.get("cv_f1", 0), 4),
        })
    return pd.DataFrame(rows)


# ── Grouped bar chart – model comparison ─────────────────────────────────────
def plot_model_comparison(meta: dict) -> plt.Figure:
    df  = metrics_dataframe(meta)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "CV F1"]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#161B22")

    for i, (_, row) in enumerate(df.iterrows()):
        color = list(_PALETTE.values())[i]
        bars  = ax.bar(x + i * w - w / 2, [row[m] for m in metrics], w,
                       label=row["Model"], color=color, alpha=0.9, edgecolor="#21262D")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#C9D1D9")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ── Confusion matrix heatmap ──────────────────────────────────────────────────
def plot_confusion_matrix(meta: dict, model_key: str = "lr") -> plt.Figure:
    m  = meta.get(model_key, {})
    cm = m.get("confusion_matrix", None)
    if cm is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data – train the model first.",
                ha="center", va="center", transform=ax.transAxes, color="white")
        return fig

    present_classes = sorted(set(range(cm.shape[0])))
    labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in present_classes]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#161B22")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#21262D",
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual", fontsize=11, labelpad=10)
    label = "Logistic Regression" if model_key == "lr" else "Multinomial Naïve Bayes"
    ax.set_title(f"Confusion Matrix – {label}", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    return fig


# ── Single-prediction probability bar chart ───────────────────────────────────
def plot_prediction_probabilities(probabilities: dict) -> plt.Figure:
    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = ["#FF6B6B" if v == max(values) else "#4ECDC4" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#161B22")

    bars = ax.barh(labels, values, color=colors, edgecolor="#21262D", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color="#C9D1D9")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Prediction Probability Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
