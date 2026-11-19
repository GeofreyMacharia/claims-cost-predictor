"""
WEEK 5 — evaluation, calibration, and picking an operating point.

Run:
    python evaluate.py

Check your work:
    pytest tests/test_evaluation.py -v
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_DIR = Path("reports")

# Must match train.py so we score on data the model never saw.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TODO 5 — your operating point (yours to choose and defend).
VALID_USE_CASES = ("care_management", "stop_loss", "pricing")

CHOSEN_USE_CASE = "care_management"
CHOSEN_THRESHOLD = 0.35
THRESHOLD_JUSTIFICATION = "Justification to be determined after testing"

# Rough economics for the care-management case, in dollars.
OUTREACH_COST_PER_FLAGGED = 250
SAVINGS_PER_TRUE_POSITIVE = 4_000


def load_holdout():
    """Rebuild the held-out test set and score it. Returns (X_test, y_test, y_proba)."""
    pipeline = joblib.load(MODEL_PATH)  # load the trained model
    df = pd.read_csv(DATA_PATH)

    features = list(pipeline["preprocessor"].feature_names_in_)  # columns the pipeline was trained on
    X = df[features]
    y = df["is_high_cost"]

    # Same split as train.py, so we only score unseen rows
    _, X_test, _, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of high cost
    return X_test, y_test, y_proba


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """One row per threshold: threshold, n_flagged, tp, fp, fn, precision, recall."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rows = []
    for t in thresholds:
        flagged = y_proba >= t  # flag members at or above the threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        n_flagged = tp + fp

        # Report 0.0 when nothing is flagged instead of 0/0
        precision = tp / n_flagged if n_flagged > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        rows.append({
            "threshold": round(float(t), 2),
            "n_flagged": n_flagged,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        })
    return pd.DataFrame(rows)


def expected_value(tp: int, fp: int, outreach_cost: float, savings_per_tp: float) -> float:
    """Net dollars of an operating point."""
    # Savings from true positives minus outreach cost for everyone flagged
    return tp * savings_per_tp - (tp + fp) * outreach_cost


def plot_curves(y_true, y_proba, out_path: Path = REPORT_DIR / "evaluation_curves.png") -> Path:
    """Three panels (ROC, precision-recall, calibration) saved as one PNG."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Where the chosen threshold sits on the ROC and PR curves
    flagged = y_proba >= CHOSEN_THRESHOLD
    tp = (flagged & (y_true == 1)).sum()
    fp = (flagged & (y_true == 0)).sum()
    tpr_c = tp / (y_true == 1).sum()
    fpr_c = fp / (y_true == 0).sum()
    prec_c = tp / flagged.sum() if flagged.sum() > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].scatter(fpr_c, tpr_c, color="red", zorder=5, label=f"threshold {CHOSEN_THRESHOLD}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend()

    # Panel 2: precision-recall
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(y_true, y_proba):.3f}")
    axes[1].scatter(tpr_c, prec_c, color="red", zorder=5, label=f"threshold {CHOSEN_THRESHOLD}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend()

    # Panel 3: calibration (above the diagonal = model under-predicts risk)
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, marker="o", label="Model")
    axes[2].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction of actual high-cost")
    axes[2].set_title("Calibration curve")
    axes[2].legend()

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    X_test, y_test, y_proba = load_holdout()
    print(f"Scored {len(y_test)} held-out members "
          f"({int(y_test.sum())} of them high-cost at ${HIGH_COST_THRESHOLD_USD:,})")

    sweep = threshold_sweep(y_test, y_proba)
    sweep = sweep.assign(
        net_dollars=[
            expected_value(row.tp, row.fp, OUTREACH_COST_PER_FLAGGED, SAVINGS_PER_TRUE_POSITIVE)
            for row in sweep.itertuples()
        ]
    )
    print("\n=== Threshold sweep ===")
    print(sweep.to_string(index=False))

    path = plot_curves(y_test, y_proba)
    print(f"\nCurves written to {path}")

    print("\n=== My operating point ===")
    print(f"Use case:      {CHOSEN_USE_CASE}")
    print(f"Threshold:     {CHOSEN_THRESHOLD}")
    print(f"Justification: {THRESHOLD_JUSTIFICATION}")


if __name__ == "__main__":
    main()