"""
WEEK 6 STARTER — audit the model across demographic groups.

The model is one number per member. A fairness audit asks whether that number
is equally wrong for everybody, and "equally wrong" turns out to have several
mutually incompatible definitions.

Fill in every section marked TODO. When you run this file it should:
  1. Score every member with the trained model
  2. Report per-group rates for sex and age_bucket
  3. Apply the 80% rule
  4. Compare against a model retrained without the demographic columns
  5. Write reports/fairness_report.md

Run:
    python fairness.py

Check your work:
    pytest tests/test_fairness.py -v
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_PATH = Path("reports/fairness_report.md")

PROTECTED_COLUMNS = ["sex", "age_bucket"]
DECISION_THRESHOLD = 0.35  # Specified Threshold
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ---------------------------------------------------------------------------
# TODO 6 — your written assessment.
#
# After you have the numbers, fill this in. It is graded on whether the claim
# follows from your own cohort's numbers, not on whether it is reassuring.
# Reference specific rates. Name the group that is worse off. If a subgroup is
# too small to conclude anything, say that instead of reporting a ratio to two
# decimal places as if it meant something.
# ---------------------------------------------------------------------------
# Understanding results
Properties = " base rate -> percentage of high cost members" \
"               positive _rate -> how many were flagged" \
"               Precision -> How many were actually high cost"

ASSESSMENT = "Utilizing the optimal threshold value chosen during evaluation of the two groups, sex passed the " \
"test with an impact ration of 0.84 while age bucket failed. The model flags women more often than men (22.1% vs 18.2%), " \
"partly because women’s base rate of high cost is higher (6.7% vs 5.4%), and partly because normal women are falsely flagged " \
"more often (FPR 16.5% vs 13.7%). Age failed with a ratio of 0.386. Members aged 50 to 64 are worst off: they have the highest false" \
"alarm rates (19.4% and 21.0%), and 50 to 59 is the only group with a missed high-cost member." \
"Regarding blind model comparison, the model with out sex and age bucket revealed the infomational impact gained," \
"For sex, the ratio rose from 0.824 to 0.983, so the gap came from the sex column itself. For age, it only rose from 0.386" \
"to 0.482, because claim counts act as proxies for age. Removing a sensitive column is therefore not enough to make a model fair "



def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """
    TODO 1 — return {"tp": .., "fp": .., "fn": .., "tn": ..}.

    Write it with vectorized comparisons rather than a loop. Both inputs are
    0/1 arrays or Series.
    """
    # tp -> ypred * y_true -> tp = 1 
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    return {
        "tp":tp,
        "tn":tn,
        "fn":fn,
        "fp":fp
    }


def group_metrics(df: pd.DataFrame, group_col: str,
                  y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    """
    TODO 2 — one row per group value, with these columns:

        group, n, base_rate, positive_rate, tpr, fpr, fnr, precision

        n              members in the group
        base_rate      share of the group that is actually high-cost
        positive_rate  share of the group the model flags   (demographic parity)
        tpr            of the actually-high-cost, share flagged
        fpr            of the actually-normal, share flagged (equalized odds)
        fnr            1 - tpr; the members the model misses
        precision      of the flagged, share actually high-cost

    Build it on top of confusion_counts(). Sort by group so the output is stable
    across runs. Guard the divisions: a group with zero actual positives has no
    TPR, and reporting 0.0 there is a lie — use np.nan.
    """

    rows = []
    # sort=True keeps the group order the same on every run.
    for group, part in df.groupby(group_col, sort=True):
        c = confusion_counts(part[y_true_col], part[y_pred_col])
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

        n = len(part)
        positives = tp + fn      # members who really are high-cost
        negatives = fp + tn      # members who really are normal
        flagged = tp + fp        # members the model flagged

        # np.nan when the denominator is 0, because reporting 0.0 would be a lie.
        tpr = tp / positives if positives > 0 else np.nan

        rows.append({
            "group": group,
            "n": n,
            "base_rate": positives / n,
            "positive_rate": flagged / n,
            "tpr": tpr,
            "fpr": fp / negatives if negatives > 0 else np.nan,
            "fnr": 1 - tpr,      # stays nan when tpr is nan
            "precision": tp / flagged if flagged > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def disparate_impact_ratio(rates: pd.Series) -> float:
    """
    TODO 3 — the four-fifths ratio: the lowest rate divided by the highest.

    Ignore nan rates. Returns a number in [0, 1], where 1.0 is exact parity.
    """

     # Drop the nan rates, as the docstring says.
    rates = rates.dropna()
    if rates.empty:
        return np.nan

    # If even the highest rate is 0, every group is treated the same, so parity is exact.
    if rates.max() == 0:
        return 1.0

    return float(rates.min() / rates.max())


def passes_80_percent_rule(ratio: float) -> bool:
    """
    TODO 4 — True when the ratio clears the EEOC four-fifths guideline.

    Careful with the boundary: exactly 0.80 passes.
    """
    return bool(ratio >= 0.8)


def blind_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO 5 — retrain without the demographic columns and re-audit.

    `df` arrives with y_true / y_pred already on it from the full model. Score
    the blind model yourself rather than reusing those columns, so the two
    numbers you compare come from two actual models.

    Steps:
      - take the fitted pipeline's feature list, drop PROTECTED_COLUMNS
      - build and fit a pipeline of the same shape on the same training split
      - score, threshold, and run group_metrics() again per protected column
      - return a DataFrame with one row per protected column and the disparate
        impact ratio with the demographic features and without, side by side

    Predict the result before you run it, in one sentence: ____________________

    Most students expect the disparity to disappear. Whatever you find, the
    interesting question is *why*. Which remaining columns carry the same
    information the demographic columns did? Check with:
        df.groupby("sex")[numeric_cols].mean()
    """

    # Step 1: the fitted pipeline's feature list, minus the protected columns.
    full = joblib.load(MODEL_PATH)
    features = list(full["preprocessor"].feature_names_in_)
    blind_features = [c for c in features if c not in PROTECTED_COLUMNS]

    # Step 2: same shape as train.py (text columns one-hot encoded, numbers scaled).
    categorical = [c for c in blind_features if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in blind_features if c not in categorical]
    blind = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        # Copy the full model's settings so only the columns differ.
        ("classifier", RandomForestClassifier(**full["classifier"].get_params())),
    ])

    # Same split as train.py, then fit on the training rows only.
    X_train, _, y_train, _ = train_test_split(
        df[blind_features], df["y_true"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df["y_true"],
    )
    blind.fit(X_train, y_train)

    # Step 3: score and threshold every member, the same way main() scores the full model.
    blind_df = df.copy()
    proba = blind.predict_proba(df[blind_features])[:, 1]
    blind_df["y_pred"] = (proba >= DECISION_THRESHOLD).astype(int)

    # Step 4: one row per protected column, with the ratio with and without it.
    rows = []
    for col in PROTECTED_COLUMNS:
        with_rates = group_metrics(df, col).set_index("group")["positive_rate"]
        without_rates = group_metrics(blind_df, col).set_index("group")["positive_rate"]
        rows.append({
            "protected_column": col,
            "ratio_with": disparate_impact_ratio(with_rates),
            "ratio_without": disparate_impact_ratio(without_rates),
        })
    return pd.DataFrame(rows)



def main() -> None:
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    features = list(pipeline["preprocessor"].feature_names_in_)
    scored = df.assign(
        y_true=df["is_high_cost"].astype(int),
        y_proba=pipeline.predict_proba(df[features])[:, 1],
    )
    scored["y_pred"] = (scored["y_proba"] >= DECISION_THRESHOLD).astype(int)

    print(f"Auditing at threshold {DECISION_THRESHOLD} "
          f"(label: >${HIGH_COST_THRESHOLD_USD:,} plan-paid)")

    lines = [f"# Fairness audit\n", f"Decision threshold: {DECISION_THRESHOLD}\n"]
    for col in PROTECTED_COLUMNS:
        table = group_metrics(scored, col)
        ratio = disparate_impact_ratio(table.set_index("group")["positive_rate"])
        verdict = "PASS" if passes_80_percent_rule(ratio) else "FAIL"

        print(f"\n=== {col} ===")
        print(table.to_string(index=False))
        print(f"disparate impact ratio (positive rate): {ratio:.3f} → {verdict}")

        lines += [f"\n## {col}\n", table.to_markdown(index=False),
                  f"\n\nDisparate impact ratio: {ratio:.3f} ({verdict})\n"]

    print("\n=== Blind model comparison ===")
    comparison = blind_model_comparison(scored)
    print(comparison.to_string(index=False))
    lines += ["\n## Blind model comparison\n", comparison.to_markdown(index=False),
              f"\n\n## Assessment\n\n{ASSESSMENT}\n"]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
