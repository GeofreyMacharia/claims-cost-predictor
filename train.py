"""
WEEK 4 STARTER — train the baseline classifier.

Fill in the sections marked TODO. When you run this file, it should:
  1. Load data/member_summary.csv (generating it if missing)
  2. Split into training and test sets
  3. Fit a scikit-learn Pipeline that preprocesses features and trains a classifier
  4. Print evaluation metrics on the test set
  5. Save the trained pipeline to model.joblib

Run:
    python train.py

Check your work:
    pytest tests/test_leakage.py -v

--------------------------------------------------------------------------------
STEP 0 — before you change anything.

The NUMERIC_FEATURES list below is every numeric column that comes out of
data/member_summary.csv, straight off the shelf. Get the TODOs working with
that list first, run the script, and write down two numbers:

    ROC-AUC:  __1.0____        recall on the "High Cost" class: ___1.0___

Then ask yourself whether you would believe those numbers if a colleague
showed them to you in a review, and only then run the leakage tests.
--------------------------------------------------------------------------------
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")

CATEGORICAL_FEATURES = ["age_bucket", "sex", "region"]

# Every numeric column in member_summary.csv, as generated. Some of these belong
# in a model and some do not — that is your call to make, not the dataset's.
NUMERIC_FEATURES = [
    # "total_claims",
    "medical_claims",
    "pharmacy_claims",
    "specialty_drug_fills",
    # "total_billed_cents", # leaky anbd high VIF
    # "total_allowed_cents",
    # "total_plan_paid_cents", # leaky
    # "total_oop_cents",
    "unique_diagnosis_codes",
    "inpatient_claims",
    # "has_specialty_drug", # samer infomaton as specialty drug fills
]
TARGET = "is_high_cost"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("data/member_summary.csv not found — generating synthetic data first.")
        from generate_claims_data import generate

        generate()
    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    """
    TODO 2 — Build and return a scikit-learn Pipeline with two steps.

    Step 1 — a ColumnTransformer named "preprocessor" that:
        - One-hot encodes CATEGORICAL_FEATURES.
          Use handle_unknown="ignore" so unseen categories at predict time
          don't crash the API.
        - Standard-scales NUMERIC_FEATURES.

    Step 2 — a classifier named "classifier".
        A RandomForestClassifier is a reasonable starting point.
        The dataset is imbalanced (positives are well under 15%), so think about
        how you want to handle that. Hint: see the `class_weight` parameter.
        Pass random_state=RANDOM_STATE so your results are reproducible.

    Docs:
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    """

    # column transformer
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"),CATEGORICAL_FEATURES )
    ])
    pipeline = Pipeline([
        ('preprocessor',preprocessor),
        ('classifier', RandomForestClassifier(class_weight="balanced",  n_estimators=400, n_jobs=-1, max_depth=7, random_state=RANDOM_STATE))
        
        ])
    return pipeline


def main():
    df = load_data()
    print(f"High-cost threshold: ${HIGH_COST_THRESHOLD_USD:,} in plan-paid claims")
    print(f"Prevalence in this cohort: {100 * df[TARGET].mean():.1f}%")


# VIF needs a constant column and all-numeric input.
    X = add_constant(df[NUMERIC_FEATURES].astype(float))

# One VIF per feature; skip the constant and sort worst first.
    vif = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
})
    print(vif[vif["feature"] != "const"].sort_values("VIF", ascending=False).round(2))

    # high VIF (1-10 okay, over 10 bad)
#                        feature       VIF
# 5      total_allowed_cents  13047.56
# 6    total_plan_paid_cents  10148.77
# 7          total_oop_cents    327.43
# 4       total_billed_cents     20.02
# 3     specialty_drug_fills      7.60
# 1           medical_claims      6.17
# 10      has_specialty_drug      5.58
# 8   unique_diagnosis_codes      4.82
# 9         inpatient_claims      1.90
# 2          pharmacy_claims      1.72

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].astype(int)

    # TODO 1: Split X and y into training and test sets.
    #   - TEST_SIZE of the data goes to the test set
    #   - Stratify on y so the rare positive class is represented in both splits
    #   - Use random_state=RANDOM_STATE for reproducibility
    # Replace the line below with your implementation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,stratify=y, random_state=RANDOM_STATE )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # TODO 3: Score the model on the test set.
    #   - Compute the predicted class labels (y_pred)
    #   - Compute the predicted probability of the positive class (y_proba)
    # Hint: pipeline.predict(...) and pipeline.predict_proba(...)[:, 1]
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    print("\n=== Test Set Results ===")
    # TODO 4: Print the classification_report comparing y_test and y_pred.
    # Use target_names=["Normal", "High Cost"] so the output is readable.
    target_names=["Normal", "High Cost"]
    class_report = classification_report(y_test, y_pred,target_names=target_names)
    print(class_report)


    # TODO 5: Print the ROC-AUC score using y_test and y_proba.
    Auc_score = roc_auc_score(y_score=y_proba, y_true=y_test)
    print('Auc_Score:', Auc_score)


    # TODO 6 (do this one last): print the 5 features the model leaned on most.
    #   feature_names = pipeline["preprocessor"].get_feature_names_out()
    #   importances = pipeline["classifier"].feature_importances_
    # Sort them together and print the top 5. If one feature dominates every
    # other feature combined, that is worth explaining before you move on.
    feature_names = pipeline['preprocessor'].get_feature_names_out()
    importances = pipeline['classifier'].feature_importances_

    # sort them
    imps = pd.DataFrame({
        'Feature Names': feature_names,
        'Values': importances
    }).sort_values('Values', ascending=False).reset_index(drop=True)
    print(imps.head(5))




    joblib.dump(pipeline, "model.joblib")
    print("\nModel saved to model.joblib")


if __name__ == "__main__":
    main()
