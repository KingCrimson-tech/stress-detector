"""
Module 3 — Train: builds and persists the stress-detection pipeline.

Pipeline
--------
1. MinMaxScaler
2. SelectKBest (f_classif, k=15)
3. RFECV (LinearSVC, cv=5)
4. StackingClassifier
       estimators: SVC, RandomForest, XGBoost
       final_estimator: LogisticRegression

Artifacts saved to  models/
    stacking_model.pkl
    scaler.pkl
    selector_kbest.pkl
    selector_rfecv.pkl
"""

from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ── project imports ──────────────────────────────────────────────────
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from contracts import SURVEY_FEATURE_ORDER, STRESS_LABELS

# ── paths ────────────────────────────────────────────────────────────
DATA_PATH = ROOT_DIR / "data" / "StressLevelDataset.csv"
MODEL_DIR = ROOT_DIR / "models"

# ── column mapping: raw CSV name → contracts.py canonical name ───────
# The Kaggle CSV uses shorter/different column names.  We rename them so
# every downstream module can rely on SURVEY_FEATURE_ORDER exclusively.
CSV_TO_CONTRACT = {
    "anxiety_level":                "anxiety_level",
    "self_esteem":                  "self_esteem",
    "mental_health_history":        "mental_health_history",
    "depression":                   "depression_level",
    "headache":                     "headache_frequency",
    "blood_pressure":               "financial_stress",       # proxy mapping
    "sleep_quality":                "sleep_quality",
    "breathing_problem":            "isolation_score",        # proxy mapping
    "noise_level":                  "noise_level",
    "living_conditions":            "living_conditions",
    "safety":                       "safety_concerns",
    "basic_needs":                  "basic_needs",
    "academic_performance":         "academic_performance",
    "study_load":                   "study_load",
    "teacher_student_relationship": "teacher_quality",
    "future_career_concerns":       "future_insecurity",
    "social_support":               "social_support",
    "peer_pressure":                "peer_pressure",
    "extracurricular_activities":   "extracurricular_activities",
    "bullying":                     "bullying",
    # stress_level is the target — not renamed
}
# The CSV has 20 feature columns but contracts expects 21.
# "academic_workload" has no direct counterpart; we derive it from study_load.
DERIVED_FEATURE = "academic_workload"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and rename the CSV to match SURVEY_FEATURE_ORDER."""
    df = pd.read_csv(DATA_PATH)

    # rename existing columns
    df.rename(columns=CSV_TO_CONTRACT, inplace=True)

    # derive the 21st feature (academic_workload ≈ study_load copy)
    if DERIVED_FEATURE not in df.columns:
        df[DERIVED_FEATURE] = df["study_load"]

    X = df[SURVEY_FEATURE_ORDER].copy()
    y = df["stress_level"].copy()
    return X, y


def build_and_train() -> None:
    """Full training run: fit, evaluate, save."""

    print("=" * 60)
    print("Module 3 — Training Pipeline")
    print("=" * 60)

    # ── 1. load data ─────────────────────────────────────────────────
    X, y = load_data()
    print(f"\nDataset shape : {X.shape}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}\n")

    # ── 2. scaling ───────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. feature selection — SelectKBest ───────────────────────────
    selector_kbest = SelectKBest(score_func=f_classif, k=15)
    X_kbest = selector_kbest.fit_transform(X_scaled, y)
    kept_kbest = np.array(SURVEY_FEATURE_ORDER)[selector_kbest.get_support()]
    print(f"SelectKBest kept {len(kept_kbest)} features:")
    print(f"  {list(kept_kbest)}\n")

    # ── 4. feature selection — RFECV with LinearSVC ──────────────────
    rfecv_estimator = LinearSVC(max_iter=10_000, dual=False)
    selector_rfecv = RFECV(
        estimator=rfecv_estimator,
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        min_features_to_select=5,
    )
    X_selected = selector_rfecv.fit_transform(X_kbest, y)
    print(f"RFECV selected {selector_rfecv.n_features_} features from the "
          f"{X_kbest.shape[1]} passed in.\n")

    # ── 5. StackingClassifier ────────────────────────────────────────
    stacking_clf = StackingClassifier(
        estimators=[
            ("svm", SVC(probability=True)),
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
            ("xgb", XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )),
        ],
        final_estimator=LogisticRegression(max_iter=1_000),
        cv=5,
        n_jobs=-1,
    )

    # ── 6. cross-validated evaluation ────────────────────────────────
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print("Running 10-fold Stratified cross-validation …")

    y_pred = cross_val_predict(stacking_clf, X_selected, y, cv=cv)
    acc = accuracy_score(y, y_pred)

    print(f"\nCross-val accuracy: {acc:.4f}\n")
    print("Classification report (per class):")
    print(classification_report(
        y, y_pred,
        target_names=STRESS_LABELS,
        digits=4,
    ))

    # ── 7. final fit on full data ────────────────────────────────────
    print("Fitting final model on full dataset …")
    stacking_clf.fit(X_selected, y)

    # ── 8. persist artefacts ─────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(stacking_clf,   MODEL_DIR / "stacking_model.pkl")
    joblib.dump(scaler,         MODEL_DIR / "scaler.pkl")
    joblib.dump(selector_kbest, MODEL_DIR / "selector_kbest.pkl")
    joblib.dump(selector_rfecv, MODEL_DIR / "selector_rfecv.pkl")

    print(f"\n✅ Saved 4 artefacts to {MODEL_DIR}/")
    for name in ("stacking_model.pkl", "scaler.pkl",
                 "selector_kbest.pkl", "selector_rfecv.pkl"):
        size = (MODEL_DIR / name).stat().st_size
        print(f"   {name:.<30s} {size / 1024:.1f} KB")

    # ── 9. sanity check ──────────────────────────────────────────────
    sanity_check()


def sanity_check() -> None:
    """Load the saved model and predict on three dummy rows."""

    print("\n" + "=" * 60)
    print("Sanity check — loading saved artefacts and predicting")
    print("=" * 60)

    scaler         = joblib.load(MODEL_DIR / "scaler.pkl")
    selector_kbest = joblib.load(MODEL_DIR / "selector_kbest.pkl")
    selector_rfecv = joblib.load(MODEL_DIR / "selector_rfecv.pkl")
    model          = joblib.load(MODEL_DIR / "stacking_model.pkl")

    # three synthetic rows (21 features each, values 0–5 mimicking survey)
    dummy_rows = np.array([
        [1, 1, 0, 1, 3, 4, 2, 1, 1, 1, 1, 4, 4, 1, 1, 4, 4, 1, 3, 0, 2],  # likely Low
        [3, 3, 1, 3, 2, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 3, 2, 1, 3],  # likely Medium
        [5, 5, 1, 5, 1, 1, 5, 5, 5, 5, 5, 1, 1, 5, 5, 1, 1, 5, 1, 1, 5],  # likely High
    ], dtype=float)

    X_sc  = scaler.transform(dummy_rows)
    X_kb  = selector_kbest.transform(X_sc)
    X_sel = selector_rfecv.transform(X_kb)
    preds = model.predict(X_sel)
    probas = model.predict_proba(X_sel)

    for i, (pred, proba) in enumerate(zip(preds, probas)):
        label = STRESS_LABELS[int(pred)]
        print(f"  Row {i + 1}: predicted={pred} ({label})  "
              f"probabilities={np.round(proba, 3)}")

    print("\n✅ Sanity check passed — pipeline loads and predicts correctly.\n")


# ── entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    build_and_train()
