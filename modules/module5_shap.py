"""SHAP explainability module for stress predictions."""

from __future__ import annotations

import functools
import pathlib

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from contracts import FusionResult, SURVEY_FEATURE_ORDER, STRESS_LABELS


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "StressLevelDataset.csv"


# ── column mapping: raw CSV name → contracts.py canonical name ───────
# Keep in sync with modules/module3_train.py.
CSV_TO_CONTRACT: dict[str, str] = {
    "anxiety_level":                "anxiety_level",
    "self_esteem":                  "self_esteem",
    "mental_health_history":        "mental_health_history",
    "depression":                   "depression_level",
    "headache":                     "headache_frequency",
    "blood_pressure":               "financial_stress",
    "sleep_quality":                "sleep_quality",
    "breathing_problem":            "isolation_score",
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
}
DERIVED_FEATURE = "academic_workload"


@functools.lru_cache(maxsize=1)
def _load_artifacts():
    stacking_model = joblib.load(MODELS_DIR / "stacking_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    selector_kbest = joblib.load(MODELS_DIR / "selector_kbest.pkl")
    selector_rfecv = joblib.load(MODELS_DIR / "selector_rfecv.pkl")
    return stacking_model, scaler, selector_kbest, selector_rfecv


@functools.lru_cache(maxsize=1)
def _load_background_transformed() -> np.ndarray:
    """Return a transformed background matrix for SHAP (RFECV-selected space)."""
    stacking_model, scaler, selector_kbest, selector_rfecv = _load_artifacts()

    df = pd.read_csv(DATA_PATH)
    df.rename(columns=CSV_TO_CONTRACT, inplace=True)
    if DERIVED_FEATURE not in df.columns:
        df[DERIVED_FEATURE] = df["study_load"]

    X = df[SURVEY_FEATURE_ORDER].astype(float)

    X_scaled = scaler.transform(X)
    X_kbest = selector_kbest.transform(X_scaled)
    X_rfecv = selector_rfecv.transform(X_kbest)

    # Sample to keep SHAP fast.
    max_rows = 200
    if X_rfecv.shape[0] > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_rfecv.shape[0], size=max_rows, replace=False)
        return X_rfecv[idx]
    return X_rfecv


def generate_shap_chart(fusion_result: FusionResult) -> plt.Figure:
    """
    Generate a horizontal bar chart showing SHAP values for the predicted class.
    
    Args:
        fusion_result: The fusion result containing raw survey vector and predictions.
        
    Returns:
        A matplotlib Figure with the SHAP explanation chart.
    """
    try:
        # Survey-only explanation: we ignore emotion completely and only use
        # the survey model + the survey feature vector.
        stacking_model, scaler, selector_kbest, selector_rfecv = _load_artifacts()

        # Transform the raw survey vector through the same pipeline as training.
        raw_df = pd.DataFrame(
            [fusion_result.raw_survey_vector.astype(float)],
            columns=SURVEY_FEATURE_ORDER,
        )
        scaled = scaler.transform(raw_df)
        kbest_selected = selector_kbest.transform(scaled)
        transformed = selector_rfecv.transform(kbest_selected)

        # Use a real background distribution so SHAP values aren't ~0.
        background = _load_background_transformed()
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(stacking_model.predict_proba, masker)

        # 4. Compute SHAP values for the transformed vector
        shap_values = explainer(transformed)
    
        # Determine explained class from the survey model output (not fusion).
        survey_proba = stacking_model.predict_proba(transformed)[0]
        predicted_class_idx = int(np.argmax(survey_proba))
        predicted_label = STRESS_LABELS[predicted_class_idx]
    
        # 6. Extract SHAP values for the predicted class
        # shap_values.values has shape (n_samples, n_features, n_classes) for multi-class
        if len(shap_values.values.shape) == 3:
            class_shap_values = shap_values.values[0, :, predicted_class_idx]
        else:
            class_shap_values = shap_values.values[0, :]
    
        # 7. Get feature names after selection
        # Track which features survive the pipeline
        kbest_mask = selector_kbest.get_support()
        rfecv_mask = selector_rfecv.get_support()
    
        # Features after kbest
        features_after_kbest = [f for f, m in zip(SURVEY_FEATURE_ORDER, kbest_mask) if m]
        # Features after rfecv
        final_features = [f for f, m in zip(features_after_kbest, rfecv_mask) if m]

        # Keep feature/value alignment safe across shap/sklearn version differences.
        n = min(len(class_shap_values), len(final_features))
        class_shap_values = class_shap_values[:n]
        final_features = final_features[:n]
    
        # 8. Get top 8 features by absolute SHAP value
        abs_shap = np.abs(class_shap_values)
        top_indices = np.argsort(abs_shap)[-8:][::-1]  # Top 8, descending
    
        top_features = [final_features[i] for i in top_indices]
        top_values = [class_shap_values[i] for i in top_indices]
    
        # 9. Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Reverse for display (top feature at top of chart)
        top_features = top_features[::-1]
        top_values = top_values[::-1]
    
        # Color bars based on sign
        colors = ["red" if v > 0 else "steelblue" for v in top_values]
    
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel("SHAP value (positive = increases stress)")
        ax.set_title("Why the model predicted this")
        ax.axvline(x=0, color="black", linewidth=0.5)

        plt.tight_layout()
        return fig
    except Exception:
        # Keep app usable even if SHAP runtime fails on a given environment.
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.axis("off")
        ax.text(
            0.01,
            0.6,
            "SHAP chart unavailable in current runtime.",
            fontsize=12,
            ha="left",
            va="center",
        )
        ax.text(
            0.01,
            0.35,
            f"Prediction: {fusion_result.stress_label} ({fusion_result.confidence:.1%})",
            fontsize=11,
            ha="left",
            va="center",
        )
        plt.tight_layout()
        return fig
