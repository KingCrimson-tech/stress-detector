from dataclasses import dataclass
import numpy as np


STRESS_LABELS = ["Low", "Medium", "High"]

SURVEY_FEATURE_ORDER = [
    "sleep_quality",
    "headache_frequency",
    "academic_performance",
    "study_load",
    "extracurricular_activities",
    "social_support",
    "anxiety_level",
    "depression_level",
    "isolation_score",
    "future_insecurity",
    "financial_stress",
    "teacher_quality",
    "peer_pressure",
    "bullying",
    "safety_concerns",
    "living_conditions",
    "basic_needs",
    "noise_level",
    "self_esteem",
    "mental_health_history",
    "academic_workload",
]

EMOTION_TO_STRESS_WEIGHT = {
    "angry":    0.85,
    "disgust":  0.70,
    "fear":     0.90,
    "sad":      0.75,
    "surprise": 0.30,
    "neutral":  0.20,
    "happy":    0.05,
}


@dataclass
class EmotionResult:
    dominant_emotion: str
    emotion_score: float  # 0.0 = calm, 1.0 = maximum stress signal


@dataclass
class SurveyResult:
    raw_vector: np.ndarray    # shape (21,), raw 1-5 scale values
    survey_proba: np.ndarray  # shape (3,), [P(Low), P(Medium), P(High)]


@dataclass
class FusionResult:
    stress_label: str              # "Low", "Medium", or "High"
    confidence: float              # 0.0 to 1.0
    survey_weight: float           # e.g. 0.7
    emotion_weight: float          # e.g. 0.3
    survey_proba: np.ndarray       # kept for SHAP
    raw_survey_vector: np.ndarray  # kept for SHAP
    emotion_result: EmotionResult  # kept for display