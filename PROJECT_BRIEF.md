# PROJECT BRIEF — Student Stress Detection System
# Paste this at the start of every AI coding session

## What this project is
A multimodal student stress detection system that:
1. Collects a 21-question survey from the student
2. Optionally captures a face photo for emotion detection
3. Fuses both signals using a weighted ensemble approach
4. Outputs a stress label (Low / Medium / High) with SHAP explainability

## Tech stack
- Python 3.10
- scikit-learn (ensemble ML)
- deepface (emotion detection, no training needed)
- shap (explainability)
- streamlit (frontend)
- joblib (model persistence)
- matplotlib (charts)

## Folder structure
stress_detector/
├── contracts.py          # shared data definitions — NEVER MODIFY
├── app.py                # streamlit UI only, no ML logic
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── module1_survey.py     # survey form → SurveyResult
│   ├── module2_emotion.py    # face photo → EmotionResult
│   ├── module3_train.py      # trains and saves model to /models/
│   ├── module3_predict.py    # loads model, predicts from SurveyResult
│   ├── module4_fusion.py     # fuses survey + emotion → FusionResult
│   └── module5_shap.py       # generates SHAP chart from FusionResult
├── models/               # joblib files saved here by module3_train.py
└── data/
    └── stress_dataset.csv

## Data flow (read this carefully)
survey form → SurveyResult → module3_predict → survey_proba (array shape 3,)
face photo  → EmotionResult → emotion_score (float 0-1)
both        → module4_fusion → FusionResult
FusionResult + model → module5_shap → matplotlib Figure
all of above → app.py → displayed in Streamlit

## The contracts.py file (DO NOT CHANGE THIS)
---
from dataclasses import dataclass
import numpy as np

STRESS_LABELS = ["Low", "Medium", "High"]

SURVEY_FEATURE_ORDER = [
    "sleep_quality", "headache_frequency", "academic_performance",
    "study_load", "extracurricular_activities", "social_support",
    "anxiety_level", "depression_level", "isolation_score",
    "future_insecurity", "financial_stress", "teacher_quality",
    "peer_pressure", "bullying", "safety_concerns",
    "living_conditions", "basic_needs", "noise_level",
    "self_esteem", "mental_health_history", "academic_workload"
]

EMOTION_TO_STRESS_WEIGHT = {
    "angry": 0.85, "disgust": 0.70, "fear": 0.90,
    "sad": 0.75,   "surprise": 0.30, "neutral": 0.20,
    "happy": 0.05
}

@dataclass
class EmotionResult:
    dominant_emotion: str
    emotion_score: float

@dataclass
class SurveyResult:
    raw_vector: np.ndarray
    survey_proba: np.ndarray

@dataclass
class FusionResult:
    stress_label: str
    confidence: float
    survey_weight: float
    emotion_weight: float
    survey_proba: np.ndarray
    raw_survey_vector: np.ndarray
    emotion_result: EmotionResult
---

## Rules the AI must follow
1. Every module imports its types from contracts.py — never redefine them
2. No ML logic in app.py — only imports and st.* display calls
3. Training (module3_train.py) and inference (module3_predict.py) are separate files
4. module3_train.py saves to /models/ using joblib
5. module3_predict.py loads from /models/ using joblib
6. Do not call plt.show() anywhere — return Figure objects
7. All functions must have type hints matching contracts.py types
8. Handle missing face gracefully: return EmotionResult("neutral", 0.2)

## Dataset
File: data/stress_dataset.csv
Source: https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis
Target column: "stress_level" with values 0 (Low), 1 (Medium), 2 (High)
Features: all 21 columns in SURVEY_FEATURE_ORDER
