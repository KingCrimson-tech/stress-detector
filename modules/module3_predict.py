import os
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in path, so 'import contracts' works when run directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from contracts import SurveyResult
from contracts import SURVEY_FEATURE_ORDER

# Define paths and load dependencies once at import time
MODELS_DIR = os.path.join(project_root, 'models')

SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
KBEST_PATH = os.path.join(MODELS_DIR, 'selector_kbest.pkl')
RFECV_PATH = os.path.join(MODELS_DIR, 'selector_rfecv.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'stacking_model.pkl')

_scaler = joblib.load(SCALER_PATH)
_selector_kbest = joblib.load(KBEST_PATH)
_selector_rfecv = joblib.load(RFECV_PATH)
_model = joblib.load(MODEL_PATH)

def predict_stress(raw_vector: np.ndarray) -> np.ndarray:
    """
    Takes raw survey values shape (21,), unnormalised (1-5 scale)
    Returns probability array shape (3,) for [Low, Med, High]
    """
    # Keep training-time feature names so sklearn transformers do not warn.
    X = pd.DataFrame(
        [raw_vector.astype(float)],
        columns=SURVEY_FEATURE_ORDER,
    )
    
    # Apply pipeline steps
    X_scaled = _scaler.transform(X)
    X_kbest = _selector_kbest.transform(X_scaled)
    X_rfecv = _selector_rfecv.transform(X_kbest)
    
    # Predict probabilities and return as 1D array
    probs = _model.predict_proba(X_rfecv)
    return probs.flatten()

if __name__ == '__main__':
    # Test random (21,) shape
    test_vector = np.random.randint(1, 6, size=21)
    print(f"Test raw_vector shape: {test_vector.shape}")
    print(f"Test raw_vector values: {test_vector}")
    
    result_probs = predict_stress(test_vector)
    
    print(f"Resulting probabilities shape: {result_probs.shape}")
    print(f"Resulting probabilities values: {result_probs}")
