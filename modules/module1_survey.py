import os
import sys
import numpy as np
import streamlit as st

# Ensure project root is in path, so 'import contracts' works when run directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from contracts import SurveyResult, SURVEY_FEATURE_ORDER
from modules import module3_predict

def render_survey_form() -> SurveyResult | None:
    with st.form("survey_form"):
        responses = []
        for feature in SURVEY_FEATURE_ORDER:
            # Format the label nicely: replace underscores with spaces, title case
            label = feature.replace('_', ' ').title()
            # Slider with min=1, max=5, value=3
            val = st.slider(label=label, min_value=1, max_value=5, value=3)
            responses.append(val)
            
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            # 1. Stack all 21 values into np.array shape (21,), dtype float32
            raw_vector = np.array(responses, dtype=np.float32)
            
            # 2. Call modules.module3_predict.predict_stress(raw_vector)
            survey_proba = module3_predict.predict_stress(raw_vector)
            
            # 3. Return SurveyResult
            return SurveyResult(raw_vector=raw_vector, survey_proba=survey_proba)
            
    return None
