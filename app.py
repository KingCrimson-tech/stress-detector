"""StressLens — Streamlit frontend for student stress detection."""

import tempfile
import os
import time

import streamlit as st

from modules.module1_survey import render_survey_form
from modules.module2_emotion import detect_emotion
from modules.module4_fusion import fuse
from contracts import EmotionResult


# 1. Page config
st.set_page_config(page_title="StressLens", layout="wide")

# 2. Sidebar
with st.sidebar:
    st.title("StressLens")
    st.write("Fill the survey below. Photo is optional.")
    uploaded = st.file_uploader(
        "Upload face photo (optional)", type=["jpg", "jpeg", "png"]
    )
    survey_weight = st.slider(
        "Survey vs emotion weight", min_value=0.5, max_value=0.9, value=0.7, step=0.1
    )
    st.caption("0.7 means 70% survey, 30% face")

# 3. Main area — two columns
left_col, right_col = st.columns(2)

# Left column: survey form
with left_col:
    st.header("Survey")
    survey_result = render_survey_form()

# Right column: results
with right_col:
    st.header("Results")

    # 4. Handle case where form not yet submitted
    if survey_result is None:
        st.info("Please complete and submit the survey to see your results.")
    else:
        # Process uploaded photo or use neutral default
        if uploaded is not None:
            # Save uploaded file to a temp file for detect_emotion
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                emotion_result, emotion_status = detect_emotion(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        else:
            emotion_result = EmotionResult("neutral", 0.2)
            emotion_status = "no_photo"

        # Fuse survey and emotion results
        fusion_result = fuse(survey_result, emotion_result, survey_weight)

        # Show stress label in big coloured text
        if fusion_result.stress_label == "Low":
            st.success(f"Stress Level: **{fusion_result.stress_label}**")
        elif fusion_result.stress_label == "Medium":
            st.warning(f"Stress Level: **{fusion_result.stress_label}**")
        else:  # High
            st.error(f"Stress Level: **{fusion_result.stress_label}**")

        # Show confidence as metric
        st.metric(label="Confidence", value=f"{fusion_result.confidence:.1%}")

        # Show detected emotion/status and fallback score behavior.
        if emotion_status == "success":
            detected_text = emotion_result.dominant_emotion.title()
        elif emotion_status == "no_face":
            detected_text = "No face detected"
        elif emotion_status == "low_confidence":
            detected_text = "Face detected, low confidence"
        elif emotion_status.startswith("error"):
            detected_text = "Unavailable (model error)"
        else:
            detected_text = "Not provided"

        st.write(f"**Emotion Input:** {detected_text}")
        st.write(f"**Emotion Stress Score:** {emotion_result.emotion_score:.2f}")
        if emotion_status == "no_face":
            st.caption("No face detected. Used neutral fallback for emotion score.")
        elif emotion_status == "low_confidence":
            st.caption("Face detected with low confidence. Used neutral fallback score.")
        elif emotion_status.startswith("error"):
            st.caption(
                "Emotion model error. Used neutral fallback score. "
                f"Reason: {emotion_status}"
            )

        # Show coping resources for high stress
        if fusion_result.stress_label == "High":
            st.info(
                "**Coping Resources:**\n"
                "1. Consider speaking with a counselor or mental health professional.\n"
                "2. Try relaxation techniques like deep breathing or meditation.\n"
                "3. Reach out to friends, family, or a support group for help."
            )

        # SHAP explainability (survey-based) — kept optional and failure-tolerant.
        st.subheader("SHAP Analysis")
        st.caption("Explains the survey model only (emotion excluded).")
        with st.spinner("Generating explanation..."):
            try:
                from modules.module5_shap import generate_shap_chart

                shap_fig = generate_shap_chart(fusion_result)
                st.pyplot(shap_fig, use_container_width=True)
            except Exception:
                st.caption("SHAP analysis unavailable in this runtime.")
