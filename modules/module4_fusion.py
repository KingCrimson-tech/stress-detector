"""Module 4: Fusion of survey and emotion results into a final stress prediction."""

import numpy as np

from contracts import (
    EmotionResult,
    FusionResult,
    STRESS_LABELS,
    SurveyResult,
)


def fuse(
    survey_result: SurveyResult,
    emotion_result: EmotionResult,
    survey_weight: float = 0.7,
) -> FusionResult:
    """
    Fuse survey and emotion results into a final stress prediction.

    Args:
        survey_result: The result from the survey prediction module.
        emotion_result: The result from the emotion detection module.
        survey_weight: Weight for the survey probability (default 0.7).

    Returns:
        FusionResult with the combined stress label and confidence.
    """
    emotion_weight = 1.0 - survey_weight

    # Convert emotion_score to a 3-class probability vector
    es = emotion_result.emotion_score
    emotion_proba = np.array([1.0 - es, es * 0.4, es * 0.6])
    emotion_proba = emotion_proba / emotion_proba.sum()  # normalise

    # Weighted fusion
    fused = (survey_result.survey_proba * survey_weight +
             emotion_proba * emotion_weight)

    stress_label = STRESS_LABELS[int(np.argmax(fused))]
    confidence = float(np.max(fused))

    return FusionResult(
        stress_label=stress_label,
        confidence=confidence,
        survey_weight=survey_weight,
        emotion_weight=emotion_weight,
        survey_proba=survey_result.survey_proba,
        raw_survey_vector=survey_result.raw_vector,
        emotion_result=emotion_result,
    )


if __name__ == "__main__":
    # Test with dummy data
    dummy_survey = SurveyResult(
        raw_vector=np.array([3] * 21, dtype=float),
        survey_proba=np.array([0.2, 0.5, 0.3]),
    )
    dummy_emotion = EmotionResult(
        dominant_emotion="sad",
        emotion_score=0.75,
    )

    result = fuse(dummy_survey, dummy_emotion)

    print("=== Fusion Test ===")
    print(f"Survey proba: {dummy_survey.survey_proba}")
    print(f"Emotion score: {dummy_emotion.emotion_score}")
    print(f"Emotion: {dummy_emotion.dominant_emotion}")
    print()
    print(f"Stress label: {result.stress_label}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Survey weight: {result.survey_weight}")
    print(f"Emotion weight: {result.emotion_weight}")
