"""Emotion detection module using DeepFace."""

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import requests

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from dotenv import load_dotenv

    # Prefer local-only secrets file (gitignored).
    load_dotenv(os.path.join(project_root, ".env.local"), override=False)
    # Backwards-compatible: also load .env if user still uses it locally.
    load_dotenv(os.path.join(project_root, ".env"), override=False)
except Exception:
    pass

from contracts import EmotionResult, EMOTION_TO_STRESS_WEIGHT

# If the winning DeepFace emotion score drops below this (0-100),
# treat the image as unreadable and fallback safely.
MIN_CONFIDENCE_DEEPFACE = 40.0

# Hugging Face model confidence is 0.0-1.0.
MIN_CONFIDENCE_HF = 0.35


def _neutral_result() -> EmotionResult:
    return EmotionResult(dominant_emotion="neutral", emotion_score=0.2)


def _error_status(reason: str) -> str:
    return f"error:{reason}"


def _normalize_emotion_label(label: str) -> str | None:
    value = label.strip().lower().replace(" ", "_")
    mapping = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "happiness": "happy",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
        "neutral": "neutral",
        "calm": "neutral",
    }
    return mapping.get(value)


def _has_face_opencv(image_path: str) -> bool:
    """Lightweight face presence check to avoid classifying non-face images."""
    try:
        import cv2
    except Exception:
        # If OpenCV is unavailable, do not block inference.
        return True

    image = cv2.imread(image_path)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    return len(faces) > 0


def _detect_emotion_huggingface(image_path: str) -> tuple[EmotionResult, str]:
    """Run facial emotion inference via Hugging Face Inference API."""
    if not os.path.exists(image_path):
        return _neutral_result(), _error_status("image_missing")

    # Non-face images should be reported as no_face, not neutral emotion.
    if not _has_face_opencv(image_path):
        return _neutral_result(), "no_face"

    token = (
        os.getenv("STRESSLENS_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        return _neutral_result(), _error_status("missing_token")

    model_id = os.getenv("STRESSLENS_HF_MODEL", "trpakov/vit-face-expression")
    timeout_s = float(os.getenv("STRESSLENS_HF_TIMEOUT", "25"))
    urls = [
        os.getenv("STRESSLENS_HF_API_URL", "").strip(),
        f"https://api-inference.huggingface.co/models/{model_id}",
        f"https://router.huggingface.co/hf-inference/models/{model_id}",
    ]
    urls = [u for u in urls if u]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    }

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    last_error_reason = "unknown"

    for url in urls:
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, data=image_bytes, timeout=timeout_s)
            except requests.RequestException:
                last_error_reason = "request_exception"
                if attempt == 2:
                    break
                time.sleep(1.0)
                continue

            if resp.status_code in (401, 403):
                return _neutral_result(), _error_status(f"auth_{resp.status_code}")

            if resp.status_code == 404:
                last_error_reason = "model_or_endpoint_not_found"
                break

            try:
                payload = resp.json()
            except ValueError:
                payload = None

            if isinstance(payload, dict) and "error" in payload:
                error_text = str(payload.get("error", "")).lower()
                if "loading" in error_text or resp.status_code == 503:
                    if attempt == 2:
                        last_error_reason = "model_loading_timeout"
                        break
                    wait_s = float(payload.get("estimated_time", 1.0) or 1.0)
                    time.sleep(min(max(wait_s, 1.0), 8.0))
                    continue
                return _neutral_result(), _error_status("api_error")

            if resp.status_code >= 400:
                return _neutral_result(), _error_status(f"http_{resp.status_code}")

            if payload is None:
                return _neutral_result(), _error_status("invalid_json")

            predictions = payload
            if isinstance(payload, dict) and "label" in payload and "score" in payload:
                predictions = [payload]
            elif isinstance(payload, dict) and "labels" in payload and "scores" in payload:
                labels = payload.get("labels", [])
                scores = payload.get("scores", [])
                predictions = [
                    {"label": lbl, "score": scr}
                    for lbl, scr in zip(labels, scores)
                ]
            elif isinstance(payload, list) and payload and isinstance(payload[0], list):
                predictions = payload[0]

            if not isinstance(predictions, list) or not predictions:
                return _neutral_result(), _error_status("unexpected_payload")

            best_emotion = "neutral"
            best_score = 0.0
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                raw_label = str(item.get("label", ""))
                mapped_label = _normalize_emotion_label(raw_label)
                if mapped_label is None:
                    continue
                score = float(item.get("score", 0.0))
                if score > best_score:
                    best_score = score
                    best_emotion = mapped_label

            if best_score == 0.0:
                return _neutral_result(), _error_status("no_supported_labels")

            if best_score < MIN_CONFIDENCE_HF:
                return _neutral_result(), "low_confidence"

            return (
                EmotionResult(
                    dominant_emotion=best_emotion,
                    emotion_score=EMOTION_TO_STRESS_WEIGHT.get(best_emotion, 0.2),
                ),
                "success",
            )

    return _neutral_result(), _error_status(last_error_reason)


def _detect_emotion_deepface(image_path: str) -> tuple[EmotionResult, str]:
    """Fallback local DeepFace detector."""
    # Keep TensorFlow/DeepFace startup logs minimal.
    # Set STRESSLENS_FORCE_CPU=1 to disable GPU explicitly.
    if os.getenv("STRESSLENS_FORCE_CPU", "0") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings(
        "ignore",
        message=r'.*"is" with a literal.*',
        category=SyntaxWarning,
    )

    from deepface import DeepFace

    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
    except ValueError:
        return _neutral_result(), "no_face"
    except Exception:
        return _neutral_result(), _error_status("deepface_exception")

    if isinstance(result, list):
        face_data = result[0]
    else:
        face_data = result

    emotion_scores = face_data.get("emotion", {})
    dominant_emotion = face_data.get("dominant_emotion", "neutral").lower()

    if not emotion_scores:
        return _neutral_result(), _error_status("deepface_empty_scores")

    dominant_confidence = emotion_scores.get(dominant_emotion, 0.0)
    if dominant_confidence < MIN_CONFIDENCE_DEEPFACE:
        return _neutral_result(), "low_confidence"

    stress_score = EMOTION_TO_STRESS_WEIGHT.get(dominant_emotion, 0.2)
    return EmotionResult(dominant_emotion=dominant_emotion, emotion_score=stress_score), "success"


def detect_emotion(image_path: str) -> tuple:
    """
    Detect dominant emotion from a face image and return stress-weighted score.

    Returns:
        tuple of (EmotionResult, status: str)
        status is one of: "success" | "no_face" | "low_confidence" | "error"
    """
    provider = os.getenv("STRESSLENS_EMOTION_PROVIDER", "huggingface").lower().strip()

    if provider == "deepface":
        return _detect_emotion_deepface(image_path)

    hf_result, hf_status = _detect_emotion_huggingface(image_path)
    if not hf_status.startswith("error"):
        return hf_result, hf_status

    if os.getenv("STRESSLENS_FALLBACK_TO_DEEPFACE", "1") == "1":
        return _detect_emotion_deepface(image_path)

    return hf_result, hf_status


def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to disk and return the temp path."""
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def cleanup_temp_file(path: str) -> None:
    """Delete the temp file once DeepFace is done with it."""
    try:
        os.unlink(path)
    except OSError:
        pass


if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image

    print("Test 1: plain white image (no face) -- expect no_face")
    white = Image.fromarray((np.ones((200, 200, 3)) * 255).astype("uint8"))
    white.save("/tmp/test_white.jpg")
    res, status = detect_emotion("/tmp/test_white.jpg")
    print(f"  status={status}  emotion={res.dominant_emotion}  score={res.emotion_score}")
    assert status == "no_face", f"FAIL: expected no_face, got {status}"
    print("  PASSED\n")

    print("Test 2: file that does not exist -- expect error")
    res, status = detect_emotion("/tmp/nonexistent_abc123.jpg")
    print(f"  status={status}  emotion={res.dominant_emotion}")
    assert status in ("no_face", "error"), f"FAIL: got {status}"
    print("  PASSED\n")

    if len(sys.argv) > 1:
        photo = sys.argv[1]
        print(f"Test 3: real photo at {photo}")
        res, status = detect_emotion(photo)
        print(f"  status={status}  emotion={res.dominant_emotion}  score={res.emotion_score}")

    print("All tests done.")