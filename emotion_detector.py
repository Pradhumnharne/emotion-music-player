from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


if not hasattr(mp, "solutions"):
    raise ImportError(
        "This project needs MediaPipe's classic Face Mesh API, but the installed "
        f"mediapipe {getattr(mp, '__version__', 'unknown')} package does not expose "
        "`mp.solutions`. Reinstall the compatible version with:\n"
        "  py -m pip install --force-reinstall mediapipe==0.10.14"
    )


EMOTIONS = ("happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral")


@dataclass(frozen=True)
class EmotionResult:
    emotion: str
    confidence: float
    stable_frames: int
    history: Tuple[str, ...]
    features: Dict[str, float]


class EmotionDetector:
    """Rule-based facial emotion detector using MediaPipe Face Mesh landmarks."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    MOUTH = (61, 13, 291, 14)
    MOUTH_CORNERS = (61, 291)
    INNER_BROWS = (70, 300)
    OUTER_BROWS = (105, 334)
    CHEEKS = (205, 425)
    NOSE = 1
    UPPER_LIP = 13
    LOWER_LIP = 14
    CHIN = 152

    def __init__(self, smoothing_window: int = 15, stable_frames_required: int = 30) -> None:
        self.smoothing_window = smoothing_window
        self.stable_frames_required = stable_frames_required
        self.history: Deque[str] = deque(maxlen=smoothing_window)
        self.last_smoothed = "neutral"
        self.stable_frames = 0

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._drawing = mp.solutions.drawing_utils
        self._mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION

    def close(self) -> None:
        self._face_mesh.close()

    def detect(self, frame: np.ndarray, draw_landmarks: bool = True) -> EmotionResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return self._update("neutral", 0.0, {})

        landmarks = result.multi_face_landmarks[0]
        points = self._landmarks_to_points(landmarks, frame.shape)
        features = self._extract_features(points)
        emotion, confidence = self._classify(features)
        emotion_result = self._update(emotion, confidence, features)

        if draw_landmarks:
            self._drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=self._mesh_connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )

        return emotion_result

    def _update(self, emotion: str, confidence: float, features: Dict[str, float]) -> EmotionResult:
        self.history.append(emotion)
        counts = Counter(self.history)
        smoothed, votes = counts.most_common(1)[0]
        smoothed_confidence = max(confidence, votes / max(1, len(self.history)))

        if smoothed == self.last_smoothed:
            self.stable_frames += 1
        else:
            self.last_smoothed = smoothed
            self.stable_frames = 1

        return EmotionResult(
            emotion=smoothed,
            confidence=float(np.clip(smoothed_confidence, 0.0, 1.0)),
            stable_frames=self.stable_frames,
            history=tuple(self.history),
            features=features,
        )

    @staticmethod
    def _landmarks_to_points(landmarks, shape: Tuple[int, int, int]) -> np.ndarray:
        height, width = shape[:2]
        return np.array([(lm.x * width, lm.y * height, lm.z * width) for lm in landmarks.landmark], dtype=np.float32)

    def _extract_features(self, points: np.ndarray) -> Dict[str, float]:
        face_width = self._distance(points[234], points[454])
        face_height = self._distance(points[10], points[self.CHIN])
        scale = max(face_width, face_height, 1.0)

        left_ear = self._eye_aspect_ratio(points, self.LEFT_EYE)
        right_ear = self._eye_aspect_ratio(points, self.RIGHT_EYE)
        eye_aspect = (left_ear + right_ear) / 2.0

        mouth_width = self._distance(points[self.MOUTH[0]], points[self.MOUTH[2]])
        mouth_open = self._distance(points[self.MOUTH[1]], points[self.MOUTH[3]])
        mouth_aspect = mouth_open / max(mouth_width, 1.0)

        mouth_center_y = (points[self.MOUTH[1]][1] + points[self.MOUTH[3]][1]) / 2.0
        corner_y = (points[self.MOUTH_CORNERS[0]][1] + points[self.MOUTH_CORNERS[1]][1]) / 2.0
        mouth_curve = (mouth_center_y - corner_y) / scale

        brow_y = (points[self.INNER_BROWS[0]][1] + points[self.INNER_BROWS[1]][1]) / 2.0
        eye_y = (points[159][1] + points[386][1]) / 2.0
        brow_raise = (eye_y - brow_y) / scale

        brow_gap = self._distance(points[self.INNER_BROWS[0]], points[self.INNER_BROWS[1]]) / scale
        cheek_nose = (
            self._distance(points[self.CHEEKS[0]], points[self.NOSE])
            + self._distance(points[self.CHEEKS[1]], points[self.NOSE])
        ) / (2.0 * scale)
        lip_raise = (points[self.LOWER_LIP][1] - points[self.UPPER_LIP][1]) / scale

        return {
            "eye_aspect": eye_aspect,
            "mouth_aspect": mouth_aspect,
            "mouth_curve": mouth_curve,
            "brow_raise": brow_raise,
            "brow_gap": brow_gap,
            "cheek_nose": cheek_nose,
            "lip_raise": lip_raise,
        }

    @staticmethod
    def _classify(features: Dict[str, float]) -> Tuple[str, float]:
        eye = features["eye_aspect"]
        mouth = features["mouth_aspect"]
        curve = features["mouth_curve"]
        brow_raise = features["brow_raise"]
        brow_gap = features["brow_gap"]
        cheek_nose = features["cheek_nose"]
        lip_raise = features["lip_raise"]

        scores = {
            "happy": 1.8 * EmotionDetector._above(curve, 0.010, 0.045)
            + 0.9 * EmotionDetector._above(mouth, 0.18, 0.42),
            "sad": 1.6 * EmotionDetector._below(curve, -0.012, -0.045)
            + 0.5 * EmotionDetector._below(brow_raise, 0.075, 0.045),
            "angry": 1.5 * EmotionDetector._below(brow_gap, 0.19, 0.12)
            + 0.8 * EmotionDetector._below(curve, -0.004, -0.035),
            "surprised": 1.4 * EmotionDetector._above(mouth, 0.32, 0.65)
            + 1.2 * EmotionDetector._above(eye, 0.24, 0.34)
            + 0.7 * EmotionDetector._above(brow_raise, 0.095, 0.15),
            "fearful": 1.0 * EmotionDetector._above(eye, 0.25, 0.36)
            + 0.9 * EmotionDetector._above(brow_raise, 0.09, 0.15)
            + 0.5 * EmotionDetector._below(mouth, 0.28, 0.12),
            "disgusted": 1.1 * EmotionDetector._below(cheek_nose, 0.36, 0.27)
            + 0.9 * EmotionDetector._above(lip_raise, 0.035, 0.08)
            + 0.5 * EmotionDetector._below(curve, 0.0, -0.035),
            "neutral": 0.55,
        }

        emotion = max(scores, key=scores.get)
        score = scores[emotion]
        total = sum(max(value, 0.0) for value in scores.values())
        confidence = score / total if total else 0.0

        if score < 0.75:
            return "neutral", 0.55
        return emotion, confidence

    @staticmethod
    def _eye_aspect_ratio(points: np.ndarray, indices: Iterable[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [points[i] for i in indices]
        vertical = EmotionDetector._distance(p2, p6) + EmotionDetector._distance(p3, p5)
        horizontal = 2.0 * EmotionDetector._distance(p1, p4)
        return vertical / max(horizontal, 1.0)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    @staticmethod
    def _above(value: float, start: float, full: float) -> float:
        return float(np.clip((value - start) / max(full - start, 1e-6), 0.0, 1.0))

    @staticmethod
    def _below(value: float, start: float, full: float) -> float:
        return float(np.clip((start - value) / max(start - full, 1e-6), 0.0, 1.0))
