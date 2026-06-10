from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pygame

from emotion_detector import EMOTIONS, EmotionDetector, EmotionResult


EMOTION_HOLD_FRAMES = 30
SMOOTHING_WINDOW = 15
FADE_DURATION_MS = 1500
WEBCAM_INDEX = 0
MUSIC_ROOT = Path("music")
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".flac"}

EMOTION_LABELS = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "surprised": "Surprised",
    "fearful": "Fearful",
    "disgusted": "Disgusted",
    "neutral": "Neutral",
}

EMOTION_COLORS = {
    "happy": (0, 210, 255),
    "sad": (255, 150, 80),
    "angry": (70, 70, 255),
    "surprised": (0, 190, 160),
    "fearful": (210, 120, 255),
    "disgusted": (80, 200, 80),
    "neutral": (210, 210, 210),
}


class MusicLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tracks: Dict[str, List[Path]] = {}
        self.indexes: Dict[str, int] = {}
        self.refresh()

    def refresh(self) -> None:
        for emotion in EMOTIONS:
            folder = self.root / emotion
            folder.mkdir(parents=True, exist_ok=True)
            files = sorted(path for path in folder.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO)
            self.tracks[emotion] = files
            self.indexes.setdefault(emotion, 0)

    def current(self, emotion: str) -> Optional[Path]:
        tracks = self.tracks.get(emotion, [])
        if not tracks:
            return None
        self.indexes[emotion] %= len(tracks)
        return tracks[self.indexes[emotion]]

    def next(self, emotion: str) -> Optional[Path]:
        tracks = self.tracks.get(emotion, [])
        if not tracks:
            return None
        self.indexes[emotion] = (self.indexes[emotion] + 1) % len(tracks)
        return self.current(emotion)


class AudioPlayer:
    def __init__(self, library: MusicLibrary) -> None:
        pygame.mixer.init()
        self.library = library
        self.current_emotion: Optional[str] = None
        self.current_track: Optional[Path] = None
        self.muted = False
        pygame.mixer.music.set_volume(0.0 if self.muted else 0.85)

    def switch_to(self, emotion: str, force_next: bool = False) -> None:
        track = self.library.next(emotion) if force_next else self.library.current(emotion)
        if track is None:
            self.current_emotion = emotion
            self.current_track = None
            pygame.mixer.music.fadeout(FADE_DURATION_MS)
            return

        if not force_next and self.current_track == track and self.current_emotion == emotion:
            return

        pygame.mixer.music.fadeout(FADE_DURATION_MS)
        pygame.mixer.music.load(str(track))
        pygame.mixer.music.play(loops=-1, fade_ms=FADE_DURATION_MS)
        pygame.mixer.music.set_volume(0.0 if self.muted else 0.85)
        self.current_emotion = emotion
        self.current_track = track

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        pygame.mixer.music.set_volume(0.0 if self.muted else 0.85)

    def stop(self) -> None:
        pygame.mixer.music.fadeout(300)
        pygame.mixer.quit()


def draw_hud(frame: np.ndarray, result: EmotionResult, player: AudioPlayer) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 92), (18, 18, 22), -1)
    cv2.rectangle(overlay, (0, height - 58), (width, height), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    emotion = result.emotion
    color = EMOTION_COLORS[emotion]
    label = EMOTION_LABELS[emotion]
    confidence = int(result.confidence * 100)
    stability = min(result.stable_frames / EMOTION_HOLD_FRAMES, 1.0)

    cv2.putText(frame, label, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence}%", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = width - 260, 26, 210, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (85, 85, 90), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * result.confidence), bar_y + bar_h), color, -1)
    cv2.putText(frame, "Stability", (bar_x, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (bar_x + 86, 60), (bar_x + 86 + int(124 * stability), 74), color, -1)
    cv2.rectangle(frame, (bar_x + 86, 60), (bar_x + 210, 74), (85, 85, 90), 1)

    draw_history(frame, result.history, height)

    track_name = player.current_track.name if player.current_track else f"No {emotion} track found"
    muted = "Muted" if player.muted else "Sound"
    footer = f"Now Playing: {track_name}    Q: Quit   N: Next   M: {muted}"
    cv2.putText(frame, footer, (22, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)


def draw_history(frame: np.ndarray, history: tuple[str, ...], height: int) -> None:
    if not history:
        return
    x, y = 22, height - 88
    segment_w, segment_h = 18, 14
    for index, emotion in enumerate(history[-SMOOTHING_WINDOW:]):
        left = x + index * (segment_w + 3)
        cv2.rectangle(frame, (left, y), (left + segment_w, y + segment_h), EMOTION_COLORS[emotion], -1)


def main() -> None:
    library = MusicLibrary(MUSIC_ROOT)
    player = AudioPlayer(library)
    detector = EmotionDetector(smoothing_window=SMOOTHING_WINDOW, stable_frames_required=EMOTION_HOLD_FRAMES)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        detector.close()
        player.stop()
        raise RuntimeError(f"No webcam found at index {WEBCAM_INDEX}")

    last_applied_emotion: Optional[str] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = detector.detect(frame)

            if result.stable_frames >= EMOTION_HOLD_FRAMES and result.emotion != last_applied_emotion:
                player.switch_to(result.emotion)
                last_applied_emotion = result.emotion

            draw_hud(frame, result, player)
            cv2.imshow("Emotion Music Player", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                player.toggle_mute()
            if key == ord("n"):
                player.switch_to(result.emotion, force_next=True)
                last_applied_emotion = result.emotion
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        player.stop()


if __name__ == "__main__":
    main()
