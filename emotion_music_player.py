from __future__ import annotations

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
    "happy": (40, 206, 255),
    "sad": (235, 145, 72),
    "angry": (78, 88, 242),
    "surprised": (68, 205, 172),
    "fearful": (205, 128, 240),
    "disgusted": (104, 198, 94),
    "neutral": (206, 211, 218),
}

APP_SIZE = (1280, 720)
VIDEO_RECT = (28, 86, 884, 498)
PANEL_RECT = (936, 86, 316, 498)
WINDOW_TITLE = "Emotion Music Player"


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


def render_app_frame(camera_frame: np.ndarray, result: EmotionResult, player: AudioPlayer, library: MusicLibrary) -> np.ndarray:
    app_width, app_height = APP_SIZE
    canvas = np.full((app_height, app_width, 3), (21, 24, 31), dtype=np.uint8)
    draw_header(canvas, result)
    draw_video_panel(canvas, camera_frame)
    draw_side_panel(canvas, result, player, library)
    draw_history(canvas, result.history)
    draw_footer(canvas, player)
    return canvas


def draw_header(canvas: np.ndarray, result: EmotionResult) -> None:
    emotion = result.emotion
    color = EMOTION_COLORS[emotion]
    label = EMOTION_LABELS[emotion]

    cv2.putText(canvas, "Emotion Music Player", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (242, 244, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Face Mesh mood detection", (30, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 158, 170), 1, cv2.LINE_AA)
    draw_pill(canvas, (1030, 26), (222, 36), color, f"Detected: {label}", text_color=(18, 20, 24))


def draw_video_panel(canvas: np.ndarray, camera_frame: np.ndarray) -> None:
    x, y, width, height = VIDEO_RECT
    draw_panel(canvas, (x - 8, y - 8, width + 16, height + 16), fill=(31, 35, 45), border=(59, 66, 80))

    video = cv2.resize(camera_frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas[y : y + height, x : x + width] = video
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (74, 82, 98), 1)


def draw_side_panel(canvas: np.ndarray, result: EmotionResult, player: AudioPlayer, library: MusicLibrary) -> None:
    x, y, width, height = PANEL_RECT
    draw_panel(canvas, PANEL_RECT, fill=(29, 33, 43), border=(61, 69, 84))

    emotion = result.emotion
    color = EMOTION_COLORS[emotion]
    label = EMOTION_LABELS[emotion]
    confidence = int(result.confidence * 100)
    stability = min(result.stable_frames / EMOTION_HOLD_FRAMES, 1.0)
    track_count = len(library.tracks.get(emotion, []))

    cv2.putText(canvas, "Current emotion", (x + 24, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (155, 164, 178), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (x + 24, y + 86), cv2.FONT_HERSHEY_SIMPLEX, 1.28, color, 2, cv2.LINE_AA)

    draw_metric(canvas, (x + 24, y + 124), "Confidence", confidence / 100.0, f"{confidence}%", color)
    draw_metric(canvas, (x + 24, y + 190), "Music switch readiness", stability, f"{int(stability * 100)}%", color)

    status = "Ready to switch" if stability >= 1.0 else f"Hold {max(0, EMOTION_HOLD_FRAMES - result.stable_frames)} frames"
    draw_pill(canvas, (x + 24, y + 262), (180, 32), color if stability >= 1.0 else (75, 82, 96), status)

    track_name = player.current_track.name if player.current_track else f"No {emotion} track found"
    cv2.putText(canvas, "Now playing", (x + 24, y + 336), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (155, 164, 178), 1, cv2.LINE_AA)
    draw_wrapped_text(canvas, track_name, (x + 24, y + 370), width - 48, 0.55, (238, 241, 246), max_lines=2)
    cv2.putText(canvas, f"{track_count} track(s) for {label.lower()}", (x + 24, y + 438), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (155, 164, 178), 1, cv2.LINE_AA)


def draw_history(canvas: np.ndarray, history: tuple[str, ...]) -> None:
    x, y, width, height = 28, 610, 884, 58
    draw_panel(canvas, (x - 8, y - 8, width + 16, height + 16), fill=(29, 33, 43), border=(61, 69, 84))
    cv2.putText(canvas, "Emotion timeline", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (155, 164, 178), 1, cv2.LINE_AA)

    if not history:
        return
    segment_w, segment_h = 44, 14
    for index, emotion in enumerate(history[-SMOOTHING_WINDOW:]):
        left = x + index * (segment_w + 10)
        cv2.rectangle(canvas, (left, y + 34), (left + segment_w, y + 34 + segment_h), EMOTION_COLORS[emotion], -1)


def draw_footer(canvas: np.ndarray, player: AudioPlayer) -> None:
    muted_label = "Muted" if player.muted else "Sound on"
    controls = (("Q", "Quit"), ("N", "Next track"), ("M", muted_label))
    x = 936
    y = 620
    cv2.putText(canvas, "Controls", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (155, 164, 178), 1, cv2.LINE_AA)
    for index, (key, label) in enumerate(controls):
        top = y + 22 + index * 38
        draw_key(canvas, (x, top), key, label)


def draw_panel(canvas: np.ndarray, rect: tuple[int, int, int, int], fill: tuple[int, int, int], border: tuple[int, int, int]) -> None:
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), border, 1)


def draw_metric(
    canvas: np.ndarray,
    origin: tuple[int, int],
    label: str,
    value: float,
    value_label: str,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    width, height = 244, 14
    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (155, 164, 178), 1, cv2.LINE_AA)
    cv2.putText(canvas, value_label, (x + 198, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (232, 235, 240), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y + 18), (x + width, y + 18 + height), (58, 64, 78), -1)
    cv2.rectangle(canvas, (x, y + 18), (x + int(width * np.clip(value, 0.0, 1.0)), y + 18 + height), color, -1)
    cv2.rectangle(canvas, (x, y + 18), (x + width, y + 18 + height), (83, 91, 108), 1)


def draw_pill(
    canvas: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    fill: tuple[int, int, int],
    text: str,
    text_color: tuple[int, int, int] = (242, 244, 248),
) -> None:
    x, y = origin
    width, height = size
    cv2.rectangle(canvas, (x, y), (x + width, y + height), fill, -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    text_x = x + max(10, (width - text_size[0]) // 2)
    text_y = y + (height + text_size[1]) // 2 - 2
    cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)


def draw_key(canvas: np.ndarray, origin: tuple[int, int], key: str, label: str) -> None:
    x, y = origin
    cv2.rectangle(canvas, (x, y), (x + 36, y + 26), (229, 233, 240), -1)
    cv2.putText(canvas, key, (x + 11, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (24, 27, 34), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (x + 50, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 235, 240), 1, cv2.LINE_AA)


def draw_wrapped_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    scale: float,
    color: tuple[int, int, int],
    max_lines: int,
) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(3, len(lines[-1]) - 3)] + "..."

    x, y = origin
    for index, line in enumerate(lines):
        line = fit_text(line, max_width, scale)
        cv2.putText(canvas, line, (x, y + index * 24), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit_text(text: str, max_width: int, scale: float) -> str:
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= max_width:
        return text

    trimmed = text
    while len(trimmed) > 3:
        candidate = trimmed[:-1] + "..."
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= max_width:
            return candidate
        trimmed = trimmed[:-1]
    return "..."


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

            app_frame = render_app_frame(frame, result, player, library)
            cv2.imshow(WINDOW_TITLE, app_frame)

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
