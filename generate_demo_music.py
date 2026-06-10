from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 44_100
DURATION_SECONDS = 7
MUSIC_ROOT = Path("music")

EMOTION_PATTERNS = {
    "happy": (440.0, 554.37, 659.25, 880.0),
    "sad": (220.0, 261.63, 329.63, 196.0),
    "angry": (110.0, 146.83, 164.81, 196.0),
    "surprised": (523.25, 659.25, 783.99, 1046.5),
    "fearful": (246.94, 277.18, 329.63, 369.99),
    "disgusted": (185.0, 207.65, 233.08, 277.18),
    "neutral": (261.63, 329.63, 392.0, 523.25),
}


def synthesize_tone(frequencies: tuple[float, ...], duration: int = DURATION_SECONDS) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    signal = np.zeros_like(t)

    segment_length = len(t) // len(frequencies)
    for index, frequency in enumerate(frequencies):
        start = index * segment_length
        end = len(t) if index == len(frequencies) - 1 else (index + 1) * segment_length
        segment_t = t[start:end]
        carrier = np.sin(2.0 * math.pi * frequency * segment_t)
        overtone = 0.35 * np.sin(2.0 * math.pi * frequency * 2.0 * segment_t)
        signal[start:end] = carrier + overtone

    envelope = np.ones_like(signal)
    fade_samples = int(0.08 * SAMPLE_RATE)
    envelope[:fade_samples] = np.linspace(0.0, 1.0, fade_samples)
    envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples)
    signal *= envelope

    signal /= max(np.max(np.abs(signal)), 1e-9)
    return (signal * 0.35 * np.iinfo(np.int16).max).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


def main() -> None:
    for emotion, frequencies in EMOTION_PATTERNS.items():
        output = MUSIC_ROOT / emotion / f"demo_{emotion}.wav"
        write_wav(output, synthesize_tone(frequencies))
        print(f"Created {output}")


if __name__ == "__main__":
    main()
