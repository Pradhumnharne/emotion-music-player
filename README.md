# Emotion-Based Music Player

OpenCV + MediaPipe Face Mesh project that detects your facial emotion in real time via webcam and plays local music matching the detected mood.

The detector does not use a trained emotion model. It computes geometric ratios from MediaPipe's 468 face landmarks, smooths predictions over recent frames, and switches tracks only after the emotion remains stable.

## Project Structure

```text
emotion_music_player/
|
├── emotion_music_player.py   # Main app
├── emotion_detector.py       # MediaPipe Face Mesh + emotion rules
├── generate_demo_music.py    # Generates test .wav files
├── requirements.txt
└── music/
    ├── happy/
    ├── sad/
    ├── angry/
    ├── surprised/
    ├── fearful/
    ├── disgusted/
    └── neutral/
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you previously installed a newer MediaPipe release and see `module 'mediapipe' has no attribute 'solutions'`, reinstall the pinned version:

```bash
py -m pip install --force-reinstall mediapipe==0.10.14
```

Generate demo tones:

```bash
python generate_demo_music.py
```

Run the player:

```bash
python emotion_music_player.py
```

## Controls

| Key | Action |
| --- | --- |
| `Q` | Quit |
| `N` | Skip to next track for current emotion |
| `M` | Toggle mute |

## Interface

The player opens a polished desktop-style OpenCV interface with:

- A large live webcam preview with Face Mesh overlay
- A right-side emotion and music status panel
- Confidence and music-switch readiness meters
- A color-coded timeline of recent emotion predictions
- Compact keyboard controls for quit, next track, and mute

## Configuration

Edit these constants in `emotion_music_player.py`:

```python
EMOTION_HOLD_FRAMES = 30
SMOOTHING_WINDOW = 15
FADE_DURATION_MS = 1500
WEBCAM_INDEX = 0
```

## Music

Drop `.mp3`, `.wav`, `.ogg`, or `.flac` files into the matching emotion folder under `music/`.

Supported emotion folders:

- `happy`
- `sad`
- `angry`
- `surprised`
- `fearful`
- `disgusted`
- `neutral`

## Troubleshooting

| Issue | Fix |
| --- | --- |
| No webcam found | Change `WEBCAM_INDEX` to `1` or `2` |
| No sound | Run `python generate_demo_music.py` or add music files |
| Low FPS | Lower camera resolution in `emotion_music_player.py` |
| Wrong emotion | Improve lighting and face the camera directly |
| Missing module | Run `pip install -r requirements.txt` |
