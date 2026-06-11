import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Camera,
  CirclePause,
  CirclePlay,
  Music2,
  SkipForward,
  Smile,
  Volume2,
  VolumeX,
  Webcam,
} from "lucide-react";
import "./styles.css";

const emotions = [
  { id: "happy", label: "Happy", color: "#ffcc40", tracks: ["upbeat_drive.mp3", "sunny_walk.wav"] },
  { id: "sad", label: "Sad", color: "#6fa8ff", tracks: ["slow_piano.mp3", "midnight_rain.wav"] },
  { id: "angry", label: "Angry", color: "#ff5b6e", tracks: ["heavy_pulse.mp3", "sharp_edges.wav"] },
  { id: "surprised", label: "Surprised", color: "#40d6b2", tracks: ["wide_eyes.mp3", "spark_jump.wav"] },
  { id: "fearful", label: "Fearful", color: "#c884ff", tracks: ["tense_pad.mp3", "shadow_steps.wav"] },
  { id: "disgusted", label: "Disgusted", color: "#86d36c", tracks: ["uneasy_groove.mp3", "sour_note.wav"] },
  { id: "neutral", label: "Neutral", color: "#cbd1dc", tracks: ["ambient_focus.mp3", "clean_room.wav"] },
];

function App() {
  const [activeEmotion, setActiveEmotion] = useState("happy");
  const [trackIndex, setTrackIndex] = useState(0);
  const [muted, setMuted] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [confidence, setConfidence] = useState(82);
  const [stability, setStability] = useState(68);
  const [history, setHistory] = useState(["neutral", "happy", "happy", "happy", "surprised", "happy"]);
  const videoRef = useRef(null);

  const emotion = useMemo(
    () => emotions.find((item) => item.id === activeEmotion) ?? emotions[0],
    [activeEmotion],
  );

  const currentTrack = emotion.tracks[trackIndex % emotion.tracks.length];

  useEffect(() => {
    let stream;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch {
        stream = null;
      }
    }

    startCamera();

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setConfidence((value) => clamp(value + randomStep(9), 42, 96));
      setStability((value) => clamp(value + randomStep(11), 18, 100));
      setHistory((items) => [...items.slice(-17), activeEmotion]);
    }, 1400);

    return () => window.clearInterval(timer);
  }, [activeEmotion]);

  function selectEmotion(id) {
    setActiveEmotion(id);
    setTrackIndex(0);
    setStability(32);
    setConfidence(72);
    setHistory((items) => [...items.slice(-17), id]);
  }

  function nextTrack() {
    setTrackIndex((value) => value + 1);
  }

  return (
    <main className="app" style={{ "--accent": emotion.color }}>
      <section className="topbar">
        <div>
          <h1>Emotion Music Player</h1>
          <p>React control surface for the Face Mesh mood-based music system</p>
        </div>
        <div className="statusPill">
          <Activity size={18} />
          <span>{playing ? "Live session" : "Paused"}</span>
        </div>
      </section>

      <section className="workspace">
        <section className="cameraPanel">
          <div className="panelHeader">
            <div>
              <span className="eyebrow">Camera</span>
              <h2>Live preview</h2>
            </div>
            <div className="chip">
              <Webcam size={16} />
              Browser webcam
            </div>
          </div>

          <div className="videoShell">
            <video ref={videoRef} autoPlay playsInline muted />
            <div className="scanFrame" />
            <div className="faceHint">
              <Camera size={18} />
              <span>Use the Python app for real MediaPipe detection</span>
            </div>
          </div>

          <div className="timeline">
            {history.map((item, index) => {
              const entry = emotions.find((emotionItem) => emotionItem.id === item);
              return <span key={`${item}-${index}`} style={{ backgroundColor: entry?.color }} />;
            })}
          </div>
        </section>

        <aside className="sidePanel">
          <div className="emotionHero">
            <span className="eyebrow">Detected emotion</span>
            <div className="emotionTitle">
              <Smile size={36} />
              <h2>{emotion.label}</h2>
            </div>
          </div>

          <Meter label="Confidence" value={confidence} />
          <Meter label="Music switch readiness" value={stability} />

          <div className="musicCard">
            <div className="musicIcon">
              <Music2 size={24} />
            </div>
            <div>
              <span className="eyebrow">Now playing</span>
              <h3>{currentTrack}</h3>
              <p>{emotion.tracks.length} tracks mapped to {emotion.label.toLowerCase()}</p>
            </div>
          </div>

          <div className="controls">
            <button type="button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Pause" : "Play"}>
              {playing ? <CirclePause size={20} /> : <CirclePlay size={20} />}
            </button>
            <button type="button" onClick={nextTrack} aria-label="Next track">
              <SkipForward size={20} />
            </button>
            <button type="button" onClick={() => setMuted((value) => !value)} aria-label={muted ? "Unmute" : "Mute"}>
              {muted ? <VolumeX size={20} /> : <Volume2 size={20} />}
            </button>
          </div>
        </aside>
      </section>

      <section className="emotionGrid" aria-label="Emotion playlist mapping">
        {emotions.map((item) => (
          <button
            key={item.id}
            type="button"
            className={item.id === activeEmotion ? "emotionButton active" : "emotionButton"}
            style={{ "--emotion-color": item.color }}
            onClick={() => selectEmotion(item.id)}
          >
            <span />
            <strong>{item.label}</strong>
            <small>{item.tracks.length} tracks</small>
          </button>
        ))}
      </section>
    </main>
  );
}

function Meter({ label, value }) {
  return (
    <div className="meter">
      <div>
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="meterTrack">
        <span style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function randomStep(range) {
  return Math.round(Math.random() * range * 2 - range);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

createRoot(document.getElementById("root")).render(<App />);
