"""
Audio Channel — Microphone capture with speech transcription and novelty.

Captures a short audio clip when the heartbeat fires, transcribes it
using OpenAI Whisper, and computes novelty vs recent transcripts.

Does NOT stream continuously — only listens when the engine is "paying attention."
"""

from __future__ import annotations

import io
import os
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

from src.models import ChannelInput


class AudioChannel:
    def __init__(
        self,
        history_window: int = 10,
        base_weight_direct: float = 1.0,
        base_weight_overheard: float = 0.25,
        capture_seconds: float = 5.0,
        sample_rate: int = 16000,
    ):
        self.history_window = history_window
        self.base_weight_direct = base_weight_direct
        self.base_weight_overheard = base_weight_overheard
        self.capture_seconds = capture_seconds
        self.sample_rate = sample_rate
        self._transcript_history: deque[str] = deque(maxlen=history_window)
        self._available = False
        self._initialized = False

    def initialize(self) -> bool:
        """Check if audio capture is possible."""
        if self._initialized:
            return self._available
        self._initialized = True
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            default_input = sd.query_devices(kind="input")
            if default_input is not None:
                self._available = True
                print(f"   [Audio] Microphone: connected ({default_input['name'][:40]})")
            else:
                print("   [Audio] Microphone: no input device found")
                self._available = False
        except ImportError:
            print("   [Audio] Microphone: sounddevice not installed")
            self._available = False
        except Exception as e:
            print(f"   [Audio] Microphone error: {e}")
            self._available = False
        return self._available

    @property
    def is_available(self) -> bool:
        return self._available

    def capture_audio(self) -> np.ndarray | None:
        """Record a short audio clip from the microphone. Returns numpy array or None."""
        if not self._available:
            return None
        try:
            import sounddevice as sd
            print(f"   [MIC ON]  Listening for {self.capture_seconds:.0f} seconds -- talk now!")
            audio = sd.rec(
                int(self.capture_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            print(f"   [MIC OFF] Recording done.")
            return audio.flatten()
        except Exception as e:
            print(f"   [MIC OFF] Capture error: {e}")
            return None

    def has_speech(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Simple voice activity detection — is the RMS energy above threshold?"""
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > threshold

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            import wave
            temp_path = Path(tempfile.gettempdir()) / "curiosity_audio.wav"
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(str(temp_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            with open(temp_path, "rb") as f:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                )
            transcript = response.text.strip()

            temp_path.unlink(missing_ok=True)
            return transcript

        except Exception as e:
            print(f"   [Audio] Transcription error: {e}")
            return ""

    def compute_novelty(self, transcript: str) -> float:
        """Compare transcript to recent history. New topics = high novelty."""
        if not transcript:
            return 0.0

        if not self._transcript_history:
            self._transcript_history.append(transcript)
            return 0.5

        current_words = set(transcript.lower().split())
        if len(current_words) < 2:
            return 0.3

        similarities = []
        for past in self._transcript_history:
            past_words = set(past.lower().split())
            if not past_words:
                continue
            overlap = len(current_words & past_words)
            union = len(current_words | past_words)
            similarities.append(overlap / union if union > 0 else 0.0)

        self._transcript_history.append(transcript)

        if not similarities:
            return 1.0

        max_similarity = max(similarities)
        return max(0.0, min(1.0, 1.0 - max_similarity))

    async def process(self) -> ChannelInput:
        """
        Full audio pipeline: capture → detect speech → transcribe → novelty.
        """
        if not self._available:
            return ChannelInput(
                channel="audio",
                raw_content="",
                novelty=0.0,
                base_weight=0.0,
                effective_weight=0.0,
                available=False,
            )

        audio = self.capture_audio()
        if audio is None:
            return ChannelInput(
                channel="audio",
                raw_content="[capture failed]",
                novelty=0.0,
                base_weight=0.0,
                effective_weight=0.0,
                available=False,
            )

        if not self.has_speech(audio):
            print("   [Audio] Audio: silence detected")
            return ChannelInput(
                channel="audio",
                raw_content="[silence]",
                novelty=0.0,
                base_weight=0.0,
                effective_weight=0.0,
                available=True,
            )

        print("   [Audio] Audio: speech detected, transcribing...")
        transcript = await self.transcribe(audio)

        if not transcript:
            return ChannelInput(
                channel="audio",
                raw_content="[transcription empty]",
                novelty=0.0,
                base_weight=self.base_weight_overheard,
                effective_weight=0.0,
                available=True,
            )

        novelty = self.compute_novelty(transcript)
        base_weight = self.base_weight_overheard
        effective_weight = base_weight * novelty

        novelty_label = "🔴 HIGH" if novelty > 0.6 else "🟡 MED" if novelty > 0.3 else "⚪ LOW"
        print(f"   [Audio] Audio [{novelty_label} novelty={novelty:.2f}]: \"{transcript[:80]}\"")

        return ChannelInput(
            channel="audio",
            raw_content=transcript,
            novelty=novelty,
            base_weight=base_weight,
            effective_weight=effective_weight,
            available=True,
        )
