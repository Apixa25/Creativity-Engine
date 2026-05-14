"""
Audio Channel — Microphone capture with speech transcription and novelty.

Captures a short audio clip when the heartbeat fires, transcribes it
using OpenAI Whisper, and computes novelty vs recent transcripts.

Does NOT stream continuously — only listens when the engine is "paying attention."
"""

from __future__ import annotations

import io
import os
from collections import deque

import numpy as np

from src.models import ChannelInput


class AudioChannel:
    def __init__(
        self,
        history_window: int = 10,
        base_weight_direct: float = 1.0,
        base_weight_overheard: float = 0.25,
        capture_seconds: float = 2.0,
        sample_rate: int = 16000,
        api_key: str = "",
        device_index: int | None = None,
        vad_threshold: float = 0.003,
    ):
        self.history_window = history_window
        self.base_weight_direct = base_weight_direct
        self.base_weight_overheard = base_weight_overheard
        self.capture_seconds = capture_seconds
        self.sample_rate = sample_rate
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.device_index = device_index
        self.vad_threshold = vad_threshold
        self._transcript_history: deque[str] = deque(maxlen=history_window)
        self._available = False
        self._initialized = False
        self._whisper_client = None
        self.paused = False

    @staticmethod
    def list_devices() -> list[dict]:
        """List all available audio input devices. Returns list of {index, name, channels, sample_rate}."""
        devices = []
        try:
            import sounddevice as sd
        except ImportError:
            return devices
        all_devices = sd.query_devices()
        for i, dev in enumerate(all_devices):
            if dev["max_input_channels"] > 0:
                devices.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": int(dev["default_samplerate"]),
                    "is_default": dev == sd.query_devices(kind="input"),
                })
        return devices

    def initialize(self) -> bool:
        """Check if audio capture is possible."""
        if self._initialized:
            return self._available
        self._initialized = True
        try:
            import sounddevice as sd
            if self.device_index is not None:
                device_info = sd.query_devices(self.device_index)
                if device_info["max_input_channels"] > 0:
                    sd.default.device[0] = self.device_index
                    self._available = True
                    print(f"   [Audio] Microphone: connected (device {self.device_index}: {device_info['name'][:40]})")
                else:
                    print(f"   [Audio] Device {self.device_index} ({device_info['name']}) is not an input device")
                    self._available = False
            else:
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

    def capture_audio(self, quiet: bool = False) -> np.ndarray | None:
        """Record a short fixed-duration audio clip. Used by heartbeat perception."""
        if not self._available:
            return None
        try:
            import sounddevice as sd
            if not quiet:
                print(f"   [MIC ON]  Listening for {self.capture_seconds:.0f} seconds -- talk now!")
            audio = sd.rec(
                int(self.capture_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            if not quiet:
                print(f"   [MIC OFF] Recording done.")
            return audio.flatten()
        except Exception as e:
            if not quiet:
                print(f"   [MIC OFF] Capture error: {e}")
            return None

    def capture_smart(
        self,
        probe_seconds: float = 0.5,
        max_seconds: float = 10.0,
        silence_timeout: float = 1.2,
        quiet: bool = True,
        debug_rms: bool = False,
    ) -> np.ndarray | None:
        """Voice-activity-driven capture: waits for speech, records until
        the speaker pauses, then returns the full utterance as one array.

        probe_seconds  — length of each mini-recording chunk
        max_seconds    — hard cap on total recording length
        silence_timeout — stop after this many seconds of silence following speech
        debug_rms      — print RMS levels for threshold tuning
        """
        if not self._available:
            return None
        try:
            import sounddevice as sd

            chunk_samples = int(probe_seconds * self.sample_rate)
            chunks: list[np.ndarray] = []
            speech_started = False
            silent_since = 0.0
            total_seconds = 0.0

            while total_seconds < max_seconds:
                if self.paused:
                    if chunks:
                        break
                    return None

                chunk = sd.rec(chunk_samples, samplerate=self.sample_rate,
                               channels=1, dtype="float32")
                sd.wait()
                flat = chunk.flatten()
                total_seconds += probe_seconds

                rms = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))
                is_speech = rms > self.vad_threshold

                if debug_rms:
                    bar = "#" * min(int(rms * 2000), 50)
                    marker = " << SPEECH" if is_speech else ""
                    print(f"   [RMS] {rms:.5f} [{bar}]{marker}")

                if is_speech:
                    if not speech_started:
                        speech_started = True
                        self._play_listening_tone()
                        if not quiet:
                            print(f"   * Recording...")
                    chunks.append(flat)
                    silent_since = 0.0
                elif speech_started:
                    chunks.append(flat)
                    silent_since += probe_seconds
                    if silent_since >= silence_timeout:
                        break

            if not chunks:
                return None

            return np.concatenate(chunks)

        except Exception as e:
            if not quiet:
                print(f"   [Audio] Smart capture error: {e}")
            return None

    def _play_listening_tone(self) -> None:
        """Play a short audible tone so the user knows recording has started.
        Uses winsound on Windows (separate from sounddevice) to avoid
        conflicts with ongoing audio recording."""
        try:
            import sys
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880, 120)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass

    def has_speech(self, audio: np.ndarray, threshold: float | None = None) -> bool:
        """Voice activity detection — is the RMS energy above threshold?"""
        audio_f = audio.astype(np.float64)
        rms = np.sqrt(np.mean(audio_f ** 2))
        t = threshold if threshold is not None else self.vad_threshold
        return float(rms) > t

    def _get_whisper_client(self):
        """Reuse a single AsyncOpenAI client for all transcriptions."""
        if self._whisper_client is None:
            from openai import AsyncOpenAI
            self._whisper_client = AsyncOpenAI(api_key=self._api_key or None)
        return self._whisper_client

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using OpenAI Whisper API. Uses in-memory buffer."""
        try:
            import wave

            buf = io.BytesIO()
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

            buf.seek(0)
            buf.name = "audio.wav"

            client = self._get_whisper_client()
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                language="en",
            )
            return response.text.strip()

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

    async def quick_capture_and_transcribe(self) -> str:
        """
        Capture audio and transcribe in one step. Used by the background listener.
        Returns the transcript string, or empty string if silence/failure.
        """
        if not self._available:
            return ""

        audio = self.capture_audio()
        if audio is None:
            return ""

        if not self.has_speech(audio):
            return ""

        return await self.transcribe(audio)

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
