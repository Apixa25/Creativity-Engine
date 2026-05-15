"""
Voice Output — Text-to-Speech for the Creativity Engine.

Gives the engine an actual voice so it speaks its interjections, observations,
and direct responses aloud through the speakers — like a friend sitting next to you.

Uses OpenAI's TTS API (tts-1 for low latency, tts-1-hd for higher quality).
Audio is played asynchronously so it doesn't block the engine's main loop.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass
class VoiceConfig:
    enabled: bool = True
    model: str = "tts-1"          # "tts-1" (fast) or "tts-1-hd" (quality)
    voice: str = "nova"           # alloy, echo, fable, onyx, nova, shimmer
    speed: float = 1.0            # 0.25 to 4.0
    response_format: str = "pcm"  # pcm for direct playback, mp3/opus for saving
    api_key_env: str = "OPENAI_API_KEY"


VOICE_DESCRIPTIONS = {
    "alloy": "neutral and balanced",
    "echo": "deeper, warm male",
    "fable": "warm and expressive, British-tinged",
    "onyx": "deep and authoritative",
    "nova": "friendly and warm female — great for companion personality",
    "shimmer": "soft and pleasant female",
}


class VoiceOutput:
    """Text-to-speech output using OpenAI's TTS API.

    Plays audio through the system speakers using sounddevice.
    Designed to run asynchronously alongside the engine's heartbeat loop.
    """

    def __init__(self, config: VoiceConfig | None = None):
        self.cfg = config or VoiceConfig()
        self._api_key = os.environ.get(self.cfg.api_key_env, "")
        self._client = None
        self._available = False
        self._speaking = False
        self._current_task: asyncio.Task | None = None

    def initialize(self) -> bool:
        """Check that TTS dependencies are available."""
        if not self.cfg.enabled:
            print("   [Voice] TTS: disabled in config")
            return False

        if not self._api_key:
            print("   [Voice] TTS: no API key found — voice disabled")
            return False

        try:
            import sounddevice  # noqa: F401
        except ImportError:
            print("   [Voice] TTS: sounddevice not installed — voice disabled")
            print("           Install with: pip install sounddevice")
            return False

        self._available = True
        voice_desc = VOICE_DESCRIPTIONS.get(self.cfg.voice, self.cfg.voice)
        model_label = "HD" if "hd" in self.cfg.model else "Standard"
        print(f"   [Voice] TTS: ready! Voice: {self.cfg.voice} ({voice_desc})")
        print(f"   [Voice] Model: {self.cfg.model} ({model_label}), Speed: {self.cfg.speed}x")
        return True

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def speak(self, text: str, wait: bool = True) -> None:
        """Convert text to speech and play through speakers.

        Args:
            text: The text to speak aloud.
            wait: If True, blocks until speech finishes. If False, plays in background.
        """
        if not self._available or not text.strip():
            return

        if self._speaking:
            await self.stop()

        if wait:
            await self._speak_impl(text)
        else:
            self._current_task = asyncio.create_task(self._speak_impl(text))

    async def _speak_impl(self, text: str) -> None:
        """Internal: generate TTS audio and play it."""
        self._speaking = True
        try:
            client = await self._get_client()

            response = await client.audio.speech.create(
                model=self.cfg.model,
                voice=self.cfg.voice,
                input=text,
                speed=self.cfg.speed,
                response_format="pcm",
            )

            pcm_bytes = response.content
            await self._play_pcm(pcm_bytes)

        except Exception as e:
            print(f"   [Voice] TTS error: {e}")
        finally:
            self._speaking = False

    async def _play_pcm(self, pcm_bytes: bytes) -> None:
        """Play raw PCM audio (24kHz, 16-bit mono) through speakers."""
        import numpy as np
        import sounddevice as sd

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sample_rate = 24000  # OpenAI TTS outputs 24kHz PCM

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_blocking, samples, sample_rate)

    @staticmethod
    def _play_blocking(samples, sample_rate: int) -> None:
        """Blocking playback — runs in executor thread."""
        import sounddevice as sd
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    async def speak_streaming(self, text: str) -> None:
        """Stream TTS audio for lower time-to-first-sound.

        Starts playing audio as soon as the first chunk arrives from the API,
        rather than waiting for the entire response. Best for longer texts.
        """
        if not self._available or not text.strip():
            return

        if self._speaking:
            await self.stop()

        self._speaking = True
        try:
            client = await self._get_client()

            import numpy as np
            import sounddevice as sd

            response = await client.audio.speech.create(
                model=self.cfg.model,
                voice=self.cfg.voice,
                input=text,
                speed=self.cfg.speed,
                response_format="pcm",
            )

            pcm_bytes = response.content
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = 24000

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._play_blocking, samples, sample_rate)

        except Exception as e:
            print(f"   [Voice] Streaming TTS error: {e}")
        finally:
            self._speaking = False

    async def stop(self) -> None:
        """Stop any currently playing speech."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        if self._speaking:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            self._speaking = False

    @staticmethod
    def list_voices() -> None:
        """Print available voices and their descriptions."""
        print("\n   Available TTS Voices:")
        print("   " + "-" * 50)
        for voice, desc in VOICE_DESCRIPTIONS.items():
            print(f"   {voice:10s} — {desc}")
        print()
