"""
Direct Address Detector — Determines if the user is talking TO the engine.

Classifies audio transcripts as:
  DIRECT    → User is addressing the engine (wake word or directed speech)
  OVERHEARD → User is talking to someone else, on a phone call, etc.
  SILENCE   → No meaningful speech detected

When DIRECT is detected, extracts the user's actual message after the wake word.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from src.config.llm_adapter import LLMAdapter


@dataclass
class AddressResult:
    mode: Literal["DIRECT", "OVERHEARD", "SILENCE"]
    message: str = ""
    transcript: str = ""
    wake_word_found: bool = False
    confidence: float = 0.0


WAKE_PHRASES = [
    "hey creativity",
    "hey creative",
    "hey, creativity",
    "hey, creative",
    "okay creativity",
    "ok creativity",
    "okay creative",
    "ok creative",
    "yo creativity",
    "yo creative",
    "hi creativity",
    "hi creative",
    "hello creativity",
    "creativity,",
    "creativity!",
    "hey creativi",
    "a creativity",
    "hey, creativi",
]

CLASSIFY_PROMPT = """You are analyzing a short audio transcript to determine if the speaker is talking TO an AI assistant named "Creativity" or just talking to someone else nearby (overheard).

Transcript: "{transcript}"

The user's current context: "{context}"

Classify this as one of:
- DIRECT: The speaker is addressing the AI. Signs: questions directed at "you", requests for opinions, statements that expect a response, phrases like "what do you think", "can you", "tell me about", etc.
- OVERHEARD: The speaker is talking to someone else, on a phone call, thinking out loud without expecting a response, or the content is clearly part of another conversation.

Most speech that doesn't contain the wake word "Hey Creativity" is OVERHEARD.
Only classify as DIRECT if you are fairly confident the speaker wants the AI to respond.

Return ONLY one word: DIRECT or OVERHEARD"""


class AddressDetector:
    CONVERSATION_WINDOW_SECONDS = 30.0

    def __init__(self, llm: LLMAdapter | None = None, wake_phrases: list[str] | None = None):
        self.llm = llm
        self.wake_phrases = [p.lower() for p in (wake_phrases or WAKE_PHRASES)]
        self._last_direct_time: float = 0.0

    @property
    def in_conversation(self) -> bool:
        """True if the user recently spoke directly to the engine."""
        if self._last_direct_time == 0.0:
            return False
        return (time.time() - self._last_direct_time) < self.CONVERSATION_WINDOW_SECONDS

    def mark_conversation_active(self) -> None:
        """Called after a direct address to keep the conversation window open."""
        self._last_direct_time = time.time()

    def end_conversation(self) -> None:
        """Manually close the conversation window."""
        self._last_direct_time = 0.0

    def detect(self, transcript: str, context: str = "") -> AddressResult:
        """
        Fast detection using wake word matching + conversation window.
        If the user recently talked to the engine (within 30s), all speech
        is treated as DIRECT — no need to repeat the wake word.
        """
        if not transcript or not transcript.strip():
            return AddressResult(mode="SILENCE", transcript=transcript)

        clean = transcript.strip()
        found, message = self._check_wake_word(clean)

        if found:
            self.mark_conversation_active()
            return AddressResult(
                mode="DIRECT",
                message=message,
                transcript=clean,
                wake_word_found=True,
                confidence=1.0,
            )

        if self.in_conversation:
            self.mark_conversation_active()
            return AddressResult(
                mode="DIRECT",
                message=clean,
                transcript=clean,
                wake_word_found=False,
                confidence=0.85,
            )

        return AddressResult(
            mode="OVERHEARD",
            message="",
            transcript=clean,
            wake_word_found=False,
            confidence=0.7,
        )

    async def detect_with_llm(self, transcript: str, context: str = "") -> AddressResult:
        """
        Two-stage detection: wake word first (fast), then LLM classification
        for ambiguous cases where there's no wake word but the speech might
        still be directed at the engine.
        """
        fast_result = self.detect(transcript, context)

        if fast_result.mode == "DIRECT" or fast_result.mode == "SILENCE":
            return fast_result

        if self.llm is None:
            return fast_result

        try:
            prompt = CLASSIFY_PROMPT.format(transcript=transcript, context=context)
            resp = await self.llm.generate(prompt, temperature=0.1)
            classification = resp.text.strip().upper()

            if "DIRECT" in classification:
                return AddressResult(
                    mode="DIRECT",
                    message=transcript.strip(),
                    transcript=transcript,
                    wake_word_found=False,
                    confidence=0.6,
                )
        except Exception as e:
            print(f"   [Address] LLM classification failed: {e}")

        return fast_result

    def _check_wake_word(self, transcript: str) -> tuple[bool, str]:
        """
        Check if any wake phrase appears in the transcript.
        Returns (found, remaining_message_after_wake_word).
        """
        lower = transcript.lower().strip()

        for phrase in self.wake_phrases:
            idx = lower.find(phrase)
            if idx != -1:
                after = transcript[idx + len(phrase):].strip().lstrip(",.!? ")
                return True, after

        return False, ""
