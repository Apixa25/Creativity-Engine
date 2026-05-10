"""
Context Assembler — Combines multimodal channels into a unified snapshot.

Takes vision, audio, and text inputs, weights them by novelty,
and assembles a single ContextSnapshot with a composite seed topic
for the association tree to work from.
"""

from __future__ import annotations

from datetime import datetime

from src.config.llm_adapter import LLMAdapter
from src.input_pipeline.vision import VisionChannel
from src.input_pipeline.audio import AudioChannel
from src.models import ContextSnapshot, ChannelInput


SEED_EXTRACTION_PROMPT = """You are observing a user's environment through multiple senses. Based on what you see, hear, and know about their activity, extract 1-2 interesting seed topics for creative exploration.

{channels}

Rules:
- Combine information from ALL available channels into a cohesive understanding
- Weight channels by their novelty score — higher novelty = more important right now
- Extract specific, interesting topics — not vague descriptions
- If vision shows something new and interesting, prioritize that
- If audio captured something the user said, incorporate that
- The text context is what the user told you they're doing

Return ONLY 1-2 seed topics as a short phrase (under 15 words). Examples:
- "building contractor scheduling software with drag-and-drop calendar"
- "person at desk with a new book about space exploration"
- "cooking pasta while talking about weekend plans"

Return just the phrase, nothing else."""


class ContextAssembler:
    def __init__(
        self,
        llm: LLMAdapter,
        vision: VisionChannel | None = None,
        audio: AudioChannel | None = None,
        text_base_weight: float = 0.40,
        skip_threshold: float = 0.15,
    ):
        self.llm = llm
        self.vision = vision
        self.audio = audio
        self.text_base_weight = text_base_weight
        self.skip_threshold = skip_threshold
        self._last_user_text = ""

    async def assemble(
        self,
        user_text: str = "",
        heartbeat_id: str = "",
    ) -> ContextSnapshot:
        """
        Capture all channels, weight by novelty, and assemble a ContextSnapshot.
        """
        sensors = []
        if self.vision and self.vision.is_available:
            sensors.append("Camera")
        if self.audio and self.audio.is_available:
            sensors.append(f"Mic ({self.audio.capture_seconds:.0f}s)")

        if sensors:
            sensor_list = " + ".join(sensors)
            print(f"\n   +{'=' * 48}+")
            print(f"   |  PERCEPTION WINDOW OPEN -- {sensor_list:<19}|")
            print(f"   |  Sensors are LIVE -- do/say something cool!  |")
            print(f"   +{'=' * 48}+")
        else:
            print("   [Assembler] Assembling context (text only)...")

        # ── Vision channel ──
        vision_input = None
        image_bytes = None
        if self.vision and self.vision.is_available:
            vision_input, image_bytes = await self.vision.process(llm=self.llm)

        # ── Audio channel ──
        audio_input = None
        if self.audio and self.audio.is_available:
            audio_input = await self.audio.process()

        # ── Text channel ──
        text_novelty = self._compute_text_novelty(user_text)
        text_weight = self.text_base_weight * text_novelty
        text_input = ChannelInput(
            channel="text",
            raw_content=user_text,
            novelty=text_novelty,
            base_weight=self.text_base_weight,
            effective_weight=text_weight,
            available=bool(user_text),
        )
        novelty_label = "🔴 HIGH" if text_novelty > 0.6 else "🟡 MED" if text_novelty > 0.3 else "⚪ LOW"
        print(f"   [Text] [{novelty_label} novelty={text_novelty:.2f}]: \"{user_text[:60]}\"")

        # ── Find dominant channel ──
        channels = []
        if vision_input and vision_input.available:
            channels.append(vision_input)
        if audio_input and audio_input.available:
            channels.append(audio_input)
        if text_input.available:
            channels.append(text_input)

        if not channels:
            if sensors:
                print(f"   +{'=' * 48}+")
                print(f"   |  PERCEPTION WINDOW CLOSED                    |")
                print(f"   +{'=' * 48}+")
            return ContextSnapshot(
                heartbeat_id=heartbeat_id,
                seed_topic=user_text or "general exploration",
                text=text_input,
                overall_novelty=0.0,
            )

        dominant = max(channels, key=lambda c: c.effective_weight)
        overall_novelty = self._compute_overall_novelty(channels)

        # ── Build seed topic from multimodal inputs ──
        has_rich_input = (
            (vision_input and vision_input.raw_content and vision_input.novelty > 0.3)
            or (audio_input and audio_input.raw_content and audio_input.novelty > 0.3
                and audio_input.raw_content not in ("[silence]", "[capture failed]", "[transcription empty]"))
        )

        if has_rich_input:
            seed_topic = await self._extract_seed_topic(vision_input, audio_input, text_input)
        else:
            seed_topic = user_text or "general exploration"

        print(f"   >> Dominant channel: {dominant.channel} | Overall novelty: {overall_novelty:.2f}")
        print(f"   >> Seed topic: \"{seed_topic}\"")
        if sensors:
            print(f"   +{'=' * 48}+")
            print(f"   |  PERCEPTION WINDOW CLOSED -- processing...   |")
            print(f"   +{'=' * 48}+")

        return ContextSnapshot(
            heartbeat_id=heartbeat_id,
            seed_topic=seed_topic,
            vision=vision_input,
            audio=audio_input,
            text=text_input,
            dominant_channel=dominant.channel,
            overall_novelty=overall_novelty,
            image_bytes=image_bytes,
        )

    async def _extract_seed_topic(
        self,
        vision: ChannelInput | None,
        audio: ChannelInput | None,
        text: ChannelInput,
    ) -> str:
        """Use the LLM to synthesize a seed topic from all available channels."""
        parts = []
        if vision and vision.raw_content and vision.available:
            parts.append(f"VISION (novelty={vision.novelty:.2f}): {vision.raw_content}")
        if audio and audio.raw_content and audio.available and audio.raw_content not in ("[silence]",):
            parts.append(f"AUDIO (novelty={audio.novelty:.2f}): {audio.raw_content}")
        if text.raw_content and text.available:
            parts.append(f"TEXT CONTEXT (novelty={text.novelty:.2f}): {text.raw_content}")

        if not parts:
            return text.raw_content or "general exploration"

        channels_text = "\n".join(parts)
        prompt = SEED_EXTRACTION_PROMPT.format(channels=channels_text)

        try:
            resp = await self.llm.generate(prompt, temperature=0.5)
            return resp.text.strip().strip('"')
        except Exception as e:
            print(f"   [!] Seed extraction failed: {e}")
            return text.raw_content or "general exploration"

    def _compute_text_novelty(self, current: str) -> float:
        """Compare current text to last text. Different content = high novelty."""
        if not current:
            return 0.0
        if not self._last_user_text:
            self._last_user_text = current
            return 0.5

        current_words = set(current.lower().split())
        past_words = set(self._last_user_text.lower().split())
        union = len(current_words | past_words)
        overlap = len(current_words & past_words)
        similarity = overlap / union if union > 0 else 0.0

        self._last_user_text = current
        return max(0.0, min(1.0, 1.0 - similarity))

    def _compute_overall_novelty(self, channels: list[ChannelInput]) -> float:
        """Weighted average of all channel novelties."""
        total_weight = sum(ch.effective_weight for ch in channels)
        if total_weight == 0:
            return 0.0
        weighted_novelty = sum(ch.novelty * ch.effective_weight for ch in channels)
        return weighted_novelty / total_weight
