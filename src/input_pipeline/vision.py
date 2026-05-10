"""
Vision Channel — Webcam capture with novelty detection.

Captures a single frame when the heartbeat fires (not continuous streaming).
Uses perceptual hashing to detect novelty — if the scene hasn't changed,
the vision channel fades to near-zero weight. New scenes spike to high weight.

For image description, sends the frame to an LLM with vision capability.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.models import ChannelInput


@dataclass
class ImageCapture:
    image_bytes: bytes
    phash: str
    description: str = ""
    novelty: float = 1.0


class VisionChannel:
    def __init__(self, history_window: int = 10, base_weight: float = 0.35, min_novelty_for_description: float = 0.3):
        self.history_window = history_window
        self.base_weight = base_weight
        self.min_novelty_for_description = min_novelty_for_description
        self._hash_history: deque[str] = deque(maxlen=history_window)
        self._camera = None
        self._available = False
        self._initialized = False

    def initialize(self) -> bool:
        """Try to open the webcam. Returns True if successful."""
        if self._initialized:
            return self._available
        self._initialized = True
        try:
            import cv2
            self._camera = cv2.VideoCapture(0)
            if self._camera.isOpened():
                self._available = True
                print("   [Vision] Webcam: connected")
            else:
                self._available = False
                self._camera = None
                print("   [Vision] Webcam: not available")
        except ImportError:
            print("   [Vision] Webcam: opencv-python not installed")
            self._available = False
        except Exception as e:
            print(f"   [Vision] Webcam error: {e}")
            self._available = False
        return self._available

    @property
    def is_available(self) -> bool:
        return self._available

    def capture_frame(self) -> bytes | None:
        """Capture a single frame from the webcam. Returns JPEG bytes or None."""
        if not self._available or self._camera is None:
            return None
        try:
            import cv2
            ret, frame = self._camera.read()
            if not ret or frame is None:
                return None
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()
        except Exception as e:
            print(f"   [Vision]  Capture error: {e}")
            return None

    def compute_novelty(self, image_bytes: bytes) -> float:
        """
        Compare this frame to recent history using perceptual hashing.
        Returns 0.0 (identical scene) to 1.0 (completely new).
        """
        current_hash = self._perceptual_hash(image_bytes)

        if not self._hash_history:
            self._hash_history.append(current_hash)
            return 1.0

        similarities = [
            self._hash_similarity(current_hash, past_hash)
            for past_hash in self._hash_history
        ]
        max_similarity = max(similarities)

        self._hash_history.append(current_hash)

        novelty = 1.0 - max_similarity
        return max(0.0, min(1.0, novelty))

    async def process(self, llm=None) -> tuple[ChannelInput, bytes | None]:
        """
        Full vision pipeline: capture → novelty → describe (if novel enough).
        Returns (ChannelInput, raw_image_bytes).
        """
        if not self._available:
            return ChannelInput(
                channel="vision",
                raw_content="",
                novelty=0.0,
                base_weight=self.base_weight,
                effective_weight=0.0,
                available=False,
            ), None

        image_bytes = self.capture_frame()
        if image_bytes is None:
            return ChannelInput(
                channel="vision",
                raw_content="[camera read failed]",
                novelty=0.0,
                base_weight=self.base_weight,
                effective_weight=0.0,
                available=False,
            ), None

        novelty = self.compute_novelty(image_bytes)
        effective_weight = self.base_weight * novelty

        description = ""
        if novelty >= self.min_novelty_for_description and llm is not None:
            description = await self._describe_image(image_bytes, llm)

        novelty_label = "🔴 HIGH" if novelty > 0.6 else "🟡 MED" if novelty > 0.3 else "⚪ LOW"
        if description:
            print(f"   [Vision]  Vision [{novelty_label} novelty={novelty:.2f}]: {description[:80]}")
        else:
            print(f"   [Vision]  Vision [{novelty_label} novelty={novelty:.2f}]: scene unchanged, skipping description")

        return ChannelInput(
            channel="vision",
            raw_content=description,
            novelty=novelty,
            base_weight=self.base_weight,
            effective_weight=effective_weight,
            available=True,
        ), image_bytes

    async def _describe_image(self, image_bytes: bytes, llm) -> str:
        """Send the image to an LLM with vision capability for description."""
        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            from openai import AsyncOpenAI
            import os
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe what you see in this image in 1-2 concise sentences. "
                                "Focus on what's interesting or notable — people, objects, activities, "
                                "screen content, environment. Be specific, not generic."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "low"},
                        },
                    ],
                }],
                max_tokens=150,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"   [Vision]  Image description failed: {e}")
            return ""

    def _perceptual_hash(self, image_bytes: bytes, hash_size: int = 16) -> str:
        """
        Compute a simple perceptual hash by resizing to a tiny grayscale
        image and comparing pixel intensities. Lightweight — no ML needed.
        """
        try:
            import cv2
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return hashlib.md5(image_bytes).hexdigest()
            resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
            mean_val = resized.mean()
            bits = (resized > mean_val).flatten()
            hash_bytes = np.packbits(bits).tobytes()
            return hashlib.md5(hash_bytes).hexdigest()
        except Exception:
            return hashlib.md5(image_bytes).hexdigest()

    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """Compare two hex hashes by character overlap. Simple but effective."""
        if hash1 == hash2:
            return 1.0
        matching = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matching / max(len(hash1), len(hash2))

    def release(self):
        """Release the webcam."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
