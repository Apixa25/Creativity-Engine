"""
Heartbeat — The Curiosity Clock.

A random timer that fires at unpredictable intervals (1–10 minutes),
creating the hidden causation that produces the appearance of spontaneous thought.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import Callable, Awaitable

from src.models import ContextSnapshot


class Heartbeat:
    def __init__(self, min_minutes: int = 1, max_minutes: int = 10):
        self.min_minutes = min_minutes
        self.max_minutes = max_minutes
        self._running = False
        self._task: asyncio.Task | None = None
        self.beat_count = 0

    def next_interval_seconds(self) -> int:
        """Pick a random interval in whole minutes, return as seconds."""
        minutes = random.randint(self.min_minutes, self.max_minutes)
        return minutes * 60

    async def start(self, on_beat: Callable[[ContextSnapshot], Awaitable[None]]) -> None:
        """Start the heartbeat loop. Calls on_beat every time the timer fires."""
        self._running = True
        while self._running:
            interval = self.next_interval_seconds()
            minutes = interval // 60
            print(f"\n💓 Heartbeat scheduled — next fire in {minutes} minute(s)...")
            await asyncio.sleep(interval)

            if not self._running:
                break

            self.beat_count += 1
            ctx = ContextSnapshot(
                timestamp=datetime.now(),
                heartbeat_id=f"hb-{self.beat_count:04d}",
            )
            print(f"\n💓 HEARTBEAT #{self.beat_count} fired at {ctx.timestamp.strftime('%H:%M:%S')}")
            await on_beat(ctx)

    def stop(self) -> None:
        self._running = False

    async def fire_once(self, seed_topic: str) -> ContextSnapshot:
        """Fire a single heartbeat immediately with a given seed topic (for testing)."""
        self.beat_count += 1
        ctx = ContextSnapshot(
            timestamp=datetime.now(),
            heartbeat_id=f"hb-{self.beat_count:04d}",
            seed_topic=seed_topic,
        )
        print(f"\n💓 HEARTBEAT #{self.beat_count} fired at {ctx.timestamp.strftime('%H:%M:%S')}")
        print(f"   Seed topic: \"{seed_topic}\"")
        return ctx
