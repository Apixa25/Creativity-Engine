"""
Heartbeat — The Creativity Clock.

A random timer that fires at unpredictable intervals (1–10 minutes),
creating the hidden causation that produces the appearance of spontaneous thought.

Supports both single-fire (testing) and continuous loop (live companion mode).
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
        self.beat_count = 0
        self._remaining_seconds = 0
        self._backoff_beats = 0

    def next_interval_seconds(self) -> int:
        """Pick a random interval in whole minutes, return as seconds."""
        minutes = random.randint(self.min_minutes, self.max_minutes)
        return minutes * 60

    async def start(self, on_beat: Callable[[ContextSnapshot], Awaitable[None]]) -> None:
        """Start the continuous heartbeat loop. Calls on_beat every time the timer fires."""
        self._running = True
        print(f"   💓 Heartbeat range: {self.min_minutes}–{self.max_minutes} minutes")

        while self._running:
            interval = self.next_interval_seconds()
            self._remaining_seconds = interval
            minutes = interval // 60
            print(f"\n   💓 Next heartbeat in {minutes} minute(s)...")

            while self._remaining_seconds > 0 and self._running:
                sleep_chunk = min(1, self._remaining_seconds)
                await asyncio.sleep(sleep_chunk)
                self._remaining_seconds -= sleep_chunk

            if not self._running:
                break

            if self._backoff_beats > 0:
                self._backoff_beats -= 1
                print(f"   💤 Heartbeat suppressed (backing off, {self._backoff_beats} skips remaining)")
                continue

            self.beat_count += 1
            ctx = ContextSnapshot(
                timestamp=datetime.now(),
                heartbeat_id=f"hb-{self.beat_count:04d}",
            )
            print(f"\n{'═' * 70}")
            print(f"💓 HEARTBEAT #{self.beat_count} fired at {ctx.timestamp.strftime('%H:%M:%S')}")
            print(f"{'═' * 70}")
            await on_beat(ctx)

    def stop(self) -> None:
        self._running = False
        self._remaining_seconds = 0

    def backoff(self, beats: int = 2) -> None:
        """Skip the next N heartbeats — user said 'not now'."""
        self._backoff_beats = beats
        print(f"   💤 Backing off for {beats} heartbeat(s)")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def time_until_next(self) -> int:
        return max(0, self._remaining_seconds)

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
