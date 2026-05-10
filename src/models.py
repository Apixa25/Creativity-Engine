"""
Data models for the Creativity Engine.

These are the core data structures that flow through the pipeline:
Heartbeat → Context → Association Tree → Scoring → Interjection
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np


DOMAINS = [
    "Science & Technology",
    "Psychology & Behavior",
    "Art & Design",
    "History & Culture",
    "Nature & Biology",
    "Philosophy & Ethics",
    "Economics & Business",
    "Mathematics & Logic",
    "Language & Literature",
    "Sports & Games",
    "Music & Sound",
    "Food & Cooking",
    "Architecture & Space",
    "Politics & Society",
    "Health & Medicine",
]


@dataclass
class AssociationNode:
    topic: str
    domain: str
    connection_reason: str
    depth: int
    embedding: np.ndarray | None = None
    parent: AssociationNode | None = field(default=None, repr=False)
    children: list[AssociationNode] = field(default_factory=list, repr=False)
    preliminary_score: float | None = None

    def chain_to_root(self) -> list[AssociationNode]:
        """Walk up to root, return path from root to self."""
        path = []
        node: AssociationNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def chain_topics(self) -> list[str]:
        return [n.topic for n in self.chain_to_root()]

    def chain_domains(self) -> list[str]:
        return [n.domain for n in self.chain_to_root()]

    def domain_crossings(self) -> int:
        domains = self.chain_domains()
        return sum(1 for i in range(1, len(domains)) if domains[i] != domains[i - 1])


@dataclass
class AssociationChain:
    seed_topic: str
    nodes: list[AssociationNode]
    total_semantic_distance: float = 0.0
    domain_crossings: int = 0
    interest_score: float | None = None
    bridge_sentence: str | None = None

    @property
    def endpoint(self) -> AssociationNode:
        return self.nodes[-1]

    @property
    def endpoint_topic(self) -> str:
        return self.endpoint.topic

    def summary(self) -> str:
        return " → ".join(n.topic for n in self.nodes)


@dataclass
class ScoringBreakdown:
    semantic_distance: float
    domain_crossings: float
    surprise: float
    bridgeability: float
    novelty: float
    total: float

    def __str__(self) -> str:
        return (
            f"  semantic_distance : {self.semantic_distance:.3f} (×0.30 = {self.semantic_distance * 0.30:.3f})\n"
            f"  domain_crossings  : {self.domain_crossings:.3f} (×0.25 = {self.domain_crossings * 0.25:.3f})\n"
            f"  surprise          : {self.surprise:.3f} (×0.20 = {self.surprise * 0.20:.3f})\n"
            f"  bridgeability     : {self.bridgeability:.3f} (×0.15 = {self.bridgeability * 0.15:.3f})\n"
            f"  novelty           : {self.novelty:.3f} (×0.10 = {self.novelty * 0.10:.3f})\n"
            f"  TOTAL             : {self.total:.3f}"
        )


@dataclass
class ChannelInput:
    """One channel's contribution to the context snapshot."""
    channel: str                    # "vision", "audio", "text"
    raw_content: str = ""           # description, transcript, or user text
    novelty: float = 1.0           # 0.0 (same as before) to 1.0 (completely new)
    base_weight: float = 0.0
    effective_weight: float = 0.0   # base_weight × novelty
    available: bool = True


@dataclass
class ContextSnapshot:
    """
    Multimodal context snapshot — what the engine perceives at this moment.
    Combines vision (webcam), audio (mic), and text (user input) channels.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    heartbeat_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # Assembled seed topic (built from all channels)
    seed_topic: str = ""

    # Individual channel data
    vision: ChannelInput | None = None
    audio: ChannelInput | None = None
    text: ChannelInput | None = None

    # Composite scores
    dominant_channel: str = "text"
    overall_novelty: float = 1.0

    # Raw image bytes (for passing to vision LLM)
    image_bytes: bytes | None = field(default=None, repr=False)


@dataclass
class Interjection:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    heartbeat_id: str = ""
    chain: AssociationChain | None = None
    scoring: ScoringBreakdown | None = None
    interjection_text: str = ""
    context: ContextSnapshot | None = None
    search_facts: list[str] = field(default_factory=list)
    search_sources: list[str] = field(default_factory=list)
