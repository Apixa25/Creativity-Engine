"""
Curiosity Engine — Proof of Concept Runner

Ties together: Heartbeat → Association Tree → Interest Scorer → Bridge Builder

Usage:
    python -m src.main                          # interactive mode
    python -m src.main "building software"      # single-fire with seed topic
"""

from __future__ import annotations

import asyncio
import sys
import time

from src.config.settings import load_config, EngineConfig
from src.config.llm_adapter import create_llm_adapter, LLMAdapter
from src.heartbeat.clock import Heartbeat
from src.association_engine.tree_generator import AssociationTreeGenerator
from src.scoring.interest_scorer import InterestScorer
from src.bridge_builder.builder import BridgeBuilder
from src.models import ContextSnapshot


class CuriosityEngine:
    """Orchestrates the full creative pipeline."""

    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or load_config()
        self.llm: LLMAdapter = create_llm_adapter(
            provider=self.cfg.llm.provider,
            model=self.cfg.llm.model,
            api_key_env=self.cfg.llm.api_key_env,
        )
        self.heartbeat = Heartbeat(
            min_minutes=self.cfg.heartbeat.min_minutes,
            max_minutes=self.cfg.heartbeat.max_minutes,
        )
        self.tree_gen = AssociationTreeGenerator(self.llm, self.cfg.association_tree)
        self.scorer = InterestScorer(self.llm, self.cfg.scoring)
        self.bridge = BridgeBuilder(self.llm)
        self.past_topics: list[str] = []

    async def run_single(self, seed_topic: str) -> None:
        """Fire a single heartbeat cycle — for testing and demos."""
        print("=" * 70)
        print("🧠 CURIOSITY ENGINE — Proof of Concept")
        print("=" * 70)

        ctx = await self.heartbeat.fire_once(seed_topic)

        t0 = time.time()

        print(f"\n🌳 Generating association tree from: \"{seed_topic}\"")
        print(f"   Config: branching={self.cfg.association_tree.branching_factor}, "
              f"depth={self.cfg.association_tree.min_depth}-{self.cfg.association_tree.max_depth}, "
              f"pruning=keep top {self.cfg.association_tree.keep_per_level}/level")

        chains = await self.tree_gen.generate_tree(seed_topic)

        if not chains:
            print("\n❌ No association chains were generated. Try a different seed topic.")
            return

        print(f"\n📊 Scoring top candidates from {len(chains)} chains...")
        ranked = await self.scorer.rank_chains(chains, ctx, self.past_topics)

        print(f"\n{'─' * 70}")
        print("📋 TOP CANDIDATES:\n")
        for i, (chain, score) in enumerate(ranked, 1):
            status = "✅ FIRE" if score.total >= self.cfg.scoring.fire_threshold else (
                "🧪 INCUBATE" if score.total >= self.cfg.scoring.incubation_threshold else "❌ DISCARD"
            )
            print(f"  #{i} [{status}] Score: {score.total:.3f}")
            print(f"     Chain: {chain.summary()}")
            print(f"     Breakdown: dist={score.semantic_distance:.2f} cross={score.domain_crossings:.2f} "
                  f"surprise={score.surprise:.2f} bridge={score.bridgeability:.2f} novel={score.novelty:.2f}")
            print()

        best_chain, best_score = ranked[0]

        if best_score.total >= self.cfg.scoring.fire_threshold:
            print(f"{'─' * 70}")
            print("🌉 Building interjection from best chain...\n")
            interjection = await self.bridge.build_interjection(best_chain, best_score, ctx)
            self.past_topics.append(best_chain.endpoint_topic)

            elapsed = time.time() - t0

            print(f"{'═' * 70}")
            print(f"💬 CURIOSITY ENGINE SAYS:\n")
            print(f"   \"{interjection.interjection_text}\"")
            print(f"\n{'═' * 70}")
            print(f"\n⏱️  Total pipeline time: {elapsed:.1f}s")
            print(f"📍 Internal chain: {best_chain.summary()}")
            print(f"📊 Interest score: {best_score.total:.3f}")
        else:
            elapsed = time.time() - t0
            print(f"{'─' * 70}")
            print(f"🤫 Nothing interesting enough to share this cycle.")
            print(f"   Best score: {best_score.total:.3f} (threshold: {self.cfg.scoring.fire_threshold})")
            print(f"   Best chain: {best_chain.summary()}")
            print(f"\n⏱️  Total pipeline time: {elapsed:.1f}s")

    async def run_interactive(self) -> None:
        """Interactive mode — enter seed topics and watch the engine think."""
        print("=" * 70)
        print("🧠 CURIOSITY ENGINE — Interactive Mode")
        print("=" * 70)
        print(f"\nProvider: {self.cfg.llm.provider} | Model: {self.cfg.llm.model}")
        print("Type a seed topic and press Enter to fire a heartbeat cycle.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                topic = input("🎯 Seed topic: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Curiosity Engine shutting down. Stay curious!")
                break

            if not topic:
                continue
            if topic.lower() in ("quit", "exit", "q"):
                print("\n👋 Curiosity Engine shutting down. Stay curious!")
                break

            await self.run_single(topic)
            print()


async def main():
    config = load_config()
    engine = CuriosityEngine(config)

    if len(sys.argv) > 1:
        seed = " ".join(sys.argv[1:])
        await engine.run_single(seed)
    else:
        await engine.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
