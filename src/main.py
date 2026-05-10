"""
Curiosity Engine — Proof of Concept Runner

Three modes:
    python -m src.main                          # interactive (manual fire)
    python -m src.main "topic"                  # single-fire with seed topic
    python -m src.main --live                   # LIVE companion mode (continuous heartbeat)
    python -m src.main --live "working on code" # live mode with initial context
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
from src.search.web_search import WebSearcher
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
        self.searcher = WebSearcher(self.llm)
        self.past_topics: list[str] = []
        self.current_context: str = ""
        self._thinking = False

    async def run_creative_cycle(self, seed_topic: str, verbose: bool = True) -> str | None:
        """
        Run one full creative cycle: tree → score → search → bridge → interjection.
        Returns the interjection text, or None if nothing interesting enough.
        """
        self._thinking = True
        t0 = time.time()

        try:
            ctx = ContextSnapshot(seed_topic=seed_topic, heartbeat_id=f"hb-{self.heartbeat.beat_count:04d}")

            if verbose:
                print(f"\n   🌳 Generating association tree from: \"{seed_topic}\"")

            chains = await self.tree_gen.generate_tree(seed_topic)

            if not chains:
                if verbose:
                    print("   ❌ No chains generated.")
                return None

            if verbose:
                print(f"   📊 Scoring top {min(5, len(chains))} of {len(chains)} chains...")

            ranked = await self.scorer.rank_chains(chains, ctx, self.past_topics)
            best_chain, best_score = ranked[0]

            if verbose:
                print(f"   🏆 Best: {best_chain.endpoint_topic} (score: {best_score.total:.3f})")

            if best_score.total < self.cfg.scoring.fire_threshold:
                if verbose:
                    print(f"   🤫 Score {best_score.total:.3f} below threshold {self.cfg.scoring.fire_threshold}")
                return None

            if verbose:
                print(f"   🔍 Searching web for facts...")

            search_result = None
            try:
                search_result = await self.searcher.search_for_chain(
                    endpoint_topic=best_chain.endpoint_topic,
                    chain_summary=best_chain.summary(),
                    context=seed_topic,
                )
                if verbose and search_result and search_result.facts:
                    print(f"   ✅ Found {len(search_result.facts)} facts")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Search failed: {e}")

            interjection = await self.bridge.build_interjection(
                best_chain,
                best_score,
                ctx,
                search_facts=search_result.facts if search_result else None,
                search_sources=search_result.source_urls if search_result else None,
            )
            self.past_topics.append(best_chain.endpoint_topic)

            elapsed = time.time() - t0
            if verbose:
                print(f"   ⏱️  Cycle took {elapsed:.1f}s")

            return interjection.interjection_text

        finally:
            self._thinking = False

    # ── LIVE COMPANION MODE ──────────────────────────────────────────

    async def run_live(self, initial_context: str = "") -> None:
        """
        Live companion mode — the engine runs continuously in the background.
        The heartbeat fires on its own timer. The user can update context,
        talk to the engine, or tell it to back off — all while it thinks.
        """
        self.current_context = initial_context

        print("=" * 70)
        print("🧠 CURIOSITY ENGINE — Live Companion Mode")
        print("=" * 70)
        print(f"\n   Provider: {self.cfg.llm.provider} | Model: {self.cfg.llm.model}")
        search_status = "✅ Tavily" if self.searcher.is_available else "⚠️  LLM fallback"
        print(f"   Search: {search_status}")
        print(f"   Heartbeat: {self.cfg.heartbeat.min_minutes}–{self.cfg.heartbeat.max_minutes} min")
        if self.current_context:
            print(f"   Context: \"{self.current_context}\"")

        print(f"\n{'─' * 70}")
        print("   Commands while running:")
        print("     Just type anything  → Update your context (what you're working on)")
        print("     'not now'           → Skip next 2 heartbeats")
        print("     'status'            → Show engine status")
        print("     'fire'              → Force a heartbeat right now")
        print("     'quit'              → Shut down")
        print(f"{'─' * 70}")

        if not self.current_context:
            print("\n   💡 Tell me what you're up to! (or just press Enter to start)")
            try:
                initial = await asyncio.get_event_loop().run_in_executor(None, input, "   📝 Context: ")
                if initial.strip():
                    self.current_context = initial.strip()
                    print(f"   ✅ Got it — I'll hang out while you work on: \"{self.current_context}\"")
                else:
                    self.current_context = "general exploration"
                    print(f"   ✅ No worries — I'll just explore whatever catches my interest!")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Shutting down.")
                return

        heartbeat_task = asyncio.create_task(self.heartbeat.start(self._on_heartbeat))
        input_task = asyncio.create_task(self._input_loop())

        try:
            done, pending = await asyncio.wait(
                [heartbeat_task, input_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            pass

        print("\n👋 Curiosity Engine shutting down. Stay curious!")

    async def _on_heartbeat(self, ctx: ContextSnapshot) -> None:
        """Called by the heartbeat timer — runs a full creative cycle."""
        ctx.seed_topic = self.current_context
        print(f"   🎯 Context: \"{self.current_context}\"")

        result = await self.run_creative_cycle(self.current_context, verbose=True)

        if result:
            print(f"\n{'═' * 70}")
            print(f"💬 CURIOSITY ENGINE SAYS:\n")
            print(f"   \"{result}\"")
            print(f"\n{'═' * 70}")
        else:
            print(f"\n   🤫 Nothing interesting enough this time. I'll keep thinking...")

    async def _input_loop(self) -> None:
        """Listen for user input while the heartbeat runs in the background."""
        loop = asyncio.get_event_loop()

        while self.heartbeat.is_running:
            try:
                user_input = await loop.run_in_executor(None, input, "")
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                self.heartbeat.stop()
                return

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "q"):
                self.heartbeat.stop()
                return

            elif cmd in ("not now", "notnow", "shh", "quiet", "hush"):
                self.heartbeat.backoff(2)
                print("   🤫 Okay, I'll be quiet for a bit!")

            elif cmd == "status":
                remaining = self.heartbeat.time_until_next
                mins = remaining // 60
                secs = remaining % 60
                thinking = "🧠 Thinking..." if self._thinking else "😌 Idle"
                print(f"\n   📊 Status:")
                print(f"      State: {thinking}")
                print(f"      Context: \"{self.current_context}\"")
                print(f"      Heartbeats fired: {self.heartbeat.beat_count}")
                print(f"      Next heartbeat: ~{mins}m {secs}s")
                print(f"      Past topics: {len(self.past_topics)}")

            elif cmd == "fire":
                if self._thinking:
                    print("   ⏳ Already thinking — hang on!")
                else:
                    print("   ⚡ Forcing heartbeat NOW!")
                    self.heartbeat._remaining_seconds = 0

            else:
                self.current_context = user_input
                print(f"   ✅ Context updated: \"{self.current_context}\"")

    # ── SINGLE-FIRE & INTERACTIVE MODES (unchanged) ──────────────────

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
            search_result = await self._search_for_facts(best_chain, ctx)

            print(f"\n🌉 Building interjection from best chain + search results...\n")
            interjection = await self.bridge.build_interjection(
                best_chain,
                best_score,
                ctx,
                search_facts=search_result.facts if search_result else None,
                search_sources=search_result.source_urls if search_result else None,
            )
            self.past_topics.append(best_chain.endpoint_topic)

            elapsed = time.time() - t0
            print(f"{'═' * 70}")
            print(f"💬 CURIOSITY ENGINE SAYS:\n")
            print(f"   \"{interjection.interjection_text}\"")
            print(f"\n{'═' * 70}")
            print(f"\n⏱️  Total pipeline time: {elapsed:.1f}s")
            print(f"📍 Internal chain: {best_chain.summary()}")
            print(f"📊 Interest score: {best_score.total:.3f}")
            if search_result and search_result.facts:
                print(f"🔍 Grounded in {len(search_result.facts)} web facts from {len(search_result.source_urls)} sources")
            if search_result and search_result.source_urls:
                for url in search_result.source_urls[:3]:
                    print(f"   📎 {url}")
        else:
            elapsed = time.time() - t0
            print(f"{'─' * 70}")
            print(f"🤫 Nothing interesting enough to share this cycle.")
            print(f"   Best score: {best_score.total:.3f} (threshold: {self.cfg.scoring.fire_threshold})")
            print(f"   Best chain: {best_chain.summary()}")
            print(f"\n⏱️  Total pipeline time: {elapsed:.1f}s")

    async def _search_for_facts(self, chain, ctx):
        """Run web search on the winning chain's endpoint."""
        print(f"🔍 Searching the web for facts about: \"{chain.endpoint_topic}\"")
        try:
            search_result = await self.searcher.search_for_chain(
                endpoint_topic=chain.endpoint_topic,
                chain_summary=chain.summary(),
                context=ctx.seed_topic,
            )
            if search_result.facts:
                print(f"   ✅ Found {len(search_result.facts)} facts:")
                for fact in search_result.facts:
                    print(f"      • {fact[:100]}{'...' if len(fact) > 100 else ''}")
            else:
                print(f"   ℹ️  No specific facts extracted from search results")
            return search_result
        except Exception as e:
            print(f"   ⚠️  Search failed: {e}")
            return None

    async def run_interactive(self) -> None:
        """Interactive mode — enter seed topics and watch the engine think."""
        print("=" * 70)
        print("🧠 CURIOSITY ENGINE — Interactive Mode")
        print("=" * 70)
        print(f"\nProvider: {self.cfg.llm.provider} | Model: {self.cfg.llm.model}")
        search_status = "✅ Tavily" if self.searcher.is_available else "⚠️  No API key (using LLM fallback)"
        print(f"Search: {search_status}")
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

    args = sys.argv[1:]

    if "--live" in args:
        args.remove("--live")
        initial_context = " ".join(args) if args else ""
        await engine.run_live(initial_context)
    elif args:
        seed = " ".join(args)
        await engine.run_single(seed)
    else:
        await engine.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
