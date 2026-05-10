"""
Creativity Engine — Proof of Concept Runner

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
from src.input_pipeline.vision import VisionChannel
from src.input_pipeline.audio import AudioChannel
from src.input_pipeline.assembler import ContextAssembler
from src.input_pipeline.address_detector import AddressDetector
from src.conversation.responder import DirectResponder
from src.models import ContextSnapshot, Interjection


class CreativityEngine:
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
        self._listening = False
        self._multimodal = False
        self.vision: VisionChannel | None = None
        self.audio: AudioChannel | None = None
        self.assembler: ContextAssembler | None = None
        self.detector: AddressDetector = AddressDetector(llm=self.llm)
        self.responder: DirectResponder = DirectResponder(llm=self.llm)
        self._overheard_buffer: list[str] = []

    def enable_multimodal(self) -> None:
        """Initialize vision + audio channels. Call before run_live for full perception."""
        print("\n   Initializing multimodal input pipeline...")
        ip = self.cfg.input_pipeline

        self.vision = VisionChannel(
            history_window=ip.vision.history_window,
            base_weight=ip.vision.base_weight,
            min_novelty_for_description=ip.vision.min_novelty_for_description,
            device_index=ip.vision.device_index,
        )
        if ip.vision.enabled:
            self.vision.initialize()

        self.audio = AudioChannel(
            api_key=getattr(self.llm, 'api_key', ''),
            capture_seconds=ip.audio.capture_seconds,
            base_weight_direct=ip.audio.base_weight_direct,
            base_weight_overheard=ip.audio.base_weight_overheard,
            device_index=ip.audio.device_index,
        )
        if ip.audio.enabled:
            self.audio.initialize()

        self.assembler = ContextAssembler(
            llm=self.llm,
            vision=self.vision if self.vision.is_available else None,
            audio=self.audio if self.audio.is_available else None,
        )
        self._multimodal = self.vision.is_available or self.audio.is_available
        if self._multimodal:
            print("   Multimodal input enabled!")
        else:
            print("   No cameras or mics detected -- using text context only")

    async def run_creative_cycle(self, seed_topic: str, verbose: bool = True) -> Interjection | None:
        """
        Run one full creative cycle: tree → score → search → bridge → interjection.
        Returns the full Interjection object, or None if nothing interesting enough.
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

            return interjection

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
        print("🧠 CREATIVITY ENGINE — Live Companion Mode")
        print("=" * 70)

        import os
        api_key = getattr(self.llm, 'api_key', '') or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("\n   ⚠️  WARNING: No OPENAI_API_KEY found!")
            print("   Set it with:  $env:OPENAI_API_KEY = \"sk-proj-your-key-here\"")
            print("   The engine will NOT work without it.\n")

        print(f"\n   Provider: {self.cfg.llm.provider} | Model: {self.cfg.llm.model}")
        search_status = "✅ Tavily" if self.searcher.is_available else "⚠️  LLM fallback"
        print(f"   Search: {search_status}")
        print(f"   Heartbeat: {self.cfg.heartbeat.min_minutes}–{self.cfg.heartbeat.max_minutes} min")
        if self.current_context:
            print(f"   Context: \"{self.current_context}\"")

        self.enable_multimodal()

        has_mic = self.audio and self.audio.is_available
        if has_mic:
            print(f"   Voice: Say 'Hey Creativity' to talk directly!")
        else:
            print(f"   Voice: No mic -- type to chat")

        print(f"\n{'─' * 70}")
        print("   Commands while running:")
        print("     Just type anything  → Update your context (what you're working on)")
        if has_mic:
            print("     'Hey Creativity'    → Talk to the engine directly (voice)")
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

        tasks = [
            asyncio.create_task(self.heartbeat.start(self._on_heartbeat)),
            asyncio.create_task(self._input_loop()),
        ]

        if self.audio and self.audio.is_available:
            tasks.append(asyncio.create_task(self._listening_loop()))

        try:
            done, pending = await asyncio.wait(
                tasks,
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

        self._listening = False
        print("\n👋 Creativity Engine shutting down. Stay creative!")
        if self.vision:
            self.vision.release()

    async def _on_heartbeat(self, ctx: ContextSnapshot) -> None:
        """Called by the heartbeat timer — runs a full creative cycle."""
        context_with_overheard = self.current_context
        if self._overheard_buffer:
            overheard_text = " | ".join(self._overheard_buffer[-3:])
            context_with_overheard += f" (overheard: {overheard_text})"
            self._overheard_buffer.clear()

        if self._multimodal and self.assembler:
            ctx = await self.assembler.assemble(
                user_text=context_with_overheard,
                heartbeat_id=ctx.heartbeat_id,
            )
            seed = ctx.seed_topic
        else:
            ctx.seed_topic = context_with_overheard
            seed = context_with_overheard
            print(f"   🎯 Context: \"{context_with_overheard}\"")

        interjection = await self.run_creative_cycle(seed, verbose=True)

        if interjection:
            print(f"\n{'═' * 70}")
            print(f"💬 CREATIVITY ENGINE SAYS:\n")
            print(f"   \"{interjection.interjection_text}\"")
            print(f"\n{'═' * 70}")
            self._print_citations(interjection)
            self.responder.add_engine_interjection(interjection.interjection_text)
        else:
            print(f"\n   🤫 Nothing interesting enough this time. I'll keep thinking...")

    # ── BACKGROUND LISTENER (Direct Address Detection) ─────────────

    async def _listening_loop(self) -> None:
        """
        Continuous background listener. Records short audio clips in a loop,
        transcribes them, and checks for direct address (wake word).

        - DIRECT  -> respond immediately via the Direct Response Engine
        - OVERHEARD -> buffer the transcript for the next heartbeat's context
        - SILENCE -> do nothing, loop again

        The synchronous audio capture (sd.rec + sd.wait) runs in an executor
        so it doesn't block the event loop. Transcription runs async normally.
        """
        self._listening = True
        pause_seconds = 1.0
        listen_count = 0

        print(f"\n   [Listener] Background listener ACTIVE -- say 'Hey Creativity' anytime!")

        while self.heartbeat.is_running and self._listening:
            if self._thinking:
                await asyncio.sleep(pause_seconds)
                continue

            if not self.audio or not self.audio.is_available:
                await asyncio.sleep(pause_seconds)
                continue

            try:
                listen_count += 1
                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(
                    None, self._capture_and_detect_speech
                )

                if audio_data is None:
                    if listen_count % 12 == 0:
                        print(f"   [Listener] ... still listening (cycle #{listen_count})")
                    continue

                print(f"   [Listener] Speech detected! Transcribing...")
                transcript = await self.audio.transcribe(audio_data)

                if not transcript:
                    print(f"   [Listener] Transcription came back empty.")
                    continue

                print(f"   [Listener] Heard: \"{transcript}\"")
                result = self.detector.detect(transcript, self.current_context)
                print(f"   [Listener] Classified as: {result.mode}"
                      f"{f' (wake word found!)' if result.wake_word_found else ''}")

                if result.mode == "DIRECT":
                    await self._handle_direct_address(result.message or transcript)
                elif result.mode == "OVERHEARD" and transcript.strip():
                    self._overheard_buffer.append(transcript.strip())
                    if len(self._overheard_buffer) > 5:
                        self._overheard_buffer = self._overheard_buffer[-5:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"   [Listener] Error: {e}")
                await asyncio.sleep(pause_seconds)

    def _capture_and_detect_speech(self):
        """
        Synchronous: capture audio and check for speech.
        Returns the raw audio numpy array if speech is detected, else None.
        Runs in an executor so it doesn't block the event loop.
        Uses quiet mode to avoid spamming [MIC ON]/[MIC OFF] every few seconds.
        """
        if not self.audio or not self.audio.is_available:
            return None

        audio = self.audio.capture_audio(quiet=True)
        if audio is None:
            return None

        if not self.audio.has_speech(audio):
            return None

        return audio

    async def _handle_direct_address(self, message: str) -> None:
        """The user said 'Hey Creativity' — respond immediately."""
        print(f"\n   +{'=' * 48}+")
        print(f"   |  DIRECT ADDRESS DETECTED                     |")
        print(f"   +{'=' * 48}+")
        print(f"   User said: \"{message[:70]}\"")

        if not message.strip():
            print(f"\n   [Heard the wake word but no message -- listening for follow-up...]")
            follow_up = await self._listen_for_follow_up()
            if follow_up:
                message = follow_up
            else:
                print("   [No follow-up detected]")
                return

        print(f"   Thinking...")
        reply = await self.responder.respond(message, self.current_context)

        print(f"\n{'═' * 70}")
        print(f"💬 CREATIVITY RESPONDS:\n")
        print(f"   \"{reply}\"")
        print(f"\n{'═' * 70}")

    async def _listen_for_follow_up(self) -> str:
        """After hearing just the wake word, listen for the actual question."""
        if not self.audio or not self.audio.is_available:
            return ""
        print(f"   [MIC ON]  Listening for your question...")
        transcript = await self.audio.quick_capture_and_transcribe()
        print(f"   [MIC OFF] Got it.")
        return transcript

    # ── USER INPUT LOOP ──────────────────────────────────────────────

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
                listening = "🎙️ Active" if self._listening else "Off"
                print(f"\n   📊 Status:")
                print(f"      State: {thinking}")
                print(f"      Listener: {listening}")
                print(f"      Context: \"{self.current_context}\"")
                print(f"      Heartbeats fired: {self.heartbeat.beat_count}")
                print(f"      Next heartbeat: ~{mins}m {secs}s")
                print(f"      Past topics: {len(self.past_topics)}")
                print(f"      Overheard buffer: {len(self._overheard_buffer)} items")
                print(f"      Conversation turns: {len(self.responder.history)}")

            elif cmd == "fire":
                if self._thinking:
                    print("   ⏳ Already thinking — hang on!")
                else:
                    print("   ⚡ Forcing heartbeat NOW!")
                    self.heartbeat._remaining_seconds = 0

            else:
                typed_result = self.detector.detect(user_input, self.current_context)
                if typed_result.mode == "DIRECT":
                    await self._handle_direct_address(typed_result.message or user_input)
                else:
                    self.current_context = user_input
                    print(f"   ✅ Context updated: \"{self.current_context}\"")

    # ── SINGLE-FIRE & INTERACTIVE MODES (unchanged) ──────────────────

    def _print_citations(self, interjection: Interjection) -> None:
        """Print source URLs and facts from an interjection's web search."""
        if interjection.search_facts:
            print(f"🔍 Grounded in {len(interjection.search_facts)} web facts"
                  f"{f' from {len(interjection.search_sources)} sources' if interjection.search_sources else ''}")
        if interjection.search_sources:
            for url in interjection.search_sources[:5]:
                print(f"   📎 {url}")

    async def run_single(self, seed_topic: str) -> None:
        """Fire a single heartbeat cycle — for testing and demos."""
        print("=" * 70)
        print("🧠 CREATIVITY ENGINE — Proof of Concept")
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
            print(f"💬 CREATIVITY ENGINE SAYS:\n")
            print(f"   \"{interjection.interjection_text}\"")
            print(f"\n{'═' * 70}")
            print(f"\n⏱️  Total pipeline time: {elapsed:.1f}s")
            print(f"📍 Internal chain: {best_chain.summary()}")
            print(f"📊 Interest score: {best_score.total:.3f}")
            self._print_citations(interjection)
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
        print("🧠 CREATIVITY ENGINE — Interactive Mode")
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
                print("\n\n👋 Creativity Engine shutting down. Stay creative!")
                break

            if not topic:
                continue
            if topic.lower() in ("quit", "exit", "q"):
                print("\n👋 Creativity Engine shutting down. Stay creative!")
                break

            await self.run_single(topic)
            print()


def print_devices():
    """List all available cameras and microphones."""
    print("=" * 70)
    print("   AVAILABLE DEVICES")
    print("=" * 70)

    print("\n   CAMERAS (Vision):")
    print("   " + "-" * 50)
    cameras = VisionChannel.list_devices()
    if cameras:
        for cam in cameras:
            status = "OK" if cam["working"] else "FOUND (no frame)"
            print(f"   [{cam['index']}] {cam['name']}  -- {status}")
    else:
        print("   No cameras detected (is opencv-python installed?)")

    print("\n   MICROPHONES (Audio):")
    print("   " + "-" * 50)
    mics = AudioChannel.list_devices()
    if mics:
        for mic in mics:
            default_tag = " <-- DEFAULT" if mic["is_default"] else ""
            print(f"   [{mic['index']}] {mic['name']}")
            print(f"        channels={mic['channels']}, rate={mic['sample_rate']}Hz{default_tag}")
    else:
        print("   No microphones detected (is sounddevice installed?)")

    print("\n   " + "-" * 50)
    print("   To select devices, add to your config.yaml:\n")
    print("   input_pipeline:")
    print("     vision:")
    print("       device_index: 0    # camera number from list above")
    print("     audio:")
    print("       device_index: 0    # microphone number from list above")
    print()


async def main():
    args = sys.argv[1:]

    if "--devices" in args:
        print_devices()
        return

    config = load_config()
    engine = CreativityEngine(config)

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
