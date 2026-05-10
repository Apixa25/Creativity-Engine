# Creativity Engine — Remaining Work

**Last Updated:** May 10, 2026

---

## Completed

- [x] Heartbeat (Creativity Clock) — random timer, continuous loop, backoff, fire command
- [x] Association Tree Generator — ternary branching, depth 4-7, depth-first pruning, domain classification
- [x] Interest Scorer — 5-metric weighted formula, pre-filtering, calibrated prompts
- [x] Bridge Builder — natural interjection generation, hides chain, uses search facts
- [x] Web Search (Tavily) — LLM query construction, fact extraction, fallback
- [x] Vision Channel — webcam capture, perceptual hashing novelty, LLM image description
- [x] Audio Channel — mic capture, VAD, Whisper transcription, Jaccard novelty
- [x] Context Assembler — combines channels with novelty weighting, LLM seed topic extraction
- [x] LLM-Agnostic Adapter — OpenAI + Anthropic, swap via config
- [x] YAML Configuration — all parameters externalized and tunable
- [x] Live Companion Mode — continuous background operation, user commands
- [x] Sensor Notifications — camera/mic on/off banners in terminal
- [x] Citation Display — web search sources shown with interjections
- [x] First-observation novelty fix — channels start at 0.5 instead of 1.0

---

## Priority 1 — Direct Address & Conversation ✅ COMPLETE

### Direct Address Detector ✅
- [x] Wake word detection ("Hey Creativity") on audio transcripts
- [x] Classify audio as DIRECT (talking to engine) vs OVERHEARD (talking to someone else)
- [x] DIRECT bypasses heartbeat, triggers immediate response
- [x] OVERHEARD feeds into context assembler at reduced weight
- [x] Continuous background listening (not just on heartbeat)

### Direct Response Engine ✅
- [x] Conversational reply when user addresses the engine directly
- [x] Uses current context + conversation history for relevant responses
- [x] Responds immediately (no heartbeat wait)
- [ ] Can still trigger creative associations if the question inspires one

---

## Priority 2 — Memory & Learning

### Memory Store
- [ ] Vector DB (ChromaDB) for conversation history and retrieval
- [ ] SQLite for past interjections, association chains, incubation queue
- [ ] Persistent novelty checking across sessions (not just in-memory)
- [ ] User preference tracking — which interjections got engagement vs ignored
- [ ] Association chain audit trail

### Incubation Queue
- [ ] Save chains scoring between incubation and fire thresholds
- [ ] Re-evaluate 1-2 incubated ideas per heartbeat against new context
- [ ] Expire after 5 re-evaluations or 24 hours
- [ ] Context-shift detection — flag when context changes enough to re-score

---

## Priority 3 — Output & Delivery

### Output Manager
- [ ] Desktop notification (toast/popup) delivery channel
- [ ] Text-to-speech (spoken aloud) option
- [ ] Chat interface (web app or terminal UI)
- [ ] Overlay widget on screen
- [ ] Rate limiting — max 1 interjection per heartbeat cycle

---

## Priority 4 — Refinement & Polish

### Adaptive Heartbeat
- [ ] More frequent when novelty is high (engine is engaged)
- [ ] Less frequent when novelty is low (quietly hanging out)
- [ ] Dynamic range adjustment based on user engagement patterns

### Medium & Long Heartbeat Cycles
- [ ] Medium cycle (hours) — "Hey, I was thinking about something..."
- [ ] Long cycle (days) — deeper reflections, synthesized insights across sessions

### Tone Calibration
- [ ] Detect user state (deep work, relaxed, frustrated, late night)
- [ ] Adapt interjection tone accordingly
- [ ] Brief/low-key during focus, conversational when relaxed, fun when frustrated

### Real Semantic Distance
- [ ] Replace logarithmic heuristic with actual embedding cosine distance
- [ ] Local sentence-transformers model or API-based embeddings
- [ ] Normalize based on observed distribution

### Full Text/Context Module
- [ ] Active window title capture
- [ ] Clipboard content awareness
- [ ] URL/document text extraction
- [ ] OS-level accessibility hooks
- [ ] Privacy exclusion rules (configurable ignored apps/sites)

### User Engagement Tracking
- [ ] Detect if user responded to interjection (typed reply, spoke back)
- [ ] Track engagement metrics per topic/domain
- [ ] Feed engagement data back into scoring weights over time
