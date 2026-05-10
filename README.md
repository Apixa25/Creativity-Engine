# 🧠 Creativity Engine

**A proactive AI companion that simulates genuine creativity through hidden causal complexity.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is This?

The Creativity Engine is an AI companion that **hangs out with you** — like a creative friend sitting next to you while you work. It doesn't wait for you to ask it questions. It watches, listens, and thinks on its own. Every few minutes, it might tap you on the shoulder with something surprising:

> *"Oh hey — did you know that NASA actually studied how construction workers communicate under stress when they were designing protocols for ISS repair crews? Turns out the hand signals are almost identical."*

It's not a chatbot. It's not an assistant. It's a **companion with creativity.**

You can also talk to it directly — just say **"Hey Creativity"** and it responds immediately, like a friend who was already paying attention.

---

## Core Philosophy

> **Intelligence is the ability to make correct or valuable associations between things that don't seem like they could be associated.**

The engine simulates creative thinking through three core mechanisms:

1. **💓 The Heartbeat** — A random timer (1–10 min) creates the illusion of spontaneous thought
2. **🌳 The Association Tree** — A branching tree of lateral associations (4–7 hops deep) generates surprising cross-domain connections
3. **📊 The Interest Scorer** — A 5-metric weighted scoring system filters for genuinely interesting insights, not noise

The philosophical foundation: *"Free will" is causation complex enough to be invisible.* If the hidden causal chain is sufficiently intricate, the output is indistinguishable from genuine creativity.

---

## Features — What's Built & Working ✅

### 🎙️ Live Companion Mode
Run `python -m src.main --live` and the engine becomes a persistent background companion:
- Proactive heartbeat-driven creative interjections
- Continuous background audio listening
- Interactive terminal commands (`status`, `fire`, `not now`, `quit`)
- Real-time context updates as you work

### 💬 Direct Conversation ("Hey Creativity")
- **Wake word detection** — Say "Hey Creativity" and the engine responds immediately
- **Direct vs. Overheard classification** — It knows the difference between you talking to IT and you talking to someone else nearby
- **Conversational personality** — Responds like a friend, not an assistant — warm, curious, goes on tangents
- **Conversation history** — Remembers what you've talked about within a session

### 👁️ Multimodal Perception
The engine perceives your world through three channels, each weighted by **novelty**:

| Channel | Source | What It Does |
|---|---|---|
| 👁️ **Vision** | Webcam | Sees what's physically happening. Same desk for the 10th time? Low weight. New whiteboard drawing? High weight. |
| 👂 **Audio** | Microphone | Hears you and distinguishes between talking TO the engine vs. overhearing your conversations. |
| 📖 **Text** | Screen/docs | Knows what you're reading, writing, and browsing. New article? High weight. Same doc for 2 hours? Low weight. |

- **Novelty weighting**: Inputs that are new/changing get higher weight; static inputs fade over time
- **Perception window notifications**: Clear `[CAMERA ON]` / `[MIC ON]` banners so you know when to provide input
- **First-observation baseline**: Channels start at 0.5 novelty (not 1.0) to avoid false-high weighting on startup

### 🌳 Association Tree Generator
- **Ternary branching** — 3 paths explored per node
- **Variable depth** — 4–7 creative hops, driven by semantic distance
- **Depth-first pruning** — Only the top 3 branches survive at each level (keeps things fast)
- **Domain classification** — Each node tagged by field (Science, Psychology, Art, Tech, etc.)
- **Cross-domain rewarding** — The more domain boundaries crossed, the higher the score

### 📊 Interest Scorer
A 5-metric weighted formula determines if an association chain is worth sharing:

```
interest_score = (semantic_distance × 0.30)    — reward boldness
               + (domain_crossings × 0.25)     — reward crossing fields
               + (surprise × 0.20)             — reward the unexpected
               + (bridgeability × 0.15)        — can we tell a compelling story?
               + (novelty × 0.10)              — haven't said this before
```

Chains scoring above the fire threshold get shared. Below that but above incubation? Saved for later re-evaluation.

### 🔍 Web Search Integration (Tavily)
- **LLM-crafted search queries** from the best association chain
- **Fact extraction** — Pulls specific, surprising facts from search results
- **Citation display** — Sources shown alongside interjections
- **Graceful fallback** — Uses LLM knowledge when no API key or search fails

### 🌉 Bridge Builder
- Constructs natural, conversational interjections from the internal association chain
- **Hides the machinery** — The user sees a friendly thought, not a scored algorithm output
- Weaves in web-sourced facts for specificity and credibility

### 🔧 LLM-Agnostic Architecture
- **Adapter pattern** — Swap between OpenAI and Anthropic via config
- `generate()`, `generate_json()`, `generate_float()` — Unified interface for all LLM calls
- Easy to add new providers without touching core logic

---

## How It Works

```
                        ┌──────────────────────────────┐
                        │   "Hey Creativity, what       │
                        │    do you think about X?"     │
                        └─────────────┬────────────────┘
                                      │ (wake word detected)
                                      ▼
                             DIRECT RESPONSE ENGINE
                          (immediate conversational reply)

                                  — OR —

💓 Heartbeat fires → Capture what's happening (webcam, mic, screen)
    → Weight inputs by novelty (new things matter more)
    → Blend in overheard speech from background listener
    → Generate branching association tree (4–7 creative hops)
    → Score chains for interestingness (bold > safe)
    → Search the web for facts about the best chain
    → Build a natural conversational interjection with citations
    → Share the thought like a friend would
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key (for GPT + Whisper transcription)
- Tavily API key (optional, for web search grounding)
- Webcam and microphone (optional, for multimodal perception)

### Installation

```bash
git clone https://github.com/Apixa25/Curiosity-Engine.git
cd Curiosity-Engine/Creativity-Engine

pip install -r requirements.txt
```

### Set Your API Keys

**PowerShell (permanent — recommended):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-proj-your-key", "User")
[System.Environment]::SetEnvironmentVariable("TAVILY_API_KEY", "tvly-your-key", "User")
```

**PowerShell (session only):**
```powershell
$env:OPENAI_API_KEY = "sk-proj-your-key"
$env:TAVILY_API_KEY = "tvly-your-key"
```

### Run It

**Live companion mode** (recommended — full experience):
```bash
python -m src.main --live
```

**Single-shot mode** (one creative cycle, then exit):
```bash
python -m src.main "building software for contractors"
```

### Live Mode Commands
| Command | What It Does |
|---|---|
| Type anything | Updates the engine's context about what you're doing |
| `status` | Shows heartbeat timer, listener state, overheard buffer, conversation turns |
| `fire` | Forces an immediate creative cycle (skip the heartbeat wait) |
| `not now` | Delays the next heartbeat (engine backs off) |
| `quit` | Graceful shutdown |
| Say "Hey Creativity" | Triggers an immediate conversational response via microphone |

---

## Project Structure

```
Creativity-Engine/
├── project-vision.md              # Philosophy, principles, success criteria
├── TODO.md                        # Remaining work tracker
├── config.example.yaml            # All tunable parameters
├── requirements.txt               # Python dependencies
├── docs/
│   ├── architecture-spec.md       # Full system component design
│   ├── input-pipeline-spec.md     # Multimodal perception & novelty weighting
│   └── creativity-engine-spec.md  # Association tree, scoring, bridge builder
├── src/
│   ├── main.py                    # Orchestrator — live mode, heartbeat loop, input handling
│   ├── models.py                  # Core data models (AssociationNode, Chain, Interjection)
│   ├── config/
│   │   ├── llm_adapter.py         # LLM-agnostic adapter (OpenAI, Anthropic)
│   │   └── settings.py            # YAML config loader with dataclasses
│   ├── heartbeat/
│   │   └── clock.py               # Random timer (creativity clock) with backoff
│   ├── input_pipeline/
│   │   ├── vision.py              # Webcam capture, perceptual hashing, novelty
│   │   ├── audio.py               # Mic capture, VAD, Whisper transcription, novelty
│   │   ├── assembler.py           # Combines channels with novelty weighting
│   │   └── address_detector.py    # Wake word detection, DIRECT vs OVERHEARD classification
│   ├── association_engine/
│   │   └── tree_generator.py      # Branching association tree (ternary, depth 4–7)
│   ├── scoring/
│   │   └── interest_scorer.py     # 5-metric weighted scoring + pre-filtering
│   ├── search/
│   │   └── web_search.py          # Tavily web search, LLM query construction, fact extraction
│   ├── bridge_builder/
│   │   └── builder.py             # Natural interjection generation from chains + facts
│   ├── conversation/
│   │   └── responder.py           # Direct response engine (conversation history, persona)
│   ├── memory/                    # 🚧 Planned — vector DB, conversation persistence
│   └── output/                    # 🚧 Planned — desktop notifications, TTS, overlay
└── tests/                         # Test suite
```

---

## Configuration

All parameters are tunable via `config.example.yaml`:

| Section | Key Parameters |
|---|---|
| **Heartbeat** | `min_minutes`, `max_minutes` — controls how often the engine thinks |
| **Vision** | `base_weight`, `min_novelty_for_description` — webcam sensitivity |
| **Audio** | `base_weight_direct`, `base_weight_overheard`, `wake_word` — mic behavior |
| **Text** | `base_weight`, `excluded_apps`, `excluded_urls` — screen awareness |
| **Association Tree** | `branching_factor`, `min_depth`, `max_depth` — creativity chain shape |
| **Scoring** | All 5 metric weights, `fire_threshold`, `incubation_threshold` |
| **LLM** | `provider`, `model`, `api_key_env` — swap LLMs without code changes |
| **Search** | `provider`, `results_per_query` — web grounding settings |

---

## LLM Calls Per Heartbeat Cycle

A typical creative cycle makes **~15–20 LLM API calls**:

| Step | Calls | Purpose |
|---|---|---|
| Vision description | 1 | Describe what the webcam sees |
| Context assembly | 1 | Synthesize seed topic from all inputs |
| Association tree | ~6–8 | Generate 3 branches at each depth level |
| Scoring (surprise) | ~3–5 | Evaluate surprise for top chains |
| Scoring (bridgeability) | ~3–5 | Evaluate bridgeability for top chains |
| Search query | 1 | Craft a web search query |
| Fact extraction | 1 | Pull interesting facts from search results |
| Bridge building | 1 | Write the final conversational interjection |

Direct conversation responses ("Hey Creativity") use **1 LLM call** for an immediate reply.

---

## Roadmap

See [TODO.md](TODO.md) for the full breakdown. Key remaining work:

- **🧠 Memory & Learning** — Vector DB (ChromaDB), persistent conversation history, user preference tracking
- **🔄 Incubation Queue** — Save interesting-but-not-ready ideas, re-evaluate against new context
- **📢 Output Channels** — Desktop notifications, text-to-speech, web chat interface, screen overlay
- **🎚️ Adaptive Heartbeat** — Speed up when novelty is high, slow down during quiet periods
- **🌅 Medium/Long Cycles** — Hourly "I was thinking about..." and daily synthesized reflections
- **🎭 Tone Calibration** — Detect user state (focused, relaxed, frustrated) and adapt personality
- **📐 Real Semantic Distance** — Replace heuristic with actual embedding cosine distance
- **📺 Full Text/Context Module** — Active window detection, clipboard awareness, URL extraction

---

## Author

**Steven Sills II** — [GitHub](https://github.com/Apixa25)

## License

MIT — see [LICENSE](LICENSE)
