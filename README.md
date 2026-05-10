# 🧠 Creativity Engine

**A proactive AI companion that simulates genuine curiosity through hidden causal complexity.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is This?

The Curiosity Engine is an AI companion that **hangs out with you** — like a creative friend sitting next to you while you work. It doesn't wait for you to ask it questions. It watches, listens, and thinks on its own. Every few minutes, it might tap you on the shoulder with something surprising:

> *"Oh hey — did you know that NASA actually studied how construction workers communicate under stress when they were designing protocols for ISS repair crews? Turns out the hand signals are almost identical."*

It's not a chatbot. It's not an assistant. It's a **companion with curiosity.**

## Core Philosophy

> **Intelligence is the ability to make correct or valuable associations between things that don't seem like they could be associated.**

The engine simulates creative curiosity through three mechanisms:

1. **💓 The Heartbeat** — A random timer (1–10 min) creates the illusion of spontaneous thought
2. **🌳 The Association Tree** — A branching tree of lateral associations (4–7 hops deep) generates surprising cross-domain connections
3. **📊 The Interest Scorer** — A weighted scoring system filters for genuinely interesting insights, not noise

The philosophical foundation: *"Free will" is causation complex enough to be invisible.* If the hidden causal chain is sufficiently intricate, the output is indistinguishable from genuine curiosity.

## How It Works

```
Heartbeat fires → Capture what's happening (webcam, mic, screen)
    → Weight inputs by novelty (new things matter more)
    → Generate branching association tree (4-7 creative hops)
    → Search the web for facts about the best branches
    → Score for interestingness (bold > safe)
    → Build a natural conversational interjection
    → Share the thought like a friend would
```

## Multimodal Perception

The engine perceives your world through three channels, each weighted by **novelty**:

| Channel | Source | What It Does |
|---|---|---|
| 👁️ **Vision** | Webcam | Sees what's physically happening. Same desk for the 10th time? Low weight. New whiteboard drawing? High weight. |
| 👂 **Audio** | Microphone | Hears you and distinguishes between talking TO the engine vs. overhearing your conversations. |
| 📖 **Text** | Screen/docs | Knows what you're reading, writing, and browsing. New article? High weight. Same doc for 2 hours? Low weight. |

## Project Structure

```
Curiosity-Engine/
├── project-vision.md          # Philosophy, principles, success criteria
├── docs/
│   ├── architecture-spec.md   # Full system component design
│   ├── input-pipeline-spec.md # Multimodal perception & novelty weighting
│   └── curiosity-engine-spec.md # Association tree, scoring, bridge builder
├── src/
│   ├── input_pipeline/        # Vision, audio, text capture & novelty
│   ├── heartbeat/             # Random timer (curiosity clock)
│   ├── association_engine/    # Branching association tree generator
│   ├── search/                # Web search integration
│   ├── scoring/               # Interest scorer & incubation queue
│   ├── bridge_builder/        # Narrative construction for interjections
│   ├── output/                # Delivery (notifications, voice, chat)
│   ├── memory/                # Conversation history & vector store
│   └── config/                # Configuration management
├── tests/                     # Test suite
├── LICENSE                    # MIT License
└── README.md                  # You are here
```

## Status

🚧 **Early stage** — Architecture and spec are defined. Implementation starting soon.

## Author

**Steven Sills II** — [GitHub](https://github.com/Apixa25)

## License

MIT — see [LICENSE](LICENSE)
