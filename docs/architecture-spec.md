# 🏗️ Architecture Spec — Curiosity Engine

**Version:** 0.1 (Draft)  
**Last Updated:** May 2026  
**Reference:** [project-vision.md](../project-vision.md)

---

## System Overview

The Curiosity Engine is a **proactive multimodal AI companion** that runs as a persistent background process. It perceives the user's environment through vision, audio, and text; generates creative associations on a timer-driven heartbeat; and delivers interjections through a conversational interface.

```
┌─────────────────────────────────────────────────────────────────┐
│                      CURIOSITY ENGINE                           │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              INPUT PIPELINE (Multimodal)                │     │
│  │                                                        │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │     │
│  │  │ 👁️ Vision │  │ 👂 Audio  │  │ 📖 Text/Context  │     │     │
│  │  │ (Webcam) │  │ (Mic)    │  │ (Screen/Docs)    │     │     │
│  │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘     │     │
│  │       │              │                 │               │     │
│  │       ▼              ▼                 ▼               │     │
│  │  ┌──────────────────────────────────────────────┐      │     │
│  │  │         NOVELTY WEIGHTING ENGINE              │      │     │
│  │  │  (Compares each input to recent history,      │      │     │
│  │  │   assigns weight based on how NEW it is)      │      │     │
│  │  └──────────────────┬───────────────────────────┘      │     │
│  └─────────────────────┼──────────────────────────────────┘     │
│                        │                                         │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CONTEXT ASSEMBLER                           │    │
│  │  (Combines weighted inputs into a unified snapshot       │    │
│  │   of "what's happening right now")                       │    │
│  └────────────────────┬────────────────────────────────────┘    │
│                       │                                          │
│           ┌───────────┴───────────┐                              │
│           ▼                       ▼                              │
│  ┌─────────────────┐    ┌──────────────────┐                    │
│  │  💓 HEARTBEAT    │    │  DIRECT ADDRESS   │                    │
│  │  (Random Timer)  │    │  DETECTOR         │                    │
│  │  Fires every     │    │  (Wake word /     │                    │
│  │  1-10 min        │    │   attention cue)  │                    │
│  └────────┬─────────┘    └────────┬─────────┘                    │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌─────────────────┐    ┌──────────────────┐                    │
│  │  🌳 ASSOCIATION  │    │  DIRECT RESPONSE  │                    │
│  │  TREE GENERATOR  │    │  ENGINE           │                    │
│  │  (Branching      │    │  (Conversational  │                    │
│  │   creative       │    │   reply when      │                    │
│  │   exploration)   │    │   addressed)      │                    │
│  └────────┬─────────┘    └────────┬─────────┘                    │
│           │                       │                              │
│           ▼                       │                              │
│  ┌─────────────────┐              │                              │
│  │  🔍 WEB SEARCH   │              │                              │
│  │  (Research the   │              │                              │
│  │   top branches)  │              │                              │
│  └────────┬─────────┘              │                              │
│           │                       │                              │
│           ▼                       │                              │
│  ┌─────────────────┐              │                              │
│  │  📊 INTEREST     │              │                              │
│  │  SCORER          │              │                              │
│  │  (Score & rank   │              │                              │
│  │   candidates)    │              │                              │
│  └────────┬─────────┘              │                              │
│           │                       │                              │
│           ▼                       │                              │
│  ┌─────────────────┐              │                              │
│  │  🌉 BRIDGE       │              │                              │
│  │  BUILDER         │              │                              │
│  │  (Construct      │              │                              │
│  │   narrative)     │              │                              │
│  └────────┬─────────┘              │                              │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  OUTPUT MANAGER                           │   │
│  │  (Formats and delivers interjection or response          │   │
│  │   through the appropriate channel — voice, text, etc.)   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  MEMORY STORE                             │   │
│  │  (Conversation history, past associations, user          │   │
│  │   preferences, novelty baselines, incubation queue)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Input Pipeline

The input pipeline continuously ingests data from three modalities and passes it through novelty weighting before assembling a unified context snapshot.

#### 1a. Vision Module
| Property | Value |
|---|---|
| **Source** | User's webcam |
| **Capture Trigger** | On each heartbeat fire + on-demand |
| **Output** | Image frame (JPEG/PNG) |
| **Processing** | Sent to LLM vision endpoint OR image embedding model |
| **Novelty Check** | Compare embedding to rolling window of last N captures |

#### 1b. Audio Module
| Property | Value |
|---|---|
| **Source** | User's microphone |
| **Processing** | Speech-to-text (STT) engine → text transcript |
| **Mode Detection** | Classify as DIRECT_ADDRESS or OVERHEARD |
| **Direct Address Cues** | Wake word ("Hey Curiosity"), name detection, gaze toward camera |
| **Overheard Cues** | No wake word, speech directed elsewhere, phone call patterns |
| **Output** | Transcript text + mode label (DIRECT / OVERHEARD) |

#### 1c. Text/Context Module
| Property | Value |
|---|---|
| **Sources** | Active window title, clipboard content, URLs, document text |
| **Capture** | Periodic polling OR OS-level accessibility hooks |
| **Output** | Text summary of current user activity |
| **Privacy** | User-configurable exclusion rules (e.g., ignore banking sites) |

#### 1d. Novelty Weighting Engine
Each input channel produces a **novelty score** between 0.0 and 1.0:

```python
novelty = 1.0 - max_similarity_to_recent_inputs(current_input, history_window)
weighted_input = base_weight * novelty
```

The history window is configurable per channel:
- Vision: last 10 captures
- Audio: last 5 minutes of transcript
- Text: last 10 context snapshots

**Decay behavior:** If the same input repeats across multiple heartbeats, its novelty approaches 0. The engine effectively stops "noticing" static environments — just like a human friend stops noticing your wallpaper but immediately notices a new poster.

---

### 2. Context Assembler

Combines all weighted inputs into a single **Context Snapshot** object:

```python
@dataclass
class ContextSnapshot:
    timestamp: datetime
    
    # Vision
    image: bytes | None
    image_description: str
    image_novelty: float        # 0.0 - 1.0
    
    # Audio
    transcript: str
    audio_mode: Literal["DIRECT", "OVERHEARD", "SILENCE"]
    audio_novelty: float        # 0.0 - 1.0
    
    # Text/Context
    active_context: str         # what the user is working on
    text_novelty: float         # 0.0 - 1.0
    
    # Composite
    dominant_channel: str       # which input has highest novelty right now
    overall_novelty: float      # weighted average across all channels
```

The Context Assembler produces a new snapshot on every heartbeat AND whenever direct address is detected (immediate snapshot).

---

### 3. Heartbeat (Curiosity Clock)

The random timer that creates the illusion of spontaneous thought.

```python
import random

class Heartbeat:
    def __init__(self, min_minutes=1, max_minutes=10):
        self.min_minutes = min_minutes
        self.max_minutes = max_minutes
    
    def next_interval(self) -> int:
        """Returns next heartbeat interval in whole minutes."""
        return random.randint(self.min_minutes, self.max_minutes)
```

**On each heartbeat fire:**
1. Capture a Context Snapshot (vision + audio + text)
2. If overall_novelty > threshold → proceed to Association Tree
3. If overall_novelty < threshold → skip this beat (nothing interesting happening)
4. Generate next random interval and reset timer

**Adaptive behavior (future):** The heartbeat could become more frequent when novelty is high (the friend is engaged and excited) and less frequent when novelty is low (the friend is quietly hanging out).

---

### 4. Direct Address Detector

Separates "talking TO the engine" from "overhearing the user's life."

| Signal | Classification | Behavior |
|---|---|---|
| Wake word detected | DIRECT | Bypass heartbeat, respond immediately |
| User looking at camera + speaking | DIRECT | Bypass heartbeat, respond immediately |
| User on phone call | OVERHEARD | Feed into context, do NOT respond |
| User talking to someone else in room | OVERHEARD | Feed into context, do NOT respond |
| Silence | SILENCE | No audio input this cycle |

When DIRECT is detected:
- The engine pauses the heartbeat cycle
- Takes an immediate Context Snapshot
- Routes to the Direct Response Engine (conversational reply)
- Resumes heartbeat after the conversation naturally ends

When OVERHEARD is detected:
- Content feeds into the Context Assembler at reduced weight
- But it IS available as association fuel for the next heartbeat

---

### 5. Association Tree Generator

The creative core. Generates a branching tree of lateral associations.

**Configuration:**
| Parameter | Value | Rationale |
|---|---|---|
| Branches per node | 3 | Ternary tree — enough diversity, manageable compute |
| Min depth | 4 hops | Minimum distance for interesting associations |
| Max depth | 7 hops | Beyond this, connections become too tenuous |
| Variable depth | Yes | Depth inversely proportional to semantic distance per hop |
| Domain crossing required | Yes (min 1) | At least one hop must cross into a different domain |

**Process:**
1. Extract **seed topics** from the Context Snapshot (weighted by novelty)
2. For each seed, generate 3 lateral associations (Hop 1 — MUST be grounded in context)
3. For each of those 3, generate 3 more (Hop 2 — free to explore)
4. Continue branching until min depth reached
5. At each hop, check semantic distance — if a single hop covers large distance, can stop earlier
6. At max depth, collect all leaf nodes as **candidate endpoints**

**Output:** Up to 243 candidate endpoints (3^5 at depth 5) with their full association chain paths.

---

### 6. Web Search Module

Takes the top-scoring candidate endpoints and searches the web for supporting content.

**Process:**
1. Take top 3–5 candidate endpoints from the Interest Scorer (preliminary pass)
2. For each, construct a search query from the endpoint topic
3. Execute web search via API (Tavily, SerpAPI, Brave Search, etc.)
4. Extract key facts, surprising statistics, or notable connections
5. Feed search results back into the Interest Scorer for final scoring

---

### 7. Interest Scorer

Evaluates each candidate endpoint to determine if it's worth sharing.

**Scoring Formula:**
```
interest_score = (semantic_distance × 0.30)
               + (domain_crossings × 0.25)
               + (surprise × 0.20)
               + (bridgeability × 0.15)
               + (novelty × 0.10)
```

| Metric | How It's Computed |
|---|---|
| **Semantic Distance** | Cosine distance between embedding of seed topic and endpoint |
| **Domain Crossings** | Count of category/field boundaries crossed in the chain |
| **Surprise** | Inverse probability the LLM assigns to this association given context |
| **Bridgeability** | LLM-evaluated: "Can you explain why this is interesting given the conversation?" Score 0-1 |
| **Novelty** | 1.0 minus similarity to any previous interjection in conversation history |

**Threshold:** Only endpoints scoring above a configurable threshold (default: 0.65) proceed to the Bridge Builder.

**Incubation Queue:** Endpoints scoring between 0.40–0.65 are saved to an "incubation queue." On future heartbeats, they're re-evaluated against new context — a mediocre idea today may become brilliant when context shifts.

---

### 8. Bridge Builder

Constructs the narrative that connects the winning association back to the conversation.

**Input:** The winning endpoint + its full association chain + web search results + current context
**Output:** A natural-language interjection that feels like a friend sharing an interesting thought

**Prompt pattern (LLM-agnostic):**
```
You are a curious, creative friend. You've been hanging out with the user and 
just had an interesting thought. Here's the chain of associations that led you 
there: [chain]. Here's what you found: [search results]. Here's what the user 
is currently doing: [context].

Share your thought naturally — as if you just thought of something cool and 
want to mention it. Don't explain the association chain. Don't be academic. 
Be conversational, warm, and genuinely enthusiastic. The thought should feel 
spontaneous, not rehearsed.

Keep it brief — 2-3 sentences max. Like a friend tapping you on the shoulder.
```

---

### 9. Output Manager

Delivers the interjection through the appropriate channel.

**Possible delivery channels (to be decided):**
- Desktop notification (toast/popup)
- Text-to-speech (spoken aloud)
- Chat interface (web app, terminal, Discord)
- Overlay widget on screen

**Rate limiting:** Maximum 1 interjection per heartbeat cycle. If the user says "not now" or ignores an interjection, increase the minimum heartbeat interval temporarily.

---

### 10. Memory Store

Persistent storage for conversation history, associations, and user preferences.

**Stores:**
| Data | Purpose | Storage |
|---|---|---|
| Conversation history | Context for future associations | Vector DB (ChromaDB / similar) |
| Past interjections | Novelty checking (don't repeat) | SQLite + embeddings |
| Association chains | Audit trail, learning | SQLite |
| Incubation queue | Re-evaluate later | SQLite |
| Novelty baselines | Per-channel input history | In-memory rolling window + periodic flush |
| User preferences | What topics the user engaged with most | SQLite + analytics |

---

## Data Flow Summary

### Proactive Path (Heartbeat-Driven)
```
Timer fires → Capture inputs → Novelty weight → Assemble context
→ Generate association tree → Search web → Score candidates
→ Build bridge narrative → Deliver interjection → Store in memory
```

### Reactive Path (Direct Address)
```
Wake word detected → Capture inputs → Assemble context
→ Generate conversational response → Deliver response → Store in memory
```

### Passive Path (Overhearing)
```
Audio/text detected → Classify as OVERHEARD → Novelty weight
→ Store in context buffer → Available as fuel for next heartbeat
```

---

## Technology Stack (Preliminary)

| Component | Candidates | Notes |
|---|---|---|
| **Language** | Python 3.11+ | Async support, rich ML ecosystem |
| **Async Runtime** | asyncio + APScheduler | For heartbeat timer and concurrent I/O |
| **LLM Interface** | LiteLLM or custom adapter | LLM-agnostic — swap providers freely |
| **Vision Capture** | OpenCV (cv2) | Cross-platform webcam access |
| **Audio Capture** | PyAudio or sounddevice | Microphone access |
| **Speech-to-Text** | Whisper (local) or cloud STT | Transcription of audio |
| **Embeddings** | sentence-transformers (local) or API | For novelty comparison and semantic distance |
| **Web Search** | Tavily / SerpAPI / Brave | For association research |
| **Vector DB** | ChromaDB | Conversation memory and retrieval |
| **Relational DB** | SQLite | Associations, incubation queue, preferences |
| **TTS (output)** | TBD | If voice output desired |
| **UI** | TBD | Desktop widget, web app, or terminal |

---

## What We're NOT Building (Scope Boundaries)

- ❌ A general-purpose assistant (no "set a timer", "what's the weather")
- ❌ A task manager or productivity tool
- ❌ A surveillance system (all input processing is local, user-controlled)
- ❌ An always-listening always-recording system (captures only on heartbeat or direct address)
- ❌ A replacement for the user's primary AI coding assistant (this is a *companion*, not a *tool*)
