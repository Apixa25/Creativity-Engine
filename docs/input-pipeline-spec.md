# 📡 Input Pipeline Spec — Multimodal Perception & Novelty Weighting

**Version:** 0.1 (Draft)  
**Reference:** [architecture-spec.md](./architecture-spec.md) · [project-vision.md](../project-vision.md)

---

## Overview

The Input Pipeline is the Curiosity Engine's sensory system. It answers the question: **"What is happening around the user right now, and how much of it is new?"**

Three channels (vision, audio, text) feed through a novelty weighting engine into a unified Context Snapshot. The key principle: **new things matter more than familiar things.** A friend sitting next to you doesn't keep commenting on your wallpaper — they notice when something changes.

---

## 1. Vision Channel (Webcam)

### Capture Strategy

The webcam does **not** stream continuously. It captures a single frame at specific moments:

| Trigger | When |
|---|---|
| **Heartbeat fire** | Every 1–10 minutes (random interval) |
| **Direct address detected** | When the user says the wake word |
| **High audio novelty** | If something unusual is heard, look to see what's happening |

This is a deliberate design choice — we're building a **companion**, not a surveillance system. Captures only when the engine is "paying attention."

### Image Processing Pipeline

```
Webcam frame (raw)
    │
    ▼
┌─────────────────────────┐
│  Image Embedding Model   │  → Produces a fixed-size vector
│  (CLIP, SigLIP, etc.)   │     representing what's in the image
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Novelty Comparison      │  → Cosine similarity against
│                          │     last N image embeddings
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Image Description       │  → LLM vision: "Describe what you see"
│  (only if novelty > 0.3) │     Skip description for low-novelty frames
└───────────┬─────────────┘
            │
            ▼
  Output: image_bytes, image_description, image_novelty
```

### Novelty Calculation for Vision

```python
def compute_image_novelty(current_embedding, history: list[np.ndarray]) -> float:
    """
    Compare current image embedding to recent history.
    Returns 0.0 (identical to recent) to 1.0 (completely new scene).
    """
    if not history:
        return 1.0  # first image is always maximally novel
    
    similarities = [
        cosine_similarity(current_embedding, past_embedding)
        for past_embedding in history
    ]
    max_similarity = max(similarities)
    
    novelty = 1.0 - max_similarity
    return max(0.0, min(1.0, novelty))
```

### Vision History Window

- **Window size:** Last 10 captures
- **Behavior:** Rolling window, FIFO. Oldest capture drops off when a new one enters.
- **Result:** If you've been sitting at your desk for 10 heartbeats, the 11th capture of the same scene has novelty ≈ 0.02 (effectively zero). But if you turn the camera to show a whiteboard with new drawings, novelty spikes to ≈ 0.85.

### Edge Cases

| Scenario | Novelty | Behavior |
|---|---|---|
| Camera covered / off | N/A | Vision channel produces null, other channels compensate |
| Same scene 10+ times | ~0.0 | Vision effectively muted, audio/text dominate |
| User moved to new room | ~0.95 | Vision dominates this heartbeat's context |
| Subtle change (new coffee mug) | ~0.3–0.5 | Moderate — mentioned if interesting, not forced |
| Monitor shows new content | ~0.6–0.8 | High — screen content is a strong signal of activity change |

---

## 2. Audio Channel (Microphone)

### Capture Strategy

Audio has two modes:

**Passive Monitoring Mode:**
- Low-power voice activity detection (VAD) runs continuously
- When speech is detected, records the utterance
- Transcribes via speech-to-text (STT)
- Classifies as DIRECT or OVERHEARD

**Active Listening Mode (triggered by direct address):**
- Full-quality recording and transcription
- Low latency path to Direct Response Engine
- Remains active until conversation naturally ends (silence > threshold)

### Speech-to-Text Pipeline

```
Microphone (raw audio)
    │
    ▼
┌─────────────────────────┐
│  Voice Activity Detection │  → Is someone speaking?
│  (VAD - Silero/WebRTC)   │     If no → SILENCE, skip processing
└───────────┬─────────────┘
            │ (speech detected)
            ▼
┌─────────────────────────┐
│  Speech-to-Text          │  → Transcribe to text
│  (Whisper / cloud STT)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Address Classification  │  → DIRECT or OVERHEARD?
│  (see rules below)       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Novelty Comparison      │  → Compare transcript embedding to
│                          │     recent audio transcript history
└───────────┬─────────────┘
            │
            ▼
  Output: transcript, audio_mode, audio_novelty
```

### Direct Address vs. Overheard — Classification Rules

This is one of the most important distinctions in the system. The engine must know when it's being spoken TO versus when it's just ambient context.

**Signals indicating DIRECT address:**
- Wake word detected: "Hey Curiosity", "Curiosity", or configured name
- Phrases like "What do you think?" / "Did you know?" when no other person is in context
- Short pause after a declarative statement (user looking for reaction)

**Signals indicating OVERHEARD:**
- Phone call cadence (one-sided conversation with pauses for the other person)
- Conversation between multiple voices (user + someone else)
- No wake word, no direct cues
- Background audio (TV, music, podcast)

**Classification confidence:** When unsure, default to OVERHEARD. False positives (responding when not addressed) are much more annoying than false negatives (missing a direct address). The user can always use the wake word to be explicit.

**Weight adjustment by mode:**
| Mode | Base Weight | Rationale |
|---|---|---|
| DIRECT | 1.0 | Full attention — user wants interaction |
| OVERHEARD | 0.4 | Reduced weight — passive context, not a request |
| SILENCE | 0.0 | No audio input this cycle |

### Audio Novelty

```python
def compute_audio_novelty(current_transcript_embedding, history: list) -> float:
    """
    Compare current transcript to recent audio history.
    Novelty is high when the user is talking about something new.
    """
    if not history:
        return 1.0
    
    similarities = [
        cosine_similarity(current_transcript_embedding, past)
        for past in history
    ]
    max_similarity = max(similarities)
    return 1.0 - max_similarity
```

### Audio History Window
- **Window size:** Last 5 minutes of transcripts (chunked by utterance)
- **Behavior:** Time-based window, not count-based. Older utterances expire.

---

## 3. Text/Context Channel (Screen Awareness)

### Capture Strategy

Text context comes from the user's digital activity — what they're reading, writing, browsing. This is the most "informational" channel.

**Possible sources (platform-dependent):**
| Source | Method | Platform |
|---|---|---|
| Active window title | OS accessibility API | Windows / macOS / Linux |
| Browser URL/tab title | Browser extension or accessibility | Cross-platform |
| Clipboard content | OS clipboard API | Cross-platform |
| Selected text | Accessibility API | Platform-dependent |
| IDE context | Editor plugin (VS Code extension, etc.) | Editor-dependent |

### Text Processing Pipeline

```
OS/Application context
    │
    ▼
┌─────────────────────────┐
│  Context Scraper         │  → Collect: window title, URL, 
│                          │     visible text, clipboard
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Privacy Filter          │  → Exclude: banking, medical,
│  (User-configurable)     │     password managers, etc.
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Text Summarizer         │  → Compress into a concise
│                          │     description of user activity
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Novelty Comparison      │  → Compare to recent text history
└───────────┬─────────────┘
            │
            ▼
  Output: active_context, text_novelty
```

### Privacy by Design

The Text/Context channel is the most privacy-sensitive. Design principles:

1. **All processing is local.** No screen content is sent to cloud APIs without explicit user consent.
2. **Exclusion list:** User can specify apps/sites to never capture (banking, medical, private messages).
3. **No screenshots stored.** Text content is summarized and the summary is stored, not raw screen captures.
4. **Opt-in by channel.** User can disable text context entirely while keeping vision and audio.

### Text Novelty

Same pattern as other channels but with a larger window since context changes less frequently:

```python
def compute_text_novelty(current_context_embedding, history: list) -> float:
    if not history:
        return 1.0
    similarities = [
        cosine_similarity(current_context_embedding, past)
        for past in history
    ]
    max_similarity = max(similarities)
    return 1.0 - max_similarity
```

### Text History Window
- **Window size:** Last 10 context snapshots
- **Behavior:** Rolling window, count-based

---

## 4. Novelty Weighting Engine

### The Core Formula

Each channel produces a weighted input:

```python
def compute_weighted_input(raw_input, base_weight: float, novelty: float) -> float:
    return base_weight * novelty
```

### Channel Base Weights

Default base weights (configurable, will be tuned through testing):

| Channel | Base Weight | Rationale |
|---|---|---|
| Vision | 0.35 | Strong context signal — what's physically happening |
| Audio (DIRECT) | 1.00 | User is talking to the engine — maximum priority |
| Audio (OVERHEARD) | 0.25 | Background context, useful but not primary |
| Text/Context | 0.40 | What the user is intellectually engaged with |

**After novelty weighting example:**

| Channel | Base Weight | Novelty | Effective Weight |
|---|---|---|---|
| Vision (same desk, 5th time) | 0.35 | 0.05 | **0.018** |
| Audio (user on phone, new topic) | 0.25 | 0.80 | **0.200** |
| Text (new article just opened) | 0.40 | 0.90 | **0.360** |

In this example, the text channel dominates because the user just opened something new. The webcam showing the same desk is effectively muted. The overheard phone call contributes moderate context.

### Overall Novelty

```python
def compute_overall_novelty(channels: list[ChannelInput]) -> float:
    """
    Weighted average of all channel novelties.
    Used to decide: should the heartbeat even bother processing this cycle?
    """
    total_weight = sum(ch.effective_weight for ch in channels)
    if total_weight == 0:
        return 0.0
    
    weighted_novelty = sum(ch.novelty * ch.effective_weight for ch in channels)
    return weighted_novelty / total_weight
```

### Novelty Threshold

If `overall_novelty < 0.15`, the heartbeat cycle is **skipped**. Nothing interesting is happening — the engine stays quiet. This prevents the AI from forcing interjections during long stretches of unchanged activity.

---

## 5. Context Snapshot Assembly

All weighted inputs are combined into a single Context Snapshot — the engine's "perception" of this moment.

```python
@dataclass
class ContextSnapshot:
    timestamp: datetime
    heartbeat_id: str           # unique ID for this cycle
    
    # Vision
    image: bytes | None         # raw image if captured
    image_description: str      # LLM-generated description of what's seen
    image_novelty: float        # 0.0 to 1.0
    image_weight: float         # base_weight * novelty
    
    # Audio
    transcript: str             # what was said (if anything)
    audio_mode: str             # "DIRECT" | "OVERHEARD" | "SILENCE"
    audio_novelty: float
    audio_weight: float
    
    # Text/Context
    active_context: str         # summary of user's digital activity
    text_novelty: float
    text_weight: float
    
    # Composite scores
    dominant_channel: str       # which channel has highest effective weight
    overall_novelty: float      # should we proceed with this cycle?
    
    # For the Association Tree
    seed_topics: list[str]      # extracted topics, ordered by channel weight
```

### Seed Topic Extraction

The Context Snapshot's `seed_topics` are the starting points for the Association Tree. They're extracted from the dominant channel(s):

```python
def extract_seed_topics(snapshot: ContextSnapshot) -> list[str]:
    """
    Pull 1-3 seed topics from the highest-weighted channels.
    These become Hop 0 of the association tree.
    """
    # Prompt the LLM with the weighted inputs:
    # "Given what you see, hear, and read, what are the 1-3 most
    #  interesting topics present in this moment?"
    # 
    # Weight the prompt by channel weights so the LLM focuses
    # on the most novel inputs.
    ...
```

---

## 6. Timing & Performance Targets

| Operation | Target Latency | Notes |
|---|---|---|
| Webcam capture | < 100ms | Single frame grab |
| Image embedding | < 500ms | Local model (CLIP) or API |
| Image description (LLM) | < 3s | Only when novelty > 0.3 |
| STT transcription | < 2s | For typical utterance length |
| Address classification | < 500ms | Can be rule-based + light ML |
| Text context scrape | < 200ms | OS API calls |
| Novelty computation | < 100ms | Cosine similarity on embeddings |
| Context assembly | < 100ms | Data structure construction |
| **Total input pipeline** | **< 5s** | **From heartbeat fire to Context Snapshot ready** |

The full creative cycle (input pipeline → association tree → search → scoring → bridge → output) should complete within **30–60 seconds** so the interjection feels timely, not stale.

---

## 7. Configuration Interface

All input pipeline parameters should be user-configurable:

```yaml
# config.yaml (example)
input_pipeline:
  vision:
    enabled: true
    history_window: 10
    base_weight: 0.35
    min_novelty_for_description: 0.3
  
  audio:
    enabled: true
    history_window_minutes: 5
    base_weight_direct: 1.0
    base_weight_overheard: 0.25
    wake_word: "Hey Curiosity"
  
  text:
    enabled: true
    history_window: 10
    base_weight: 0.40
    excluded_apps: ["1Password", "banking*"]
    excluded_urls: ["*.bank.com", "*.medical.*"]
  
  novelty:
    skip_threshold: 0.15    # below this, skip the heartbeat cycle
    decay_rate: 0.95        # how fast repeated inputs lose novelty
```
