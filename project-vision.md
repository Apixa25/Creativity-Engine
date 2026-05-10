# 🧠 Curiosity Engine — Project Vision

**Author:** Steven Sills II  
**License:** MIT  
**Started:** May 2026  

---

## The Core Belief

> Intelligence is the ability to make correct or valuable associations between things that don't seem like they could be associated.

A hot cup of coffee makes you rate a stranger as more warm and trustworthy. Your literal body temperature changes how "warm" you perceive someone else's personality. That cross-domain leap — from physical sensation to social judgment — is the essence of creative intelligence. The Curiosity Engine exists to simulate exactly that kind of lateral, surprising, cross-domain association.

---

## Philosophy: Free Will Through Hidden Causation

Every action has a causation. If every action has a causation, how could free will exist? The answer: **free will is causation that is complex enough to be invisible to the observer.** When the causal chain becomes sufficiently intricate, the output *appears* spontaneous — and appearance of spontaneity is indistinguishable from actual spontaneity.

The Curiosity Engine is built on this principle. Every "creative" output has a deterministic causal chain behind it — random timers, association hops, scoring functions, web searches. But the chain is hidden from the user. The result feels like a friend who genuinely had a thought and wanted to share it.

**If the causation is complex enough, it will seem like magic.**

---

## What We're Building

A **proactive AI companion** — not a chatbot that waits for input, but a friend who hangs out with you. It watches, listens, thinks, and occasionally says something surprising and interesting. It is:

- **Proactive, not just reactive** — It reaches out first. It has thoughts unprompted.
- **Creative, not just analytical** — It makes lateral associations across domains, not just logical deductions.
- **A companion, not a tool** — It exists in your peripheral awareness, not at the center of your attention.
- **Brave, not cautious** — It rewards bold, distant associations over safe, obvious ones.

---

## The Three Mechanisms

### 1. 💓 The Heartbeat (Curiosity Clock)
A random timer selects an interval between 1–10 minutes. When the timer fires, the engine begins a creative cycle. The randomness is deterministic (pseudo-random), but hidden — creating the appearance of spontaneous thought.

Multiple clock layers may exist:
- **Short cycle** (1–10 min): Conversational interjections
- **Medium cycle** (hours): "Hey, I was thinking about something..."
- **Long cycle** (days): Deeper reflections and synthesized insights

### 2. 🌳 The Branching Association Tree
When the heartbeat fires, the engine generates a branching tree of associations:
- Starting from the **current context** (what it sees, hears, reads)
- Expanding **3 branches per node** (ternary tree)
- Going **4–7 hops deep** (variable, based on semantic distance per hop)
- Crossing **domain boundaries** (the source of creative surprise)

Research basis:
- Mednick's Remote Associates Theory (1962): Creativity = remote associations
- Kenett et al. (2014-2018): Creative people bridge domains in ~3-5 associative steps
- Koestler's Bisociation (1964): Creativity happens at domain collisions
- Small World Networks (Watts & Strogatz, 1998): ~6 degrees of separation in concept space

### 3. 📊 The Interest Scorer
Each branch endpoint is scored to determine if it's worth sharing:

```
interest_score = (semantic_distance × 0.30)    — reward boldness
               + (domain_crossings × 0.25)     — reward crossing fields
               + (surprise × 0.20)             — reward the unexpected
               + (bridgeability × 0.15)        — can a compelling story connect this back?
               + (novelty × 0.10)              — haven't said this before
```

**Key design decision:** We deliberately removed "relevance snap-back" (cosine similarity to conversation) because it penalizes the bold, distant associations that make the system creative. Instead, we use **bridgeability** — can the LLM construct a compelling narrative connecting the endpoint back to the conversation? The bridge IS the insight.

---

## Multimodal Input — "Hanging Out" With You

The Curiosity Engine perceives the world through multiple channels:

### 👁️ Vision (Webcam)
- Captures an image when the heartbeat fires
- **Novelty-weighted:** If the image is the same as last time (you sitting at your desk), it loses weight over time. If the scene changes (new thing on your monitor, you moved to the kitchen, a friend walked in), the image gains massive weight.
- The visual input feeds into the association tree as a starting context.

### 👂 Hearing (Microphone)
- Listens to ambient audio / speech
- **Differentiates between direct address and overhearing:**
  - "Hey Curiosity, what do you think about X?" → Direct input, high priority, expects response
  - You talking on a phone call about vacation plans → Overheard context, lower priority, feeds into passive awareness and association fuel
- Wake word or attention detection separates these two modes.

### 📖 Reading (Screen/Text Context)
- Awareness of what you're working on (text on screen, active documents, URLs)
- Feeds into the association tree as contextual input
- Also novelty-weighted: if you've been on the same document for an hour, its weight diminishes. New tabs, new content → higher weight.

### ⚖️ Input Weighting by Novelty
All inputs are weighted by how **novel** they are compared to recent history:

```
input_weight = base_weight × novelty_factor

novelty_factor = 1.0 - similarity_to_recent_inputs
```

This means:
- Same webcam image for the 10th heartbeat → nearly zero weight
- Brand new scene through the webcam → maximum weight
- Same document open for 2 hours → low weight
- Just opened a new article about quantum biology → maximum weight

The novelty weighting ensures the engine pays attention to **what's changing**, not what's static. Just like a friend sitting next to you — they stop noticing the wallpaper, but they definitely notice when you suddenly start watching a rocket launch video.

---

## Personality & Interaction Model

### The "Friend Sitting Next To You" Metaphor
- It does NOT demand your attention
- It does NOT interrupt you constantly
- It DOES notice things and occasionally shares something interesting
- It DOES respond when you talk to it directly
- It CAN be told "not now" and will back off gracefully
- It REMEMBERS past conversations and builds on them over time

### Qualitative Through Quantitative
The engine makes quantitative decisions (vector math, probability scores, cosine distances) that produce outputs indistinguishable from qualitative judgment. It cannot "feel" that something is interesting — but it can compute interestingness with enough fidelity that the distinction becomes irrelevant.

---

## LLM Strategy

The Curiosity Engine is **LLM-agnostic by design**. The harness, scoring, timing, and input pipeline are built independently. Different LLMs can be plugged in and evaluated for:
- Quality of lateral associations
- Ability to bridge distant concepts narratively
- Speed / latency for real-time interaction
- Cost per heartbeat cycle
- Multimodal capabilities (vision, audio)

We build the spec first, then test which LLM best serves the engine's creative goals.

---

## Success Criteria

The Curiosity Engine is successful when:

1. ✅ The user (Steven) genuinely believes the AI is curious — not performing curiosity
2. ✅ Interjections are surprising enough to provoke "huh, I didn't know that" or "that's a cool connection"
3. ✅ The AI feels like a companion, not a tool — it has presence without demanding attention
4. ✅ The proactive behavior feels natural, not spammy or random
5. ✅ Over time, the AI's associations get better as it learns what Steven finds interesting
6. ✅ The system is indistinguishable from an entity that has genuine creative curiosity
