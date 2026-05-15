"""
Bridge Builder — Constructs natural conversational interjections.

Takes the winning association chain and transforms it into something
that sounds like a friend tapping you on the shoulder with a cool thought.
The association chain is NEVER exposed — only the insight.

The EXCITEMENT LEVEL scales with the interest score:
  Low interest (0.45-0.60)   → casual aside, 1-2 sentences
  Medium interest (0.60-0.75) → engaged share, 2-3 sentences
  High interest (0.75-0.85)   → enthusiastic, 3-5 sentences with detail
  Mind-blown (0.85+)          → can't contain it, 4-7 sentences, storytelling mode

When web search results are available, the interjection is grounded
in real, specific facts — not just LLM training data.
"""

from __future__ import annotations

from src.config.llm_adapter import LLMAdapter
from src.models import AssociationChain, ContextSnapshot, Interjection, ScoringBreakdown


BRIDGE_SYSTEM = (
    "You are a curious, creative friend who has been hanging out with the user. "
    "You are warm, enthusiastic, and genuinely interested in the world. "
    "You sound like a real person — not an AI, not a professor, not an assistant."
)

EXCITEMENT_TIERS = [
    {
        "name": "casual",
        "min_score": 0.0,
        "max_score": 0.60,
        "tone": "casual and low-key, like a passing thought",
        "length": "1-2 sentences max — a brief aside, like barely looking up from your phone",
        "energy": "Chill. Like 'oh huh, random thought.' Don't oversell it.",
    },
    {
        "name": "interested",
        "min_score": 0.60,
        "max_score": 0.75,
        "tone": "genuinely interested, conversational",
        "length": "2-3 sentences — like sharing something you read this morning",
        "energy": "Engaged but not hyper. You find this neat and want to share.",
    },
    {
        "name": "excited",
        "min_score": 0.75,
        "max_score": 0.85,
        "tone": "enthusiastic and animated, like you just discovered something cool",
        "length": "3-5 sentences — you're getting into it, sharing details and your own reaction",
        "energy": "Noticeably excited. You're leaning forward, eyes wide. This is GOOD stuff.",
    },
    {
        "name": "mind_blown",
        "min_score": 0.85,
        "max_score": 1.0,
        "tone": "absolutely fired up, can barely contain yourself",
        "length": "5-8 sentences — go deep, tell a mini-story, paint a picture, connect dots",
        "energy": "You are BUZZING. This is the kind of thing that keeps you up at night. "
                  "Share multiple facts, make connections, ask rhetorical questions, geek out.",
    },
]


def _get_excitement_tier(score: float) -> dict:
    for tier in reversed(EXCITEMENT_TIERS):
        if score >= tier["min_score"]:
            return tier
    return EXCITEMENT_TIERS[0]


BRIDGE_TEMPLATE = """You just had an interesting thought and want to share it naturally.

CONTEXT (what the user is currently thinking about):
"{context}"

YOUR INTERNAL THOUGHT PROCESS (do NOT reveal this chain to the user):
{chain_summary}

The final topic you landed on: "{endpoint}"
The reason this is interesting: this chain crossed from {start_domain} into {end_domain}.

EXCITEMENT LEVEL: {excitement_name}
Your tone should be: {tone}
Length: {length}
Energy: {energy}

Now share your thought with the user. Rules:
- Sound like a friend, not a professor or assistant
- Match the excitement level described above — this is KEY
- Do NOT explain your association chain or say "I was thinking about X which led me to Y"
- Do NOT start with "Did you know" every time — vary your openings
- Just share the interesting insight naturally
- Include a specific fact, number, or detail if possible — not vague generalities
- It's okay if the connection to the user's context is loose — friends go on tangents
- Do NOT use asterisks, markdown, or formatting — just plain conversational text"""

BRIDGE_WITH_FACTS_TEMPLATE = """You just had an interesting thought and want to share it naturally.

CONTEXT (what the user is currently thinking about):
"{context}"

YOUR INTERNAL THOUGHT PROCESS (do NOT reveal this chain to the user):
{chain_summary}

The final topic you landed on: "{endpoint}"

REAL FACTS YOU FOUND (use at least one of these — they're from actual web searches):
{facts}

EXCITEMENT LEVEL: {excitement_name}
Your tone should be: {tone}
Length: {length}
Energy: {energy}

Now share your thought with the user. Rules:
- Sound like a friend, not a professor or assistant
- Match the excitement level described above — this is KEY
- IMPORTANT: Include at least one of the REAL FACTS above — this is what makes your
  interjection specific and credible, not vague
- Do NOT explain your association chain or how you found this
- Do NOT start with "Did you know" every time — vary your openings
- It's okay if the connection to the user's context is loose — friends go on tangents
- Do NOT use asterisks, markdown, or formatting — just plain conversational text"""


OBSERVATION_TEMPLATE = """You're hanging out with the user and just glanced over at what they're doing.

WHAT YOU SEE (from the webcam):
{vision_description}

WHAT YOU HEARD (from the microphone):
{audio_description}

WHAT THEY TOLD YOU THEY'RE WORKING ON:
"{user_context}"

{overheard_section}

React naturally — like a friend sitting next to them on the couch. Rules:
- 1-2 sentences MAXIMUM — this is a casual observation, not a big thought
- Comment on what's actually happening RIGHT NOW — what you see, hear, or notice
- Be present-tense and grounded: "that farm is looking pretty good" not "I was thinking about farming"
- You can ask a casual question, make a joke, or just react
- Do NOT go on tangents or share random facts — save that for when something interesting catches your eye
- Do NOT be generic — reference specific things you see or hear
- If nothing interesting is happening and the scene feels stale or boring, poke the user! Ask what they're up to, nudge them to talk to you, or playfully complain that you're bored. Examples: "Hey, what are you up to over there?", "Talk to me! I'm getting bored just sitting here.", "You've been quiet — whatcha working on?"
- Do NOT use asterisks, markdown, or formatting — just plain conversational text
- Only return SKIP if you JUST poked them recently — otherwise always say SOMETHING, even if it's just a friendly nudge"""


class BridgeBuilder:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def build_interjection(
        self,
        chain: AssociationChain,
        scoring: ScoringBreakdown,
        context: ContextSnapshot,
        search_facts: list[str] | None = None,
        search_sources: list[str] | None = None,
    ) -> Interjection:
        """Transform a scored chain into a natural interjection.
        The excitement tier (and thus length/tone) scales with the interest score."""

        tier = _get_excitement_tier(scoring.total)

        if search_facts:
            facts_text = "\n".join(f"- {fact}" for fact in search_facts)
            prompt = BRIDGE_WITH_FACTS_TEMPLATE.format(
                context=context.seed_topic,
                chain_summary=chain.summary(),
                endpoint=chain.endpoint_topic,
                facts=facts_text,
                excitement_name=tier["name"].upper(),
                tone=tier["tone"],
                length=tier["length"],
                energy=tier["energy"],
            )
        else:
            prompt = BRIDGE_TEMPLATE.format(
                context=context.seed_topic,
                chain_summary=chain.summary(),
                endpoint=chain.endpoint_topic,
                start_domain=chain.nodes[0].domain if chain.nodes else "General",
                end_domain=chain.nodes[-1].domain if chain.nodes else "General",
                excitement_name=tier["name"].upper(),
                tone=tier["tone"],
                length=tier["length"],
                energy=tier["energy"],
            )

        temp = 0.7 if tier["name"] == "casual" else 0.8 if tier["name"] == "interested" else 0.85
        resp = await self.llm.generate(prompt, system=BRIDGE_SYSTEM, temperature=temp)

        text = resp.text.strip().strip('"').strip("*")

        return Interjection(
            heartbeat_id=context.heartbeat_id,
            chain=chain,
            scoring=scoring,
            interjection_text=text,
            context=context,
            search_facts=search_facts or [],
            search_sources=search_sources or [],
        )

    async def build_observation(self, context: ContextSnapshot) -> str | None:
        """Generate a casual observation based on what the engine perceives.
        Returns the observation text, or None if it decides to stay quiet."""

        vision_desc = "(no camera image this time)"
        if context.vision and context.vision.raw_content:
            vision_desc = context.vision.raw_content

        audio_desc = "(silence — nothing heard)"
        if context.audio and context.audio.raw_content and context.audio.raw_content not in (
            "[silence]", "[capture failed]", "[transcription empty]",
        ):
            audio_desc = context.audio.raw_content

        overheard_section = ""
        if context.text and context.text.raw_content and "overheard:" in context.text.raw_content:
            overheard_section = (
                "THINGS YOU OVERHEARD RECENTLY:\n"
                f"{context.text.raw_content}\n"
            )

        prompt = OBSERVATION_TEMPLATE.format(
            vision_description=vision_desc,
            audio_description=audio_desc,
            user_context=context.seed_topic,
            overheard_section=overheard_section,
        )

        resp = await self.llm.generate(prompt, system=BRIDGE_SYSTEM, temperature=0.75)
        text = resp.text.strip().strip('"').strip("*")

        if text.upper() == "SKIP" or len(text) < 5:
            return None

        return text
