"""
Bridge Builder — Constructs natural conversational interjections.

Takes the winning association chain and transforms it into something
that sounds like a friend tapping you on the shoulder with a cool thought.
The association chain is NEVER exposed — only the insight.

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

BRIDGE_TEMPLATE = """You just had an interesting thought and want to share it naturally.

CONTEXT (what the user is currently thinking about):
"{context}"

YOUR INTERNAL THOUGHT PROCESS (do NOT reveal this chain to the user):
{chain_summary}

The final topic you landed on: "{endpoint}"
The reason this is interesting: this chain crossed from {start_domain} into {end_domain}.

Now share your thought with the user. Rules:
- Sound like a friend, not a professor or assistant
- Be genuinely enthusiastic but not over-the-top
- Keep it to 2-3 sentences max — like tapping someone on the shoulder
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

Now share your thought with the user. Rules:
- Sound like a friend, not a professor or assistant
- Be genuinely enthusiastic but not over-the-top
- Keep it to 2-3 sentences max — like tapping someone on the shoulder
- IMPORTANT: Include at least one of the REAL FACTS above — this is what makes your
  interjection specific and credible, not vague
- Do NOT explain your association chain or how you found this
- Do NOT start with "Did you know" every time — vary your openings
- It's okay if the connection to the user's context is loose — friends go on tangents
- Do NOT use asterisks, markdown, or formatting — just plain conversational text"""


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
        """Transform a scored chain into a natural interjection."""

        if search_facts:
            facts_text = "\n".join(f"- {fact}" for fact in search_facts)
            prompt = BRIDGE_WITH_FACTS_TEMPLATE.format(
                context=context.seed_topic,
                chain_summary=chain.summary(),
                endpoint=chain.endpoint_topic,
                facts=facts_text,
            )
        else:
            prompt = BRIDGE_TEMPLATE.format(
                context=context.seed_topic,
                chain_summary=chain.summary(),
                endpoint=chain.endpoint_topic,
                start_domain=chain.nodes[0].domain if chain.nodes else "General",
                end_domain=chain.nodes[-1].domain if chain.nodes else "General",
            )

        resp = await self.llm.generate(prompt, system=BRIDGE_SYSTEM, temperature=0.8)

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
