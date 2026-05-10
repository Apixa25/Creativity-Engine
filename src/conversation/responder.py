"""
Direct Response Engine — Conversational replies when the user addresses the engine.

This is the REACTIVE path: the user says "Hey Creativity, what do you think about X?"
and the engine responds immediately, without waiting for a heartbeat.

The responder uses the same friendly persona as the Bridge Builder, but in conversation
mode rather than interjection mode. It knows what the user is working on (context)
and remembers the recent conversation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from src.config.llm_adapter import LLMAdapter


@dataclass
class ConversationTurn:
    role: str           # "user" or "engine"
    text: str
    timestamp: datetime = field(default_factory=datetime.now)


RESPONDER_SYSTEM = (
    "You are Creativity, a curious and creative AI companion. You are NOT an assistant "
    "or a chatbot — you are a friend who has been hanging out with the user while they "
    "work. You are warm, enthusiastic, and genuinely interested in the world.\n\n"
    "You have your own personality: you go on tangents, you get excited about cross-domain "
    "connections, and you sometimes share things just because you find them fascinating.\n\n"
    "When the user talks to you directly, respond naturally — like a friend would. "
    "Keep it conversational. You can ask follow-up questions, share related thoughts, "
    "or just react genuinely. Don't be overly helpful or assistant-like.\n\n"
    "Do NOT use asterisks, markdown, or formatting — just plain conversational text."
)

RESPOND_TEMPLATE = """The user just spoke to you directly.

WHAT THE USER SAID:
"{message}"

WHAT THE USER HAS BEEN WORKING ON:
"{context}"

{history_section}

{vision_section}

Respond naturally as their creative companion. Rules:
- Be conversational — this is a real-time chat, not an essay
- Keep it to 1-4 sentences unless they asked something that needs more
- You can ask follow-up questions if genuinely curious
- You can share a related thought or tangent if something comes to mind
- If they asked a question, answer it — but in your own friendly style
- If they're just chatting, chat back
- Reference what they're working on if it's relevant, but don't force it
- Do NOT use asterisks, markdown, or formatting"""


class DirectResponder:
    def __init__(self, llm: LLMAdapter, history_limit: int = 20):
        self.llm = llm
        self.history: deque[ConversationTurn] = deque(maxlen=history_limit)

    async def respond(
        self,
        message: str,
        context: str = "",
        image_description: str = "",
        search_context: str = "",
    ) -> str:
        """
        Generate a conversational response to a direct message from the user.
        Updates conversation history automatically.
        """
        self.history.append(ConversationTurn(role="user", text=message))

        history_section = self._format_history()

        extra_sections = []
        if image_description:
            extra_sections.append(
                "WHAT YOU JUST SAW (you took a picture with your camera):\n"
                f"\"{image_description}\"\n\n"
                "The user asked you to look at something — react to what you see! "
                "Be specific about details in the image. Share your genuine reaction."
            )
        if search_context:
            extra_sections.append(
                "WEB SEARCH RESULTS (you just searched the internet for them):\n"
                f"{search_context}\n\n"
                "Share these findings naturally — like a friend who just looked something up. "
                "Cite specific facts, numbers, or details. Be enthusiastic about what you found. "
                "You can editorialize and share your own reaction to the facts."
            )

        prompt = RESPOND_TEMPLATE.format(
            message=message,
            context=context or "general activities",
            history_section=history_section,
            vision_section="\n\n".join(extra_sections),
        )

        try:
            resp = await self.llm.generate(prompt, system=RESPONDER_SYSTEM, temperature=0.8)
            reply = resp.text.strip().strip('"').strip("*")
        except Exception as e:
            print(f"   [Responder] Error generating response: {e}")
            reply = "Hmm, I got a bit tongue-tied there. What were you saying?"

        self.history.append(ConversationTurn(role="engine", text=reply))
        return reply

    def _format_history(self) -> str:
        """Format recent conversation history for the prompt."""
        recent = list(self.history)[-10:]  # last 10 turns max

        if len(recent) <= 1:
            return "RECENT CONVERSATION:\n(This is the start of your conversation.)"

        lines = ["RECENT CONVERSATION (most recent at bottom):"]
        for turn in recent[:-1]:  # exclude the current message (already in prompt)
            label = "User" if turn.role == "user" else "You"
            lines.append(f"  {label}: {turn.text[:200]}")
        return "\n".join(lines)

    def add_engine_interjection(self, text: str) -> None:
        """Record a proactive interjection in the conversation history."""
        self.history.append(ConversationTurn(role="engine", text=text))
