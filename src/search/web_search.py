"""
Web Search Module — Grounds creative associations in real-world facts.

Takes the top association chain endpoints and searches the web for
surprising facts, statistics, and stories that make interjections
concrete and specific rather than vague LLM-generated generalities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.config.llm_adapter import LLMAdapter


@dataclass
class SearchResult:
    query: str
    facts: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    summary: str = ""
    raw_results: list[dict] = field(default_factory=list, repr=False)


QUERY_SYSTEM = (
    "You construct highly specific web search queries. Your queries find "
    "surprising facts, statistics, and real-world stories — not generic overviews. "
    "Think like a curious person at 2 AM falling down a Wikipedia rabbit hole."
)

QUERY_TEMPLATE = """Given this creative association endpoint: "{endpoint}"
Which arrived through this chain: {chain_summary}

The user's current context: "{context}"

Construct a web search query that would find:
- Surprising facts or statistics about "{endpoint}"
- Real-world examples or stories
- Scientific research or specific numbers

The query should be SPECIFIC. Not "3D printed food" but "3D printed food NASA astronaut nutrition research statistics".

Return ONLY the search query string, nothing else."""

EXTRACT_TEMPLATE = """Here are web search results for the query: "{query}"

Results:
{results_text}

Extract the 2-3 most SURPRISING and SPECIFIC facts from these results.
Each fact should include a concrete detail — a number, a name, a date, or a specific example.

Skip anything vague, generic, or that sounds like a Wikipedia intro paragraph.
Prefer facts that would make someone say "wait, really?"

Return ONLY a JSON array of strings, each being one fact:
["fact 1", "fact 2", "fact 3"]"""


class WebSearcher:
    def __init__(self, llm: LLMAdapter, api_key_env: str = "TAVILY_API_KEY"):
        self.llm = llm
        self.api_key = os.environ.get(api_key_env, "")
        self._client = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def _get_client(self):
        if self._client is None:
            from tavily import AsyncTavilyClient
            self._client = AsyncTavilyClient(api_key=self.api_key)
        return self._client

    async def search_for_chain(
        self,
        endpoint_topic: str,
        chain_summary: str,
        context: str,
    ) -> SearchResult:
        """
        Full pipeline: construct query → search web → extract facts.
        Falls back to LLM knowledge if Tavily is unavailable.
        """
        query = await self._construct_query(endpoint_topic, chain_summary, context)
        print(f"   🔍 Searching: \"{query}\"")

        if self.is_available:
            raw_results = await self._tavily_search(query)
        else:
            return await self._fallback_search(endpoint_topic, query)

        if not raw_results:
            return SearchResult(query=query, summary="No results found.")

        results_text = self._format_results(raw_results)
        facts = await self._extract_facts(query, results_text)
        urls = [r.get("url", "") for r in raw_results if r.get("url")]

        summary = " | ".join(facts) if facts else "No specific facts extracted."

        return SearchResult(
            query=query,
            facts=facts,
            source_urls=urls[:5],
            summary=summary,
            raw_results=raw_results,
        )

    async def _construct_query(self, endpoint: str, chain_summary: str, context: str) -> str:
        prompt = QUERY_TEMPLATE.format(
            endpoint=endpoint,
            chain_summary=chain_summary,
            context=context,
        )
        resp = await self.llm.generate(prompt, system=QUERY_SYSTEM, temperature=0.5)
        return resp.text.strip().strip('"').strip("'")

    async def _tavily_search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            client = await self._get_client()
            response = await client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
            )
            return response.get("results", [])
        except Exception as e:
            print(f"   ⚠️  Tavily search failed: {e}")
            return []

    def _format_results(self, results: list[dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            content = r.get("content", "")[:500]
            url = r.get("url", "")
            parts.append(f"[{i}] {title}\n    URL: {url}\n    {content}")
        return "\n\n".join(parts)

    async def _extract_facts(self, query: str, results_text: str) -> list[str]:
        prompt = EXTRACT_TEMPLATE.format(query=query, results_text=results_text)
        try:
            raw = await self.llm.generate_json(prompt, temperature=0.3)
            if isinstance(raw, list):
                return [str(f) for f in raw[:3]]
        except Exception as e:
            print(f"   ⚠️  Fact extraction failed: {e}")
        return []

    async def _fallback_search(self, endpoint: str, query: str) -> SearchResult:
        """When no search API is available, ask the LLM for its best knowledge."""
        print(f"   ℹ️  No Tavily key — using LLM knowledge as fallback")
        prompt = (
            f'What are 2-3 surprising, specific facts about "{endpoint}"? '
            f"Include real numbers, names, or dates. "
            f"Return ONLY a JSON array of fact strings."
        )
        try:
            raw = await self.llm.generate_json(prompt, temperature=0.5)
            facts = [str(f) for f in raw[:3]] if isinstance(raw, list) else []
        except Exception:
            facts = []

        return SearchResult(
            query=query,
            facts=facts,
            summary=" | ".join(facts) if facts else "LLM fallback — no web results.",
        )
