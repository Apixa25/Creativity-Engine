"""
Interest Scorer — Evaluates association chains for interestingness.

Five metrics, weighted per spec:
  semantic_distance  × 0.30  — reward boldness (REAL cosine distance with embeddings)
  domain_crossings   × 0.25  — reward crossing fields
  surprise           × 0.20  — reward the unexpected
  bridgeability      × 0.15  — can a story be told?
  novelty            × 0.10  — haven't said this before (embedding similarity with history)

When embeddings are available, semantic_distance uses real cosine distance
between the seed and endpoint in embedding space. The distance is also
weighted by an "efficiency ratio" per Kenett et al. (2014-2018): chains
that reach far in fewer hops score higher — mimicking the flat associative
networks of highly creative people.
"""

from __future__ import annotations

import math

from src.config.llm_adapter import LLMAdapter
from src.config.settings import ScoringConfig
from src.embeddings.provider import EmbeddingProvider, cosine_distance, cosine_similarity
from src.models import AssociationChain, ScoringBreakdown, ContextSnapshot

import numpy as np


SURPRISE_PROMPT = """On a scale of 0.0 to 1.0, how SURPRISING is it to connect "{seed}" to "{endpoint}"?

Be a tough critic. Most connections are only moderately surprising.

0.0 = completely obvious, anyone would connect these
0.2 = mildly interesting but predictable
0.4 = somewhat surprising, a decent lateral jump
0.6 = genuinely surprising, most people wouldn't see this connection
0.8 = very surprising, a creative cross-domain leap
1.0 = extraordinary, a once-in-a-lifetime "aha!" connection

The chain was: {chain_summary}

Be honest and critical. Most associations deserve 0.3-0.6.
Return ONLY a single number, nothing else."""

BRIDGEABILITY_PROMPT = """A creative association chain went from "{seed}" to "{endpoint}" through:
{chain_summary}

The user is currently focused on: "{context}"

Could you tell a compelling 1-sentence story connecting "{endpoint}" back to "{context}" in a way that would make someone say "huh, that's interesting"?

Rate honestly on 0.0 to 1.0:
0.0 = no meaningful connection, pure noise
0.3 = very tenuous, really stretching
0.5 = there's a connection but it requires explanation
0.7 = solid connection, would genuinely interest someone
1.0 = brilliant, illuminating connection

Most connections are 0.3-0.6. Be critical.
Return ONLY a single number, nothing else."""


class InterestScorer:
    def __init__(
        self,
        llm: LLMAdapter,
        config: ScoringConfig | None = None,
        embedder: EmbeddingProvider | None = None,
    ):
        self.llm = llm
        self.cfg = config or ScoringConfig()
        self.embedder = embedder
        self._past_embeddings: list[np.ndarray] = []

    async def score_chain(
        self,
        chain: AssociationChain,
        context: ContextSnapshot,
        past_interjection_topics: list[str] | None = None,
    ) -> ScoringBreakdown:
        """
        Score a single association chain on all five metrics.
        Uses real embeddings when available for semantic_distance and novelty.
        """
        w = self.cfg.weights

        sd = self._compute_semantic_distance(chain)
        dc = self._compute_domain_crossing_score(chain)
        surprise = await self._compute_surprise(chain)
        bridge = await self._compute_bridgeability(chain, context)
        novelty = self._compute_novelty(chain, past_interjection_topics or [])

        total = (
            sd * w.semantic_distance
            + dc * w.domain_crossings
            + surprise * w.surprise
            + bridge * w.bridgeability
            + novelty * w.novelty
        )

        return ScoringBreakdown(
            semantic_distance=sd,
            domain_crossings=dc,
            surprise=surprise,
            bridgeability=bridge,
            novelty=novelty,
            total=total,
        )

    def _compute_semantic_distance(self, chain: AssociationChain) -> float:
        """
        Compute semantic distance between seed and endpoint.

        With embeddings: uses real cosine distance, boosted by an efficiency
        ratio (distance / hops). This rewards chains that reach far in fewer
        hops — the "flat hierarchy" of creative minds per Kenett et al.
        A chain that gets from "silver coins" to "cancer cure" in 4 hops
        scores higher than one that takes 7 hops to get the same distance.

        Without embeddings: falls back to the log-curve heuristic based on
        depth and domain crossings (original behavior).
        """
        root = chain.nodes[0] if chain.nodes else None
        endpoint = chain.nodes[-1] if chain.nodes else None

        if (root and endpoint
                and root.embedding is not None
                and endpoint.embedding is not None):
            raw_distance = cosine_distance(root.embedding, endpoint.embedding)
            hops = max(len(chain.nodes) - 1, 1)
            efficiency = raw_distance / hops
            efficiency_bonus = min(1.0, efficiency * 3.0)
            score = (raw_distance * 0.7) + (efficiency_bonus * 0.3)
            return min(1.0, score)

        depth = len(chain.nodes) - 1
        crossings = chain.domain_crossings
        depth_score = min(1.0, math.log(1 + depth) / math.log(1 + 8))
        crossing_score = min(1.0, math.log(1 + crossings) / math.log(1 + 5))
        return (depth_score * 0.4) + (crossing_score * 0.6)

    def _compute_domain_crossing_score(self, chain: AssociationChain) -> float:
        """Normalized domain crossing score per spec."""
        crossings = chain.domain_crossings
        scoring_map = {0: 0.0, 1: 0.4, 2: 0.7}
        return scoring_map.get(crossings, 1.0)

    async def _compute_surprise(self, chain: AssociationChain) -> float:
        """Ask the LLM how surprising this connection is."""
        prompt = SURPRISE_PROMPT.format(
            seed=chain.seed_topic,
            endpoint=chain.endpoint_topic,
            chain_summary=chain.summary(),
        )
        try:
            return await self.llm.generate_float(prompt)
        except (ValueError, Exception) as e:
            print(f"   ⚠️  Surprise scoring failed: {e}, defaulting to 0.5")
            return 0.5

    async def _compute_bridgeability(self, chain: AssociationChain, context: ContextSnapshot) -> float:
        """Ask the LLM if a compelling bridge can be told."""
        prompt = BRIDGEABILITY_PROMPT.format(
            seed=chain.seed_topic,
            endpoint=chain.endpoint_topic,
            chain_summary=chain.summary(),
            context=context.seed_topic or chain.seed_topic,
        )
        try:
            return await self.llm.generate_float(prompt)
        except (ValueError, Exception) as e:
            print(f"   ⚠️  Bridgeability scoring failed: {e}, defaulting to 0.5")
            return 0.5

    def _compute_novelty(self, chain: AssociationChain, past_topics: list[str]) -> float:
        """
        Check how novel this endpoint is compared to past interjections.

        With embeddings: computes cosine similarity against all past endpoint
        embeddings. If any past topic is >0.85 similar, it's a near-duplicate.
        Continuous scoring means "kinda similar" topics still get partial credit.

        Without embeddings: falls back to substring matching (original behavior).
        """
        endpoint = chain.nodes[-1] if chain.nodes else None

        if endpoint and endpoint.embedding is not None and self._past_embeddings:
            max_sim = 0.0
            for past_emb in self._past_embeddings:
                sim = cosine_similarity(endpoint.embedding, past_emb)
                max_sim = max(max_sim, sim)
            return max(0.0, 1.0 - max_sim)

        if not past_topics:
            return 1.0

        endpoint_lower = chain.endpoint_topic.lower()
        for past in past_topics:
            if past.lower() in endpoint_lower or endpoint_lower in past.lower():
                return 0.1
        return 1.0

    def record_interjection(self, chain: AssociationChain) -> None:
        """Record a chain's endpoint embedding for future novelty checks."""
        endpoint = chain.nodes[-1] if chain.nodes else None
        if endpoint and endpoint.embedding is not None:
            self._past_embeddings.append(endpoint.embedding)

    async def rank_chains(
        self,
        chains: list[AssociationChain],
        context: ContextSnapshot,
        past_interjection_topics: list[str] | None = None,
        max_to_score: int = 5,
    ) -> list[tuple[AssociationChain, ScoringBreakdown]]:
        """
        Pre-filter chains using cheap metrics (no LLM calls), then only
        full-score the top candidates. This keeps LLM calls bounded.
        With embeddings, pre-filtering uses real semantic distance.
        """
        past = past_interjection_topics or []

        pre_scored = []
        for chain in chains:
            sd = self._compute_semantic_distance(chain)
            dc = self._compute_domain_crossing_score(chain)
            novelty = self._compute_novelty(chain, past)
            cheap_score = sd * 0.4 + dc * 0.35 + novelty * 0.25
            pre_scored.append((chain, cheap_score))

        pre_scored.sort(key=lambda x: x[1], reverse=True)
        top_chains = [c for c, _ in pre_scored[:max_to_score]]

        print(f"   📋 Pre-filtered {len(chains)} chains → top {len(top_chains)} for full scoring")

        results = []
        for chain in top_chains:
            score = await self.score_chain(chain, context, past)
            chain.interest_score = score.total
            results.append((chain, score))

        results.sort(key=lambda x: x[1].total, reverse=True)
        return results
