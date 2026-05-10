"""
Configuration loader for the Creativity Engine.

Reads from config.yaml (user's copy) or falls back to config.example.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class HeartbeatConfig:
    min_minutes: int = 1
    max_minutes: int = 10


@dataclass
class ScoringWeights:
    semantic_distance: float = 0.30
    domain_crossings: float = 0.25
    surprise: float = 0.20
    bridgeability: float = 0.15
    novelty: float = 0.10


@dataclass
class AssociationTreeConfig:
    branching_factor: int = 3
    min_depth: int = 4
    max_depth: int = 7
    min_domain_crossings: int = 1
    early_stop_distance: float = 0.8
    keep_per_level: int = 3


@dataclass
class ScoringConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    fire_threshold: float = 0.45
    incubation_threshold: float = 0.30


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


@dataclass
class EngineConfig:
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    association_tree: AssociationTreeConfig = field(default_factory=AssociationTreeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_config(config_path: str | Path | None = None) -> EngineConfig:
    """
    Load configuration from YAML file.
    Priority: explicit path → config.yaml → config.example.yaml → defaults
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    if config_path is not None:
        path = Path(config_path)
    elif (project_root / "config.yaml").exists():
        path = project_root / "config.yaml"
    elif (project_root / "config.example.yaml").exists():
        path = project_root / "config.example.yaml"
    else:
        return EngineConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    hb = raw.get("heartbeat", {})
    tree = raw.get("association_tree", {})
    sc = raw.get("scoring", {})
    sw = sc.get("weights", {})
    llm_raw = raw.get("llm", {})

    return EngineConfig(
        heartbeat=HeartbeatConfig(
            min_minutes=hb.get("min_minutes", 1),
            max_minutes=hb.get("max_minutes", 10),
        ),
        association_tree=AssociationTreeConfig(
            branching_factor=tree.get("branching_factor", 3),
            min_depth=tree.get("min_depth", 4),
            max_depth=tree.get("max_depth", 7),
            min_domain_crossings=tree.get("min_domain_crossings", 1),
            early_stop_distance=tree.get("early_stop_distance", 0.8),
        ),
        scoring=ScoringConfig(
            weights=ScoringWeights(
                semantic_distance=sw.get("semantic_distance", 0.30),
                domain_crossings=sw.get("domain_crossings", 0.25),
                surprise=sw.get("surprise", 0.20),
                bridgeability=sw.get("bridgeability", 0.15),
                novelty=sw.get("novelty", 0.10),
            ),
            fire_threshold=sc.get("fire_threshold", 0.65),
            incubation_threshold=sc.get("incubation_threshold", 0.40),
        ),
        llm=LLMConfig(
            provider=llm_raw.get("provider") or "openai",
            model=llm_raw.get("model") or "gpt-4o-mini",
            api_key_env=llm_raw.get("api_key_env") or "OPENAI_API_KEY",
            base_url=llm_raw.get("base_url"),
        ),
    )
