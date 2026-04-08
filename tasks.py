from dataclasses import dataclass
from typing import List


@dataclass
class TaskConfig:
    name: str
    n_documents: int
    doc_types: List[str]
    top_k: int
    token_budget: int
    n_queries: int
    patterns: List[str]
    max_steps: int = 1


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        n_documents=50,
        doc_types=["faq"],
        top_k=4,
        token_budget=200,    # raised: comfortably fits 2–3 chunks so selection matters
        n_queries=15,
        patterns=["redundancy"],
        max_steps=2,
    ),
    "medium": TaskConfig(
        name="medium",
        n_documents=200,
        doc_types=["policy", "technical", "news"],
        top_k=6,
        token_budget=250,    # raised: 2–3 chunk selection is viable, not forced to pick 1
        n_queries=15,
        patterns=["redundancy", "noise", "contradictions", "stale_docs", "adversarial"],
        max_steps=3,
    ),
    "hard": TaskConfig(
        name="hard",
        n_documents=500,
        doc_types=["policy", "technical", "news", "multihop", "medical"],
        top_k=8,
        token_budget=200,    # raised: tight but 2-chunk multi-hop is feasible
        n_queries=20,
        patterns=["redundancy", "noise", "contradictions", "multihop", "stale_docs", "adversarial"],
        max_steps=4,
    ),
}
