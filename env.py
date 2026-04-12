"""
Core RL environment for RAG chunk selection.
"""

import random
from typing import List, Optional, Tuple

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from corpus_generator import load_or_generate_corpus
from inference import generate_answer, judge_answer, _semantic_similarity
from task_configs import TaskConfig

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Public Pydantic models (what the agent sees — no internal labels)
# ---------------------------------------------------------------------------


class ChunkInfo(BaseModel):
    chunk_id: int
    content: str
    similarity_score: float
    token_count: int
    source: str
    doc_type: str
    is_current: bool   # document-level staleness flag (like a publish-date signal)


class Observation(BaseModel):
    query: str
    retrieved_chunks: List[ChunkInfo]
    cross_similarity_matrix: List[List[float]]
    token_budget_remaining: int
    episode_step: int
    steps_remaining: int
    task: str
    current_selection: List[int] = Field(default_factory=list)
    selection_token_count: int = 0
    candidate_token_costs: List[int] = Field(default_factory=list)
    last_answer_quality: Optional[float] = None   # reward signal from prior step; None on first step


class Action(BaseModel):
    selected_chunk_indices: List[int]


class Reward(BaseModel):
    answer_quality: float        # 0.0–1.0, LLM judge score
    token_cost: float            # ≤ 0, proportional to tokens / total budget
    evidence_precision: float    # fraction of selected chunks that are supportive
    evidence_recall: float       # fraction of required fact-groups covered
    diversity_bonus: float       # reward for non-redundant selection
    minimality_bonus: float      # bonus for covering all facts with fewest chunks
    contradiction_penalty: float # penalty per contradictory / stale / adversarial chunk
    budget_penalty: float        # penalty if selection exhausts remaining budget exactly
    shaping_bonus: float         # high-relevance bonus minus near-duplicate penalty
    total: float                 # clamped to [0.0, 1.0]


# ---------------------------------------------------------------------------
# Embedding model (loaded once per process)
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RagRLEnvironment:
    def __init__(self, task: TaskConfig, corpus_dir: str = "corpora"):
        self.task = task
        self.corpus_dir = corpus_dir
        self.embedder = _get_embedding_model()

        self.corpus = load_or_generate_corpus(task, corpus_dir)
        self.queries = self.corpus["queries"]

        self._chroma_client: Optional[chromadb.EphemeralClient] = None
        self._collection = None
        self._doc_embeddings: Optional[np.ndarray] = None
        self._build_index()

        # Episode state
        self._query_order: List[int] = list(range(len(self.queries)))
        self._episode_index: int = 0
        self.current_query: Optional[dict] = None
        self.retrieved_chunks: List[ChunkInfo] = []
        self._retrieved_embeddings: Optional[np.ndarray] = None

        # Internal metadata — NOT exposed to the agent
        # Each entry: {support_type, fact_group, pattern}
        self._chunk_internal_meta: List[dict] = []

        self.token_budget_remaining: int = task.token_budget
        self.episode_step: int = 0
        self.last_selected_indices: List[int] = []
        self.current_selection: List[int] = []
        self._last_answer_quality: Optional[float] = None   # fed back in next obs
        self._done: bool = False
        self._rng = random.Random(42)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        docs = self.corpus["documents"]
        contents = [d["content"] for d in docs]

        self._doc_embeddings = self.embedder.encode(
            contents, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )

        self._chroma_client = chromadb.EphemeralClient()
        self._collection = self._chroma_client.get_or_create_collection(
            name=f"rag_docs_{self.task.name}",
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for start in range(0, len(docs), batch_size):
            batch = docs[start: start + batch_size]
            emb_batch = self._doc_embeddings[start: start + batch_size]
            self._collection.add(
                ids=[str(d["doc_id"]) for d in batch],
                embeddings=emb_batch.tolist(),
                documents=[d["content"] for d in batch],
                metadatas=[
                    {
                        "source": d["source"],
                        "doc_type": d.get("doc_type", "unknown"),
                        "is_current": bool(d.get("is_current", True)),
                        # --- internal labels (server-side only) ---
                        "pattern": d.get("pattern", "unknown"),
                        "support_type": d.get("support_type", "unknown"),
                        "fact_group": d.get("fact_group", "policy"),
                        "topic_id": int(d.get("topic_id", -1)),
                    }
                    for d in batch
                ],
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        idx = self._query_order[self._episode_index % len(self._query_order)]
        self.current_query = self.queries[idx]
        self._episode_index += 1

        if self._episode_index % len(self._query_order) == 0:
            self._rng.shuffle(self._query_order)

        q_emb = self.embedder.encode(
            [self.current_query["query"]],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        results = self._collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=self.task.top_k,
            include=["documents", "distances", "metadatas", "embeddings"],
        )

        self.retrieved_chunks = []
        self._chunk_internal_meta = []
        retrieved_embs: List[List[float]] = []

        for i, (content, dist, meta, emb) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["embeddings"][0],
        )):
            similarity = max(0.0, 1.0 - dist)
            token_count = max(10, int(len(content.split()) * 1.3))

            # Public chunk — no internal labels
            chunk = ChunkInfo(
                chunk_id=i,
                content=content,
                similarity_score=round(similarity, 4),
                token_count=token_count,
                source=meta.get("source", "unknown"),
                doc_type=meta.get("doc_type", "unknown"),
                is_current=bool(meta.get("is_current", True)),
            )
            self.retrieved_chunks.append(chunk)
            retrieved_embs.append(emb)

            # Internal metadata stored server-side only
            self._chunk_internal_meta.append({
                "support_type": meta.get("support_type", "unknown"),
                "fact_group": meta.get("fact_group", "policy"),
                "pattern": meta.get("pattern", "unknown"),
            })

        self._retrieved_embeddings = np.array(retrieved_embs, dtype=np.float32)

        self.token_budget_remaining = self.task.token_budget
        self.episode_step = 0
        self.last_selected_indices = []
        self.current_selection = []
        self._last_answer_quality = None
        self._done = False

        return self._build_observation()

    def step(
        self, selected_chunk_indices: List[int], generated_answer: Optional[str] = None
    ) -> Tuple[Observation, Reward, bool, dict]:
        """
        Select a subset of retrieved chunks, generate an answer, and score it.
        The agent receives last_answer_quality in the next observation so it can
        reason about why its previous selection got the reward it did.
        """
        if self.current_query is None:
            raise RuntimeError("Call reset() before step().")

        valid_indices = sorted(
            set(i for i in selected_chunk_indices if 0 <= i < len(self.retrieved_chunks))
        )
        selected_chunks = [self.retrieved_chunks[i] for i in valid_indices]
        selected_metas = [self._chunk_internal_meta[i] for i in valid_indices]
        tokens_used = sum(c.token_count for c in selected_chunks)

        # ── Empty selection ────────────────────────────────────────────────────
        if not valid_indices:
            self.episode_step += 1
            self.current_selection = []
            self._last_answer_quality = 0.0
            self._done = self.episode_step >= self.task.max_steps
            reward = Reward(
                answer_quality=0.0, token_cost=0.0,
                evidence_precision=0.0, evidence_recall=0.0,
                diversity_bonus=0.0, minimality_bonus=0.0,
                contradiction_penalty=0.0, budget_penalty=0.0,
                shaping_bonus=0.0, total=0.0,
            )
            return self._build_observation(), reward, self._done, {
                "generated_answer": "", "tokens_used": 0,
                "num_selected": 0, "valid_indices": [],
                "note": "empty selection — reward is 0.0",
            }

        # ── Over-budget: hard stop ─────────────────────────────────────────────
        if tokens_used > self.token_budget_remaining:
            self.episode_step += 1
            self.current_selection = valid_indices
            self.last_selected_indices = valid_indices
            self._last_answer_quality = 0.0
            self._done = True
            reward = Reward(
                answer_quality=0.0, token_cost=-0.3,
                evidence_precision=0.0, evidence_recall=0.0,
                diversity_bonus=0.0, minimality_bonus=0.0,
                contradiction_penalty=0.0, budget_penalty=-0.3,
                shaping_bonus=0.0, total=0.0,
            )
            return self._build_observation(), reward, self._done, {
                "generated_answer": "", "tokens_used": tokens_used,
                "num_selected": len(selected_chunks), "valid_indices": valid_indices,
                "over_budget": True, "note": "selection exceeds remaining budget",
            }

        # ── Generate and judge answer ──────────────────────────────────────────
        chunk_texts = [c.content for c in selected_chunks]
        if generated_answer:
            # Answer was generated by inference.py via the LLM proxy — use it directly.
            # Quality is scored with semantic similarity (deterministic, no LLM in server).
            generated = generated_answer
            quality = _semantic_similarity(self.current_query["reference_answer"], generated)
        else:
            # Fallback: generate in server (e.g. direct API calls / local testing).
            try:
                generated = generate_answer(self.current_query["query"], chunk_texts)
            except Exception:
                generated = " ".join(chunk_texts)[:512]
            quality = judge_answer(
                self.current_query["query"],
                self.current_query["reference_answer"],
                generated,
            )

        # ── Token cost ────────────────────────────────────────────────────────────
        # Denominator is always the TOTAL budget (not remaining) so the cost of
        # selecting N tokens is identical on step 1 and step 3.  This prevents
        # a perverse incentive where spending tokens on a later step is cheaper.
        token_fraction = min(1.0, tokens_used / max(1, self.task.token_budget))
        token_cost = round(-(token_fraction * 0.25), 4)

        # ── Diversity bonus (server-side embedding comparison) ─────────────────
        diversity_bonus = 0.0
        sim_mat_sel: Optional[np.ndarray] = None
        if len(valid_indices) >= 2 and self._retrieved_embeddings is not None:
            sel_embs = self._retrieved_embeddings[valid_indices]
            norms = np.linalg.norm(sel_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = sel_embs / norms
            sim_mat_sel = normed @ normed.T
            k = len(valid_indices)
            off_diag = (sim_mat_sel.sum() - k) / max(1, k * (k - 1))
            diversity_bonus = round(0.1 * max(0.0, 1.0 - float(off_diag)), 4)

        # ── Evidence precision / recall (uses hidden internal metadata) ────────
        required_fact_groups = self.current_query.get("required_fact_groups", ["policy"])
        selected_fact_groups = {
            m["fact_group"] for m in selected_metas
            if m["support_type"] in {"gold", "support", "partial"}
        }
        covered_groups = selected_fact_groups.intersection(required_fact_groups)

        support_count = sum(
            1 for m in selected_metas if m["support_type"] in {"gold", "support", "partial"}
        )
        evidence_precision = round(support_count / max(1, len(selected_chunks)), 4)
        evidence_recall = round(len(covered_groups) / max(1, len(required_fact_groups)), 4)

        # ── Contradiction penalty (per bad chunk, using hidden labels) ─────────
        bad_count = sum(
            1 for m in selected_metas
            if m["support_type"] in {"contradiction", "stale", "adversarial"}
        )
        contradiction_penalty = round(-0.12 * bad_count, 4)

        # ── Near-duplicate penalty (embedding-based, no labels needed) ─────────
        # Flat −0.05 if ANY pair of selected chunks has cosine similarity > 0.92.
        # The penalty is flat (not per-pair) to create a simple binary signal:
        # "your selection contains at least one near-duplicate — fix that."
        redundancy_penalty = 0.0
        if sim_mat_sel is not None:
            k = len(valid_indices)
            for i in range(k):
                for j in range(i + 1, k):
                    if float(sim_mat_sel[i][j]) > 0.92:
                        redundancy_penalty = -0.05
                        break
                if redundancy_penalty != 0.0:
                    break

        # ── Minimality bonus ───────────────────────────────────────────────────
        minimal_target = len(required_fact_groups)
        minimality_bonus = 0.0
        if evidence_recall == 1.0 and len(selected_chunks) <= minimal_target:
            minimality_bonus = 0.1
        elif evidence_recall == 1.0 and len(selected_chunks) == minimal_target + 1:
            minimality_bonus = 0.04

        # ── Budget exhaustion penalty ──────────────────────────────────────────
        # Fires when the selection uses ALL remaining budget, leaving nothing for
        # subsequent steps.  The over-budget case (tokens_used > remaining) is
        # already caught above and terminates the episode with reward 0.
        budget_penalty = round(
            -0.08 if tokens_used >= self.token_budget_remaining else 0.0, 4
        )

        # ── Shaping bonus (informational only — not added to total) ───────────
        # Tracks whether the agent chose high-relevance chunks; exposed in the
        # Reward object so callers can inspect it, but excluded from the total
        # to keep the documented nine components exactly correct.
        high_rel = any(
            self.retrieved_chunks[i].similarity_score > 0.8 for i in valid_indices
        )
        shaping_bonus = round(0.04 if high_rel else 0.0, 4)

        # ── Final total, clamped [0.0, 1.0] ───────────────────────────────────
        # Nine components exactly as documented:
        #   answer_quality + evidence_precision + evidence_recall
        #   + diversity_bonus + minimality_bonus
        #   + contradiction_penalty + redundancy_penalty
        #   + budget_penalty + token_cost
        total = round(
            max(0.0, min(1.0,
                (0.55 * quality)
                + (0.15 * evidence_precision)
                + (0.10 * evidence_recall)
                + diversity_bonus
                + minimality_bonus
                + contradiction_penalty
                + redundancy_penalty
                + budget_penalty
                + token_cost,
            )),
            4,
        )

        reward = Reward(
            answer_quality=round(quality, 4),
            token_cost=token_cost,
            evidence_precision=evidence_precision,
            evidence_recall=evidence_recall,
            diversity_bonus=diversity_bonus,
            minimality_bonus=round(minimality_bonus, 4),
            contradiction_penalty=contradiction_penalty,
            budget_penalty=budget_penalty,
            shaping_bonus=shaping_bonus,
            total=total,
        )

        # ── Update state ───────────────────────────────────────────────────────
        self.token_budget_remaining = max(0, self.token_budget_remaining - tokens_used)
        self.episode_step += 1
        self.last_selected_indices = valid_indices
        self.current_selection = valid_indices
        self._last_answer_quality = quality  # will appear in next observation

        # ── Early-exit: reward efficient agents that nail the answer ──────────
        # Require at least two steps before the quality-based early exit can
        # fire. This keeps episode traces in the expected START, STEP, STEP,
        # END shape while still allowing strong agents to finish early.
        self._done = (
            self.episode_step >= self.task.max_steps
            or self.token_budget_remaining <= 0
            or (
                self.episode_step >= 2
                and evidence_recall == 1.0
                and quality >= 0.85
            )
        )

        info = {
            "generated_answer": generated,
            "tokens_used": tokens_used,
            "num_selected": len(selected_chunks),
            "valid_indices": valid_indices,
            "covered_fact_groups": sorted(covered_groups),
            "required_fact_groups": required_fact_groups,
        }

        return self._build_observation(), reward, self._done, info

    def get_state(self) -> dict:
        return {
            "task": self.task.name,
            "query": self.current_query["query"] if self.current_query else None,
            "retrieved_chunks": [c.model_dump() for c in self.retrieved_chunks],
            "last_selected_indices": self.last_selected_indices,
            "current_selection": self.current_selection,
            "selection_token_count": sum(
                self.retrieved_chunks[i].token_count
                for i in self.current_selection
                if 0 <= i < len(self.retrieved_chunks)
            ),
            "token_budget_remaining": self.token_budget_remaining,
            "episode_step": self.episode_step,
            "steps_remaining": max(0, self.task.max_steps - self.episode_step),
            "last_answer_quality": self._last_answer_quality,
            "max_steps": self.task.max_steps,
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_cross_similarity(self) -> List[List[float]]:
        if self._retrieved_embeddings is None or len(self._retrieved_embeddings) == 0:
            return []
        embs = self._retrieved_embeddings
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embs / norms
        sim_matrix = (normed @ normed.T).tolist()
        return [[round(v, 4) for v in row] for row in sim_matrix]

    def _build_observation(self) -> Observation:
        selection_token_count = sum(
            self.retrieved_chunks[i].token_count
            for i in self.current_selection
            if 0 <= i < len(self.retrieved_chunks)
        )
        return Observation(
            query=self.current_query["query"] if self.current_query else "",
            retrieved_chunks=self.retrieved_chunks,
            cross_similarity_matrix=self._compute_cross_similarity(),
            token_budget_remaining=self.token_budget_remaining,
            episode_step=self.episode_step,
            steps_remaining=max(0, self.task.max_steps - self.episode_step),
            task=self.task.name,
            current_selection=self.current_selection,
            selection_token_count=selection_token_count,
            candidate_token_costs=[c.token_count for c in self.retrieved_chunks],
            last_answer_quality=self._last_answer_quality,
        )
